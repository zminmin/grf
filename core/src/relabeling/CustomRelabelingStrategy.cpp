/*-------------------------------------------------------------------------------
  This file is part of generalized random forest (grf).

  grf is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  grf is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with grf. If not, see <http://www.gnu.org/licenses/>.
 #-------------------------------------------------------------------------------*/

#include "CustomRelabelingStrategy.h"

namespace grf {


CustomRelabelingStrategy::CustomRelabelingStrategy(const std::vector<double>& overall_beta,
                                                  size_t ll_split_cutoff):
  overall_beta(overall_beta),
  ll_split_cutoff(ll_split_cutoff){
};


bool CustomRelabelingStrategy::relabel(
    const std::vector<size_t>& samples,
    const Data& data,
    std::vector<double>& responses_by_sample) const {

  size_t num_variables = 6;
  size_t num_data_points = samples.size();

  
  Eigen::MatrixXd tempW (num_data_points, num_variables + 1);
  Eigen::MatrixXd Y (num_data_points, 1);
  double barY = 0.0;
  Eigen::MatrixXd W (num_data_points, num_variables);
  Eigen::MatrixXd barW (1, num_variables);

  for (size_t j = 0; j < num_variables; ++j) {
      barW(0, j) = 0.0;
  }
  for (size_t i = 0; i < num_data_points; ++i) { 
    Y(i) = data.get_outcome(samples[i]);
    tempW(i, 0) = 1.0;
    tempW(i, 1) = data.get_expe_1(samples[i]);
    tempW(i, 2) = data.get_expe_2(samples[i]);
    tempW(i, 3) = data.get_expe_3(samples[i]);
    tempW(i, 4) = data.get_fami_1(samples[i]);
    tempW(i, 5) = data.get_fami_2(samples[i]);
    tempW(i, 6) = data.get_fami_3(samples[i]);

    W(i, 0) = tempW(i, 1);
    W(i, 1) = tempW(i, 2);
    W(i, 2) = tempW(i, 3);
    W(i, 3) = tempW(i, 4);
    W(i, 4) = tempW(i, 5);
    W(i, 5) = tempW(i, 6);

    for (size_t j = 0; j < num_variables; ++j) {
      barW(0, j) += W(i, j);
    }
    barY += Y(i);
  }

  for (size_t j = 0; j < num_variables; ++j) {
    barW(0, j) = barW(0, j) / num_data_points;
  }
  barY = barY / num_data_points;

  // for (size_t i = 0; i < num_data_points; ++i) {
  //   for (size_t j = 0; j < num_variables; ++j) {
  //     std::cout << W(i,j) << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;

  
  Eigen::MatrixXd leaf_predictions (num_data_points, 1);
  Eigen::MatrixXd average_treatment (1, 1);

  // minimal size of sample in the node to run regression;
  // otherwise use the overall beta
  if (num_data_points < ll_split_cutoff) {
    // use overall beta for regression predictions
    Eigen::MatrixXd eigen_beta (num_variables, 1);
    for(size_t j = 0; j < num_variables; ++j) {
      eigen_beta(j, 0) = overall_beta[j+1];
    }
    leaf_predictions = W * eigen_beta;
    average_treatment = barW * eigen_beta;
  } 
  else {
    // find regression predictions
    Eigen::MatrixXd M(num_variables + 1, num_variables + 1);
    M.noalias() = tempW.transpose() * tempW;
    Eigen::MatrixXd local_coefficients = M.ldlt().solve(tempW.transpose() * Y);
    Eigen::MatrixXd eigen_beta (num_variables, 1);
    for(size_t j = 0; j < num_variables; ++j) {
      eigen_beta(j, 0) = local_coefficients(j+1, 0);
      // std::cout << eigen_beta(j, 0) << " ";
    }
    leaf_predictions = W * eigen_beta;
    average_treatment = barW * eigen_beta;
  }

  // for (size_t i = 0; i < num_data_points; ++i) {
  //   std::cout << leaf_predictions(i) << " ";
  // }
  // std::cout << std::endl;


  Eigen::MatrixXd Ap (num_variables, num_variables);
  Eigen::MatrixXd Wi (num_variables, 1);
  for (size_t i = 0; i < num_variables; ++i) {
    for (size_t j = 0; j < num_variables; ++j) {
      Ap(i, j) = 0.0;
    }
  }
  for (size_t i = 0; i < num_data_points; ++i) {
    for (size_t j = 0; j < num_variables; ++j) {
      Wi(j, 0) = W(i, j);
    }
    Ap += (Wi - barW.transpose()) * (Wi.transpose() - barW);
  }
  Ap = Ap / num_data_points;

  // std::cout << barY << std::endl;
  // for (size_t i = 0; i < num_variables; ++i) {
  //   std::cout << barW(0,i) << " ";
  // }
  // std::cout << std::endl;
  // for (size_t i = 0; i < num_variables; ++i) {
  //   for (size_t j = 0; j < num_variables; ++j) {
  //     std::cout << Ap(i,j) << " ";
  //   }
  //   std::cout << std::endl;
  // }

  Eigen::MatrixXd myEps (1, num_variables);
  myEps(0, 0) = 0.0; myEps(0, 1) = 0.0; myEps(0, 2) = 0.0;
  myEps(0, 3) = 1.0; myEps(0, 4) = 1.0; myEps(0, 5) = 1.0;

  Eigen::MatrixXd EpsApinv (1, num_variables);
  EpsApinv = myEps * Ap.inverse();

  size_t i = 0;
  for (size_t sample : samples) {
    double commonP = 0.0;
    for (size_t j = 0; j < num_variables; ++j) {
      commonP += EpsApinv(0, j) * (W(i, j) - barW(0, j));
    }
    // pay attention: the previous design has Y(i,0) as Y(sample), which leads to error in access (index out of range)
    double residual =  commonP * (Y(i, 0) - barY - leaf_predictions(i, 0) + average_treatment(0, 0));
    responses_by_sample[sample] = residual;
    i++;
  }

  return false;
}

} // namespace grf
