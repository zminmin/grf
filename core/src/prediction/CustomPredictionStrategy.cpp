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

#include "CustomPredictionStrategy.h"

namespace grf {

size_t CustomPredictionStrategy::prediction_length() const {
  return 3;
}

std::vector<double> CustomPredictionStrategy::predict(size_t sample,
    const std::unordered_map<size_t, double>& weights_by_sample,
    const Data& train_data,
    const Data& data) const {

  size_t num_variables = 6;
  size_t num_nonzero_weights = weights_by_sample.size();
  size_t num_lambdas = 3;

  // Creating a vector of neighbor weights weights
  // Weights by sample ID contains pairs [sample ID, weight for test point]
  // Weights vec is a vector of weights indexed by their corresponding sample ID
  // For example:
  // weights_by_sampleID = [(0, 0.04), (1, 0f.20), (3, 0.01), (4, 0.05), ...]
  // weights_vec = [0.04, 0.20, 0.0, 0.01, 0.05, ....]

  std::vector<size_t> indices(num_nonzero_weights);
  Eigen::MatrixXd weights_vec = Eigen::VectorXd::Zero(num_nonzero_weights);
  {
    size_t i = 0;
    for (const auto& it : weights_by_sample) {
      size_t index = it.first;
      double weight = it.second;
      indices[i] = index;
      weights_vec(i) = weight;
      i++;
    }
  }

  // The matrix X consists of differences of linear correction variables from their target.
  // Only observations with nonzero weights need to be filled.
  // For example, if there are K+1 linear correction variables, and m nonzero weights,
  //  then X will be:
  //    1.   (X[0,0] - x[0])   ...  
  //    1.   (X[1,0] - x[0])   ...   
  //    1.   (X[3,0] - x[0])   ...   # Observation 2 is skipped due to zero weights

  size_t dim_X = num_variables + 1;
  Eigen::MatrixXd X (num_nonzero_weights, dim_X);
  Eigen::MatrixXd Y (num_nonzero_weights, 1);

  for (size_t i = 0; i < num_nonzero_weights; ++i) {
    // Index of next neighbor with nonzero weights
    size_t index = indices[i];
    // double treatment = train_data.get_treatment(index);

    // Intercept
    X(i, 0) = 1.0;

    X(i, 1) = train_data.get_expe_1(index);
    X(i, 2) = train_data.get_expe_2(index);
    X(i, 3) = train_data.get_expe_3(index);
    X(i, 4) = train_data.get_fami_1(index);
    X(i, 5) = train_data.get_fami_2(index);
    X(i, 6) = train_data.get_fami_3(index);

    Y(i) = train_data.get_outcome(index);
  }

  // find regression predictions
  Eigen::MatrixXd M_unpenalized (dim_X, dim_X);
  M_unpenalized.noalias() = X.transpose() * weights_vec.asDiagonal() * X;

  std::vector<double> predictions(num_lambdas);

  Eigen::MatrixXd local_coefficients = M_unpenalized.ldlt().solve(X.transpose()*weights_vec.asDiagonal()*Y);

  predictions[0] = local_coefficients(4);
  predictions[1] = local_coefficients(5);
  predictions[2] = local_coefficients(6);


  return predictions;
}

std::vector<double> CustomPredictionStrategy::compute_variance(
    size_t sample,
    const std::vector<std::vector<size_t>>& samples_by_tree,
    const std::unordered_map<size_t, double>& weights_by_sampleID,
    const Data& train_data,
    const Data& data,
    size_t ci_group_size) const {


  size_t num_variables = 6;
  size_t num_nonzero_weights = weights_by_sampleID.size();
  size_t num_lambdas = 3;

  std::vector<size_t> sample_index_map(train_data.get_num_rows());
  std::vector<size_t> indices(num_nonzero_weights);
  Eigen::MatrixXd weights_vec = Eigen::VectorXd::Zero(num_nonzero_weights);
  {
    size_t i = 0;
    for (const auto& it : weights_by_sampleID) {
      size_t index = it.first;
      double weight = it.second;
      indices[i] = index;
      sample_index_map[index] = i;
      weights_vec(i) = weight;
      i++;
    }
  }

  size_t dim_X = num_variables + 1;
  Eigen::MatrixXd X (num_nonzero_weights, dim_X);
  Eigen::MatrixXd Y (num_nonzero_weights, 1);

  for (size_t i = 0; i < num_nonzero_weights; ++i) {
    // Index of next neighbor with nonzero weights
    size_t index = indices[i];
    // double treatment = train_data.get_treatment(index);

    // Intercept
    X(i, 0) = 1.0;

    X(i, 1) = train_data.get_expe_1(index);
    X(i, 2) = train_data.get_expe_2(index);
    X(i, 3) = train_data.get_expe_3(index);
    X(i, 4) = train_data.get_fami_1(index);
    X(i, 5) = train_data.get_fami_2(index);
    X(i, 6) = train_data.get_fami_3(index);

    Y(i) = train_data.get_outcome(index);
  }

  // find regression predictions
  Eigen::MatrixXd M_unpenalized (dim_X, dim_X);
  M_unpenalized.noalias() = X.transpose() * weights_vec.asDiagonal() * X;

  std::vector<double> predictions(num_lambdas);

  Eigen::MatrixXd theta = M_unpenalized.ldlt().solve(X.transpose()*weights_vec.asDiagonal()*Y);

  size_t f1_index = 4;
  size_t f2_index = 5;
  size_t f3_index = 6;

  Eigen::VectorXd e_trt_f1 = Eigen::VectorXd::Zero(dim_X);
  Eigen::VectorXd e_trt_f2 = Eigen::VectorXd::Zero(dim_X);
  Eigen::VectorXd e_trt_f3 = Eigen::VectorXd::Zero(dim_X);
  e_trt_f1(f1_index) = 1.0;
  e_trt_f2(f2_index) = 1.0;
  e_trt_f3(f3_index) = 1.0;
  Eigen::VectorXd zeta_f1 = M_unpenalized.ldlt().solve(e_trt_f1);
  Eigen::VectorXd zeta_f2 = M_unpenalized.ldlt().solve(e_trt_f2);
  Eigen::VectorXd zeta_f3 = M_unpenalized.ldlt().solve(e_trt_f3);


  Eigen::VectorXd X_times_zeta_f1 = X * zeta_f1;
  Eigen::VectorXd X_times_zeta_f2 = X * zeta_f2;
  Eigen::VectorXd X_times_zeta_f3 = X * zeta_f3;

  Eigen::VectorXd local_prediction = X * theta;
  Eigen::VectorXd pseudo_residual_f1 = Eigen::VectorXd::Zero(num_nonzero_weights);
  Eigen::VectorXd pseudo_residual_f2 = Eigen::VectorXd::Zero(num_nonzero_weights);
  Eigen::VectorXd pseudo_residual_f3 = Eigen::VectorXd::Zero(num_nonzero_weights);

  for (size_t i = 0; i < num_nonzero_weights; i++) {
    pseudo_residual_f1(i) = X_times_zeta_f1(i) * (Y(i) - local_prediction(i));
    pseudo_residual_f2(i) = X_times_zeta_f2(i) * (Y(i) - local_prediction(i));
    pseudo_residual_f3(i) = X_times_zeta_f3(i) * (Y(i) - local_prediction(i));
  }

  double num_good_groups = 0;
  double psi_squared_f1 = 0;
  double psi_squared_f2 = 0;
  double psi_squared_f3 = 0;
  double psi_grouped_squared_f1 = 0;
  double psi_grouped_squared_f2 = 0;
  double psi_grouped_squared_f3 = 0;

  double avg_score_f1 = 0;
  double avg_score_f2 = 0;
  double avg_score_f3 = 0;

  for (size_t group = 0; group < samples_by_tree.size() / ci_group_size; ++group) {
    bool good_group = true;
    for (size_t j = 0; j < ci_group_size; ++j) {
      if (samples_by_tree[group * ci_group_size + j].size() == 0) {
        good_group = false;
      }
    }
    if (!good_group) continue;

    num_good_groups++;

    double group_psi_f1 = 0;
    double group_psi_f2 = 0;
    double group_psi_f3 = 0;

    for (size_t j = 0; j < ci_group_size; ++j) {
      size_t b = group * ci_group_size + j;
      double psi_1 = 0;
      double psi_2 = 0;
      double psi_3 = 0;
      for (size_t sample : samples_by_tree[b]) {
        psi_1 += pseudo_residual_f1(sample_index_map[sample]);
        psi_2 += pseudo_residual_f2(sample_index_map[sample]);
        psi_3 += pseudo_residual_f3(sample_index_map[sample]);
      }
      psi_1 /= samples_by_tree[b].size();
      psi_2 /= samples_by_tree[b].size();
      psi_3 /= samples_by_tree[b].size();
      psi_squared_f1 += psi_1 * psi_1;
      psi_squared_f2 += psi_2 * psi_2;
      psi_squared_f3 += psi_3 * psi_3;
      group_psi_f1 += psi_1;
      group_psi_f2 += psi_2;
      group_psi_f3 += psi_3;
    }

    group_psi_f1 /= ci_group_size;
    group_psi_f2 /= ci_group_size;
    group_psi_f3 /= ci_group_size;
    psi_grouped_squared_f1 += group_psi_f1 * group_psi_f1;
    psi_grouped_squared_f2 += group_psi_f2 * group_psi_f2;
    psi_grouped_squared_f3 += group_psi_f3 * group_psi_f3;

    avg_score_f1 += group_psi_f1;
    avg_score_f2 += group_psi_f2;
    avg_score_f3 += group_psi_f3;
  }

  avg_score_f1 /= num_good_groups;
  avg_score_f2 /= num_good_groups;
  avg_score_f3 /= num_good_groups;

  double var_between_f1 = psi_grouped_squared_f1 / num_good_groups - avg_score_f1 * avg_score_f1;
  double var_between_f2 = psi_grouped_squared_f2 / num_good_groups - avg_score_f2 * avg_score_f2;
  double var_between_f3 = psi_grouped_squared_f3 / num_good_groups - avg_score_f3 * avg_score_f3;
  double var_total_f1 = psi_squared_f1 / (num_good_groups * ci_group_size) - avg_score_f1 * avg_score_f1;
  double var_total_f2 = psi_squared_f2 / (num_good_groups * ci_group_size) - avg_score_f2 * avg_score_f2;
  double var_total_f3 = psi_squared_f3 / (num_good_groups * ci_group_size) - avg_score_f3 * avg_score_f3;

  // This is the amount by which var_between is inflated due to using small groups
  double group_noise_f1 = (var_total_f1 - var_between_f1) / (ci_group_size - 1);
  double group_noise_f2 = (var_total_f2 - var_between_f2) / (ci_group_size - 1);
  double group_noise_f3 = (var_total_f3 - var_between_f3) / (ci_group_size - 1);

  // A simple variance correction, would be to use:
  // var_debiased = var_between - group_noise.
  // However, this may be biased in small samples; we do an objective
  // Bayes analysis of variance instead to avoid negative values.
  double var_debiased_f1 = bayes_debiaser.debias(var_between_f1, group_noise_f1, num_good_groups);
  double var_debiased_f2 = bayes_debiaser.debias(var_between_f2, group_noise_f2, num_good_groups);
  double var_debiased_f3 = bayes_debiaser.debias(var_between_f3, group_noise_f3, num_good_groups);

  return { var_debiased_f1, var_debiased_f2, var_debiased_f3 };
}

} // namespace grf
