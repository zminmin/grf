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

#include <map>
#include <unordered_set>
#include <fstream>
#include "commons/DefaultData.h"
#include "commons/utility.h"
#include "prediction/CustomPredictionStrategy.h"

#include "catch.hpp"

using namespace grf;

TEST_CASE("first test", "[custom, prediction]") {

    std::vector<double> observations = {3.8771, 3.2724, 2.9853, 3.2818, 2.4600, 2.9764, 2.7263, 3.4246, 3.2192, 1.9851, 0.7713, 0.0208, 0.6336, 0.7488, 0.4985, 0.2248, 0.1981, 0.7605, 0.1691, 0.0883, 0.6854, 0.9534, 0.0039, 0.5122, 0.8126, 0.6125, 0.7218, 0.2919, 0.9178, 0.7146, 0.5425, 0.1422, 0.3733, 0.6741, 0.4418, 0.4340, 0.6178, 0.5131, 0.6504, 0.6010, 0.8052, 0.5216, 0.9086, 0.3192, 0.0905, 0.3007, 0.1140, 0.8287, 0.0469, 0.6263, 0.5476, 0.8193, 0.1989, 0.8569, 0.3517, 0.7546, 0.2960, 0.8839, 0.3255, 0.1650, 0.3925, 0.0935, 0.8211, 0.1512, 0.3841, 0.9443, 0.9876, 0.4563, 0.8261, 0.2514};

	DefaultData data(observations, 10, 7);

	data.set_outcome_index(0);

	data.set_expe_1_index(1);
	data.set_expe_2_index(2);
	data.set_expe_3_index(3);
	data.set_fami_1_index(4);
	data.set_fami_2_index(5);
	data.set_fami_3_index(6);

	std::unordered_map<size_t, double> weights_by_sample = {
	  {0, 0.0}, {1, 0.1}, {2, 0.2}, {3, 0.1}, {4, 0.1},
	  {5, 0.1}, {6, 0.2}, {7, 0.1}, {8, 0.0}, {9, 0.1}};


	CustomPredictionStrategy prediction_strategy;
	std::vector<double> first_prediction = prediction_strategy.predict(0, weights_by_sample, data, data);

	for (size_t i = 0; i < first_prediction.size(); ++i) {
        double first_outcome = first_prediction[i];
        std::cout << first_outcome << std::endl;
   	}
}

// TEST_CASE("regression variance estimates are positive", "[regression, prediction]") {
//   std::vector<double> averages = {1.12};
//   std::vector<std::vector<double>> leaf_values = {{3.2}, {4.5}, {6.7}, {-3.5}};

//   RegressionPredictionStrategy prediction_strategy;
//   std::vector<double> variance = prediction_strategy.compute_variance(
//       averages, PredictionValues(leaf_values, 1), 2);

//   REQUIRE(variance.size() == 1);
//   REQUIRE(variance[0] > 0);
// }

// TEST_CASE("scaling outcome scales regression variance", "[regression, prediction]") {
//   std::vector<double> averages = {2.725};
//   std::vector<std::vector<double>> leaf_values = {{3.2}, {4.5}, {6.7}, {-3.5}};

//   std::vector<double> scaled_average = {5.45};
//   std::vector<std::vector<double>> scaled_leaf_values = {{6.4}, {9.0}, {13.4}, {-7.0}};

//   RegressionPredictionStrategy prediction_strategy;
//   std::vector<double> first_variance = prediction_strategy.compute_variance(
//       averages,
//       PredictionValues(leaf_values, 1)
//       , 2);
//   std::vector<double> second_variance = prediction_strategy.compute_variance(
//       scaled_average,
//       PredictionValues(scaled_leaf_values, 1), 2);

//   REQUIRE(first_variance.size() == 1);
//   REQUIRE(second_variance.size() == 1);
//   REQUIRE(equal_doubles(first_variance[0], second_variance[0] / 4, 1.0e-10));
// }


// TEST_CASE("debiased errors are smaller than raw errors", "[regression, prediction]") {
//   std::vector<double> average = {2.725};
//   std::vector<std::vector<double>> leaf_values = {{3.2}, {4.5}, {6.7}, {-3.5}};

//   std::vector<double> outcomes = {6.4, 9.0, 13.4, -7.0};
//   DefaultData data(outcomes, 4, 1);
//   data.set_outcome_index(0);

//   RegressionPredictionStrategy prediction_strategy;

//   for (size_t sample=0; sample < 4; ++sample) {
//     auto error = prediction_strategy.compute_error(
//           sample,
//           average,
//           PredictionValues(leaf_values, 1),
//           data).at(0);
//     double debiased_error = error.first;

//     // Raw error
//     double outcome = data.get_outcome(sample);
//     double raw_error = average.at(0) - outcome;
//     double mse = raw_error * raw_error;

//     REQUIRE(debiased_error < mse);
//   }
// }
