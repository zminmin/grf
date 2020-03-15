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

#include <fstream>
#include <map>
#include <memory>
#include <unordered_set>

#include "catch.hpp"
#include "commons/utility.h"
#include "relabeling/RelabelingStrategy.h"
#include "relabeling/CustomRelabelingStrategy.h"

using namespace grf;

std::vector<double> get_relabeled_outcomes_custom(std::vector<double> observations, size_t num_samples) {
  DefaultData data(observations, num_samples, 7);
  
  data.set_outcome_index(0);

  data.set_expe_1_index(1);
  data.set_expe_2_index(2);
  data.set_expe_3_index(3);
  data.set_fami_1_index(4);
  data.set_fami_2_index(5);
  data.set_fami_3_index(6);

  
  std::vector<size_t> samples;
  for (size_t i = 0; i < num_samples; ++i) {
    samples.push_back(i);
  }

  std::vector<double> overall_beta = {1,2,3,4,5,6,7};
  size_t ll_split_cutoff = 0;

  std::unique_ptr<RelabelingStrategy> relabeling_strategy(new CustomRelabelingStrategy(overall_beta, ll_split_cutoff));

  std::vector<double> relabeled_observations(num_samples);
  bool stop = relabeling_strategy->relabel(samples, data, relabeled_observations);
  if (stop) {
    return std::vector<double>();
  }

  std::vector<double> relabeled_outcomes;
  relabeled_outcomes.reserve(samples.size());
  for (auto& sample : samples) {
    relabeled_outcomes.push_back(relabeled_observations.at(sample));
  }
  return relabeled_outcomes;
}



TEST_CASE("manually calculated result", "[custom, relabeling]") {


    std::vector<double> observations = {3.8771, 3.2724, 2.9853, 3.2818, 2.4600, 2.9764, 2.7263, 3.4246, 3.2192, 1.9851, 0.7713, 0.0208, 0.6336, 0.7488, 0.4985, 0.2248, 0.1981, 0.7605, 0.1691, 0.0883, 0.6854, 0.9534, 0.0039, 0.5122, 0.8126, 0.6125, 0.7218, 0.2919, 0.9178, 0.7146, 0.5425, 0.1422, 0.3733, 0.6741, 0.4418, 0.4340, 0.6178, 0.5131, 0.6504, 0.6010, 0.8052, 0.5216, 0.9086, 0.3192, 0.0905, 0.3007, 0.1140, 0.8287, 0.0469, 0.6263, 0.5476, 0.8193, 0.1989, 0.8569, 0.3517, 0.7546, 0.2960, 0.8839, 0.3255, 0.1650, 0.3925, 0.0935, 0.8211, 0.1512, 0.3841, 0.9443, 0.9876, 0.4563, 0.8261, 0.2514};

    std::vector<double> first_outcomes = get_relabeled_outcomes_custom(observations, 10);

    for (size_t i = 0; i < first_outcomes.size(); ++i) {
        double first_outcome = first_outcomes[i];
        std::cout << first_outcome << std::endl;
   }
}




// TEST_CASE("flipping signs of treatment does not affect relabeled outcomes", "[instrumental, relabeling]") {
//   std::vector<double> observations = {
//       -9.99984, -7.36924, 5.11211, -0.826997, 0.655345, -5.62082, -9.05911, 3.57729, 3.58593, 8.69386, // outcomes
//       1, 0, 0, 0, 1, 0, 1, 0, 0, 0, // treatment
//       0, 0, 1, 1, 1, 0, 1, 0, 1, 0 }; // instrument

//   std::vector<double> flipped_observations = {
//       -9.99984, -7.36924, 5.11211, -0.826997, 0.655345, -5.62082, -9.05911, 3.57729, 3.58593, 8.69386, // outcomes
//       0, 1, 1, 1, 0, 1, 0, 1, 1, 1, // treatment
//       0, 0, 1, 1, 1, 0, 1, 0, 1, 0 }; // instrument

//   std::vector<double> first_outcomes = get_relabeled_outcomes_custom(observations, 10);
//   std::vector<double> second_outcomes = get_relabeled_outcomes_custom(flipped_observations, 10);

//   REQUIRE(first_outcomes.size() == second_outcomes.size());
//   for (size_t i = 0; i < first_outcomes.size(); ++i) {
//     double first_outcome = first_outcomes[i];
//     double second_outcome = second_outcomes[i];

//     REQUIRE(equal_doubles(first_outcome, second_outcome, 1.0e-10));
//   }
// }

// TEST_CASE("scaling instrument scales relabeled outcomes", "[instrumental, relabeling]") {
//   std::vector<double> outcomes = { };
//   std::vector<double> treatment = {1, 0, 0, 0, 1, 0, 1, 0, 0, 0};
//   std::vector<double> instrument = {0, 0, 1, 1, 1, 0, 1, 0, 1, 0};
//   std::vector<double> scaled_instrument = {0, 0, 3, 3, 3, 0, 3, 0, 3, 0};

//   std::vector<double> observations = {
//       -9.99984, -7.36924, 5.11211, -0.826997, 0.655345, -5.62082, -9.05911, 3.57729, 3.58593, 8.69386, // outcomes
//       1, 0, 0, 0, 1, 0, 1, 0, 0, 0, // treatment
//       0, 0, 1, 1, 1, 0, 1, 0, 1, 0 }; // instrument

//   std::vector<double> scaled_observations = {
//       -9.99984, -7.36924, 5.11211, -0.826997, 0.655345, -5.62082, -9.05911, 3.57729, 3.58593, 8.69386, // outcomes
//       1, 0, 0, 0, 1, 0, 1, 0, 0, 0, // treatment
//       0, 0, 3, 3, 3, 0, 3, 0, 3, 0 }; // scaled instrument

//   std::vector<double> first_outcomes = get_relabeled_outcomes_custom(observations, 10);
//   std::vector<double> second_outcomes = get_relabeled_outcomes_custom(scaled_observations, 10);

//   REQUIRE(first_outcomes.size() == second_outcomes.size());
//   for (size_t i = 0; i < first_outcomes.size(); ++i) {
//     double first_outcome = first_outcomes[i];
//     double second_outcome = second_outcomes[i];

//     REQUIRE(equal_doubles(3 * first_outcome, second_outcome, 1.0e-10));
//   }
// }

// TEST_CASE("constant treatment leads to no splitting", "[instrumental, relabeling]") {
//   std::vector<double> observations = {
//       -9.99984, -7.36924, 5.11211, -0.826997, 0.655345, -5.62082, -9.05911, 3.57729, 3.58593, 8.69386, // outcomes
//       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // treatment
//       0, 0, 1, 1, 1, 0, 1, 0, 1, 0 }; // instrument

//   std::vector<double> relabeled_outcomes = get_relabeled_outcomes_custom(observations, 10);
//   REQUIRE(relabeled_outcomes.empty()); // An empty map signals that no splitting should be performed.
// }

// TEST_CASE("constant instrument leads to no splitting", "[instrumental, relabeling]") {
//   std::vector<double> observations = {
//       -9.99984, -7.36924, 5.11211, -0.826997, 0.655345, -5.62082, -9.05911, 3.57729, 3.58593, 8.69386, // outcomes
//       0, 0, 1, 1, 0, 0, 1, 0, 1, 0, // treatment
//       1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }; // instrument

//   std::vector<double> relabeled_outcomes = get_relabeled_outcomes_custom(observations, 10);
//   REQUIRE(relabeled_outcomes.empty()); // An empty map signals that no splitting should be performed.
// }
