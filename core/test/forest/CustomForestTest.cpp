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

#include "commons/utility.h"
#include "forest/ForestPredictor.h"
#include "forest/ForestPredictors.h"
#include "forest/ForestTrainer.h"
#include "forest/ForestTrainers.h"
#include "utilities/ForestTestUtilities.h"

#include "catch.hpp"

using namespace grf;

TEST_CASE("custom forests predict 0 by default", "[custom, forest]") {
  // Train an honest custom forest.
  // std::unique_ptr<Data> data = load_data("test/forest/resources/simple_case.csv");
  std::unique_ptr<Data> data = load_data("/Users/zhangminmin/Desktop/Document/D/Code/grf/core/test/forest/resources/simple_case.csv");
    
  data->set_outcome_index(6);
  data->set_expe_1_index(7);
  data->set_expe_2_index(8);
  data->set_expe_3_index(9);
  data->set_fami_1_index(10);
  data->set_fami_2_index(11);
  data->set_fami_3_index(12);


//  double e1 = data->get_expe_1(0);
//  std::cout << e1 << std::endl;

  std::vector<double> overall_beta = {-0.32, 0.99, 1.02, 1.00, 1.02, 0.52, 0.99};
  size_t ll_split_cutoff = 30;


  ForestTrainer trainer = custom_trainer(overall_beta, ll_split_cutoff);
  ForestOptions options = ForestTestUtilities::default_honest_options();
  Forest forest = trainer.train(*data, options);

  // Predict on the same data.
  ForestPredictor predictor = custom_predictor(3);
  std::vector<Prediction> predictions = predictor.predict_oob(forest, *data, false);

  // Check the dummy predictions look as expected.
  REQUIRE(predictions.size() == data->get_num_rows());

  int i = 0;
  for (const Prediction& prediction : predictions) {
      double value1 = prediction.get_predictions()[0];
      double value2 = prediction.get_predictions()[1];
      double value3 = prediction.get_predictions()[2];
     if(i < 10) {
         std::cout << value1 << " " << value2 << " " << value3 << std::endl;
     }
     i++;
   }
}
