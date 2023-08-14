#ifndef FTRL_FFM_FTRL_MODEL_H
#define FTRL_FFM_FTRL_MODEL_H

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "model/ftrl_unit.h"
#include "utils/types.h"

namespace ftrl {

class FtrlModel {
 public:
  FtrlModel(float _mean, float _stddev, std::string _model_type);
  FtrlModel(float _mean, float _stddev, int _n_factors, std::string _model_type);
  FtrlModel(float _mean, float _stddev, int _n_factors, int _n_fields, std::string _model_type);

  std::shared_ptr<ftrl_model_unit> &get_or_init_weight(int index);
  std::shared_ptr<ftrl_model_unit> &get_or_init_bias();

  float predict(const feat_vec &feats, bool sigmoid);

  float compute_logit(const feat_vec &feats, bool update_model);

  float train(const feat_vec &feats, int label, float w_alpha, float w_beta, float w_l1,
              float w_l2);

  void output_model(std::ofstream &ofs);
  [[maybe_unused]] void debug_print_model();
  bool load_model(std::ifstream &ifs);

 private:
  std::string model_type;
  int n_factors{1};
  int n_fields{1};
  float init_mean;
  float init_stddev;
  std::vector<float> sum_vx;
  std::shared_ptr<ftrl_model_unit> model_bias;
  std::unordered_map<int, std::shared_ptr<ftrl_model_unit>> model_weight;
  std::mutex weight_mutex;
  std::mutex bias_mutex;
};

}  // namespace ftrl

#endif  // FTRL_FFM_FTRL_MODEL_H