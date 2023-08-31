#ifndef FTRL_FFM_FTRL_MODEL_H
#define FTRL_FFM_FTRL_MODEL_H

#include <cmath>
#include <mutex>
#include <vector>

#include "utils/cmd_option.h"
#include "utils/types.h"
#include "utils/utils.h"

namespace ftrl {

class FtrlModel {
 public:
  explicit FtrlModel(const config_options &opt);
  virtual ~FtrlModel() = default;
  virtual float train(feat_vec &features, int label) = 0;
  virtual float predict(feat_vec &features, bool output_prob) = 0;

  float compute_linear_logit(const feat_vec &features);
  void update_linear_w(const feat_vec &features);
  void update_bias();
  void update_linear_nz(const feat_vec &features, float tmp_grad);
  void update_bias_nz(float tmp_grad);
  virtual void remove_out_range(feat_vec &feats);

  template <typename T>
  inline T maybe_zero_weight(T n, T z) {
    return std::fabs(z) <= w_l1
               ? 0.0
               : -1.0 * (z - utils::sgn(z) * w_l1) / (w_l2 + (w_beta + std::sqrt(n)) / w_alpha);
  }

  void output_model(std::ofstream &ofs);
  [[maybe_unused]] void debug_print_model();
  bool load_model(std::ifstream &ifs);

  ModelType model_type;

 protected:
  int n_feats;
  float w_alpha;
  float w_beta;
  float w_l1;
  float w_l2;
  float bias;
  float bias_n;
  float bias_z;
  std::vector<float> lin_w;
  std::vector<float> lin_w_n;
  std::vector<float> lin_w_z;
  std::vector<std::mutex> lin_w_mutex;
  std::mutex bias_mutex;
};

}  // namespace ftrl

#endif  // FTRL_FFM_FTRL_MODEL_H
