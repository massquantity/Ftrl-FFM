#include "model/lr.h"

namespace ftrl {

LR::LR(const config_options &opt) : FtrlModel(opt) {}

float LR::train(feat_vec &features, int label) {
  remove_out_range(features);
  update_linear_w(features);
  update_bias();
  const float logit = compute_linear_logit(features);
  const float tmp_grad = utils::sigmoid(logit) - static_cast<float>(label);
  update_linear_nz(features, tmp_grad);
  update_bias_nz(tmp_grad);
  return logit;
}

float LR::predict(feat_vec &features, bool output_prob) {
  remove_out_range(features);
  const float logit = compute_linear_logit(features);
  return output_prob ? utils::sigmoid(logit) : logit;
}

}  // namespace ftrl
