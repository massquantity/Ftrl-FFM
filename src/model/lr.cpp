#include "model/lr.h"

#include "compression/compress.h"

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

void LR::save_compressed_model(std::string_view file_name, int compress_level) {
  std::vector<float> weights{bias};
  weights.insert(weights.end(), lin_w.begin(), lin_w.end());
  const size_t weight_size = weights.size() * sizeof(float);
  compress_weights(weights.data(), weight_size, file_name, compress_level);
}

void LR::load_compressed_model(std::string_view file_name) {
  float *buffer_ptr = decompress_weights(file_name);
  bias = buffer_ptr[0];
  float *offset = buffer_ptr + 1;
  std::move(offset, offset + lin_w.size(), lin_w.data());
  free(buffer_ptr);
}

}  // namespace ftrl
