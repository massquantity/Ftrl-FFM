#include "model/fm.h"

#include <utility>

#include "utils/utils.h"

namespace ftrl {

FM::FM(const config_options &opt) : FtrlModel(opt), n_factors(opt.n_factors) {
  sum_vx.resize(n_factors);
  vec_w = std::move(utils::init_weights(n_feats, n_factors, opt.init_mean, opt.init_stddev));
  vec_w_n.resize(n_feats);
  vec_w_z.resize(n_feats);
  for (int i = 0; i < n_feats; i++) {
    vec_w_n[i] = std::move(std::vector<float>(n_factors, 0.0));
    vec_w_z[i] = std::move(std::vector<float>(n_factors, 0.0));
  }
  vec_w_mutex = std::move(std::vector<std::shared_mutex>(n_feats));
}

float FM::train(feat_vec &features, int label) {
  remove_out_range(features);
  update_linear_w(features);
  update_bias();
  update_vector_w(features);
  const float logit = compute_fm_logit(features, true);
  const float tmp_grad = utils::sigmoid(logit) - static_cast<float>(label);
  update_linear_nz(features, tmp_grad);
  update_bias_nz(tmp_grad);
  update_vector_nz(features, tmp_grad);
  return logit;
}

float FM::predict(feat_vec &features, bool output_prob) {
  remove_out_range(features);
  const float logit = compute_fm_logit(features, false);
  return output_prob ? utils::sigmoid(logit) : logit;
}

float FM::compute_fm_logit(const feat_vec &features, bool update_model) {
  float result = FtrlModel::compute_linear_logit(features);
  // store sum_vx for later model updating
  if (update_model) {
    for (int f = 0; f < n_factors; f++) {
      sum_vx[f] = 0.0;
      float sum_sqr = 0.0;
      for (const auto &[_, i, x] : features) {
        const float vx = vec_w[i][f] * x;
        sum_vx[f] += vx;
        sum_sqr += vx * vx;
      }
      result += 0.5f * (sum_vx[f] * sum_vx[f] - sum_sqr);
    }
  } else {
    for (int f = 0; f < n_factors; f++) {
      float s_vx = 0.0;
      float sum_sqr = 0.0;
      for (const auto &[_, i, x] : features) {
        const float vx = vec_w[i][f] * x;
        s_vx += vx;
        sum_sqr += vx * vx;
      }
      result += 0.5f * (s_vx * s_vx - sum_sqr);
    }
  }
  return result;
}

void FM::update_vector_w(const feat_vec &features) {
  for (const auto &feat : features) {
    const int i = std::get<1>(feat);
    std::unique_lock lock(vec_w_mutex[i]);
    for (int f = 0; f < n_factors; f++) {
      vec_w[i][f] = maybe_zero_weight(vec_w_n[i][f], vec_w_z[i][f]);
    }
    lock.unlock();
  }
}

void FM::update_vector_nz(const feat_vec &features, float tmp_grad) {
  for (const auto &[_, i, x] : features) {
    std::vector<float> zi(n_factors), ni(n_factors);
    std::shared_lock<std::shared_mutex> read_lock(vec_w_mutex[i]);
    for (int f = 0; f < n_factors; f++) {
      const float vif = vec_w[i][f];
      const float v_nif = vec_w_n[i][f];
      const float v_zif = vec_w_z[i][f];
      const float s_vx = sum_vx[f];
      const float v_gif = tmp_grad * (x * s_vx - vif * x * x);
      const float v_sif = (sqrtf(v_nif + v_gif * v_gif) - sqrtf(v_nif)) / w_alpha;
      zi[f] = v_zif + v_gif - v_sif * vif;
      ni[f] = v_nif + v_gif * v_gif;
    }
    read_lock.unlock();

    std::unique_lock<std::shared_mutex> write_lock(vec_w_mutex[i]);
    vec_w_z[i] = std::move(zi);
    vec_w_n[i] = std::move(ni);
    write_lock.unlock();
  }
}

}  // namespace ftrl
