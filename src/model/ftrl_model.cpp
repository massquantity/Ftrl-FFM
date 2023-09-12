#include "model/ftrl_model.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

#include "utils/utils.h"

namespace ftrl {

FtrlModel::FtrlModel(const config_options &opt)
    : n_feats(opt.n_feats),
      w_alpha(opt.w_alpha),
      w_beta(opt.w_beta),
      w_l1(opt.w_l1),
      w_l2(opt.w_l2),
      bias(0.0),
      bias_n(0.0),
      bias_z(0.0) {
  if (opt.model_type == "LR") {
    model_type = ModelType::LR;
  } else if (opt.model_type == "FM") {
    model_type = ModelType::FM;
  } else if (opt.model_type == "FFM") {
    model_type = ModelType::FFM;
  }
  lin_w = std::move(utils::init_weights(n_feats, opt.init_mean, opt.init_stddev));
  lin_w_n.resize(n_feats);
  std::fill(lin_w_n.begin(), lin_w_n.end(), 0.0);
  lin_w_z.resize(n_feats);
  std::fill(lin_w_z.begin(), lin_w_z.end(), 0.0);
  lin_w_mutex = std::move(std::vector<std::mutex>(n_feats));
}

void FtrlModel::remove_out_range(feat_vec &feats) {
  auto new_end = std::remove_if(feats.begin(), feats.end(), [&](auto &f) {
    const int i = std::get<1>(f);
    return i < 0 || i >= n_feats;
  });
  feats.erase(new_end, feats.end());
}

float FtrlModel::compute_linear_logit(const feat_vec &features) {
  return std::accumulate(features.cbegin(), features.cend(), bias,
                         [&](float acc, const std::tuple<int, int, float> &feat) {
                           const auto &[_, i, x] = feat;
                           return acc + lin_w[i] * x;
                         });
}

void FtrlModel::update_linear_w(const feat_vec &features) {
  for (const auto &feat : features) {
    const int i = std::get<1>(feat);
    std::unique_lock lock(lin_w_mutex[i]);
    lin_w[i] = maybe_zero_weight(lin_w_n[i], lin_w_z[i]);
    lock.unlock();
  }
}

void FtrlModel::update_bias() {
  const std::scoped_lock lock(bias_mutex);
  bias = maybe_zero_weight(bias_n, bias_z);
}

void FtrlModel::update_linear_nz(const feat_vec &features, float tmp_grad) {
  for (const auto &[_, i, x] : features) {
    std::unique_lock lock(lin_w_mutex[i]);
    const float wi = lin_w[i];
    const float ni = lin_w_n[i];
    const float gi = tmp_grad * x;
    const float si = (sqrtf(ni + gi * gi) - sqrtf(ni)) / w_alpha;
    lin_w_z[i] += gi - si * wi;
    lin_w_n[i] += gi * gi;
    lock.unlock();
  }
}

void FtrlModel::update_bias_nz(float tmp_grad) {
  const std::scoped_lock lock(bias_mutex);
  const float gi = tmp_grad;
  const float si = (sqrtf(bias_n + gi * gi) - sqrtf(bias_n)) / w_alpha;
  bias_z += gi - si * bias;
  bias_n += gi * gi;
}

}  // namespace ftrl
