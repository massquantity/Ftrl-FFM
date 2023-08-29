#include "model/ffm.h"

#include <algorithm>
#include <numeric>
#include <utility>

#include "utils/utils.h"

namespace ftrl {

FFM::FFM(const config_options &opt)
    : FtrlModel(opt), n_fields(opt.n_fields), n_factors(opt.n_factors) {
  vec_w =
      std::move(utils::init_weights(n_feats, n_fields, n_factors, opt.init_mean, opt.init_stddev));
  vec_w_n.resize(n_feats);
  vec_w_z.resize(n_feats);
  for (int i = 0; i < n_feats; i++) {
    vec_w_n[i] = std::move(std::vector<float>(n_fields * n_factors, 0.0));
    vec_w_z[i] = std::move(std::vector<float>(n_fields * n_factors, 0.0));
  }
  vec_w_mutex = std::move(std::vector<std::shared_mutex>(n_feats));
}

void FFM::remove_out_range(feat_vec &feats) {
  auto new_end = std::remove_if(feats.begin(), feats.end(), [&](auto &f) {
    const auto [field, feat, _] = f;
    return field < 0 || feat < 0 || field >= n_fields || feat >= n_feats;
  });
  feats.erase(new_end, feats.end());
}

float FFM::train(feat_vec &features, int label) {
  remove_out_range(features);
  update_linear_w(features);
  update_bias();
  update_vector_w(features);
  const float logit = compute_ffm_logit(features);
  const float tmp_grad = utils::sigmoid(logit) - static_cast<float>(label);
  update_linear_nz(features, tmp_grad);
  update_bias_nz(tmp_grad);
  update_vector_nz(features, tmp_grad);
  return logit;
}

float FFM::predict(feat_vec &features, bool output_prob) {
  remove_out_range(features);
  const float logit = compute_ffm_logit(features);
  return output_prob ? utils::sigmoid(logit) : logit;
}

float FFM::compute_ffm_logit(const feat_vec &features) {
  float result = FtrlModel::compute_linear_logit(features);
  for (int m = 0; m < features.size(); m++) {
    for (int n = m + 1; n < features.size(); n++) {
      const auto [field1, i, x1] = features[m];
      const auto [field2, j, x2] = features[n];
      const float dot = std::inner_product(vec_w[i].cbegin() + field2 * n_factors,
                                           vec_w[i].cbegin() + field2 * n_factors + n_factors,
                                           vec_w[j].cbegin() + field1 * n_factors, 0.0f);
      result += dot * x1 * x2;
    }
  }
  return result;
}

void FFM::update_vector_w(const feat_vec &features) {
  for (int m = 0; m < features.size(); m++) {
    for (int n = m + 1; n < features.size(); n++) {
      const auto [field1, i, _x1] = features[m];
      const auto [field2, j, _x2] = features[n];
      {
        const std::scoped_lock lock(vec_w_mutex[i], vec_w_mutex[j]);
        for (int f = 0; f < n_factors; f++) {
          const int f1 = field2 * n_factors + f;
          vec_w[i][f1] = maybe_zero_weight(vec_w_n[i][f1], vec_w_z[i][f1]);
          const int f2 = field1 * n_factors + f;
          vec_w[j][f2] = maybe_zero_weight(vec_w_n[j][f2], vec_w_z[j][f2]);
        }
      }
    }
  }
}

void FFM::update_vector_nz(const feat_vec &features, float tmp_grad) {
  for (int m = 0; m < features.size(); m++) {
    for (int n = m + 1; n < features.size(); n++) {
      const auto [field1, i, x1] = features[m];
      const auto [field2, j, x2] = features[n];
      const float x = x1 * x2;
      std::vector<float> zi1(n_factors), ni1(n_factors), ni2(n_factors), zi2(n_factors);
      {
        // https://stackoverflow.com/questions/54641216/can-scoped-lock-lock-a-shared-mutex-in-read-mode
        std::shared_lock<std::shared_mutex> read_lock1(vec_w_mutex[i], std::defer_lock);
        std::shared_lock<std::shared_mutex> read_lock2(vec_w_mutex[j], std::defer_lock);
        const std::scoped_lock read_lock(read_lock1, read_lock2);
        for (int f = 0; f < n_factors; f++) {
          const int f1 = field2 * n_factors + f;
          const float vif1 = vec_w[i][f1];
          const float v_nif1 = vec_w_n[i][f1];
          const float v_zif1 = vec_w_z[i][f1];
          const int f2 = field1 * n_factors + f;
          const float vif2 = vec_w[j][f2];
          const float v_nif2 = vec_w_n[j][f2];
          const float v_zif2 = vec_w_z[j][f2];

          const float v_gif1 = tmp_grad * vif2 * x;
          const float v_sif1 = (sqrtf(v_nif1 + v_gif1 * v_gif1) - sqrtf(v_nif1)) / w_alpha;
          zi1[f] = v_zif1 + v_gif1 - v_sif1 * vif1;
          ni1[f] = v_nif1 + v_gif1 * v_gif1;

          const float v_gif2 = tmp_grad * vif1 * x;
          const float v_sif2 = (sqrtf(v_nif2 + v_gif2 * v_gif1) - sqrtf(v_nif2)) / w_alpha;
          zi2[f] = v_zif2 + v_gif2 - v_sif2 * vif2;
          ni2[f] = v_nif2 + v_gif2 * v_gif2;
        }
      }

      {
        const std::scoped_lock write_lock(vec_w_mutex[i], vec_w_mutex[j]);
        // std::unique_lock<std::shared_mutex> write_lock1(mu1->mtx, std::defer_lock);
        // std::unique_lock<std::shared_mutex> write_lock2(mu2->mtx, std::defer_lock);
        // std::lock(write_lock1, write_lock2);
        std::move(zi1.cbegin(), zi1.cend(), vec_w_z[i].begin() + field2 * n_factors);
        std::move(ni1.cbegin(), ni1.cend(), vec_w_n[i].begin() + field2 * n_factors);
        std::move(zi2.cbegin(), zi2.cend(), vec_w_z[j].begin() + field1 * n_factors);
        std::move(ni2.cbegin(), ni2.cend(), vec_w_n[j].begin() + field1 * n_factors);
      }
    }
  }
}

}  // namespace ftrl
