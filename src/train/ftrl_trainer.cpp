#include "train/ftrl_trainer.h"

#include <cmath>
#include <mutex>

namespace ftrl {

void update_linear_weight(std::vector<std::shared_ptr<ftrl_model_unit>> &params, size_t feat_len,
                          float w_alpha, float w_beta, float w_l1, float w_l2) {
  for (int i = 0; i <= feat_len; i++) {
    ftrl_model_unit &mu = *(params[i]);
    std::unique_lock<std::mutex> lck(mu.mtx);
    if (std::fabs(mu.w_zi) <= w_l1) {
      mu.wi = 0.0;
    } else {
      mu.wi = -1 * (mu.w_zi - utils::sgn(mu.w_zi) * w_l1) /
              (w_l2 + (w_beta + std::sqrt(mu.w_ni)) / w_alpha);
    }
    lck.unlock();
  }
}

void update_fm_weight(std::vector<std::shared_ptr<ftrl_model_unit>> &params,
                      const std::vector<std::tuple<int, int, float>> &x, int n_factors,
                      size_t feat_len, float w_alpha, float w_beta, float w_l1, float w_l2) {
  for (int i = 0; i < feat_len; i++) {
    ftrl_model_unit &mu = *(params[i]);
    for (int f = 0; f < n_factors; f++) {
      std::unique_lock<std::mutex> lck(mu.mtx);
      float &vif = mu.vi[f];
      const float &v_nif = mu.v_ni[f];
      const float &v_zif = mu.v_zi[f];
      if (std::fabs(v_zif) <= w_l1) {
        vif = 0.0;
      } else {
        vif = -1 * (v_zif - utils::sgn(v_zif) * w_l1) /
              (w_l2 + (w_beta + std::sqrt(v_nif)) / w_alpha);
      }
      lck.unlock();
    }
  }
}

void update_ffm_weight(std::vector<std::shared_ptr<ftrl_model_unit>> &params,
                       const std::vector<std::tuple<int, int, float>> &x, int n_factors,
                       size_t feat_len, float w_alpha, float w_beta, float w_l1, float w_l2) {
  for (int i = 0; i < feat_len; i++) {
    for (int j = i + 1; j < feat_len; j++) {
      const int fi1 = std::get<0>(x[i]);
      const int fi2 = std::get<0>(x[j]);
      ftrl_model_unit &mu1 = *(params[i]);
      ftrl_model_unit &mu2 = *(params[j]);
      for (int f = 0; f < n_factors; f++) {
        std::unique_lock<std::mutex> lck1(mu1.mtx, std::defer_lock);
        std::unique_lock<std::mutex> lck2(mu2.mtx, std::defer_lock);
        std::lock(lck1, lck2);
        float &vif1 = mu1.vi[fi2 * n_factors + f];
        const float &v_nif1 = mu1.v_ni[fi2 * n_factors + f];
        const float &v_zif1 = mu1.v_zi[fi2 * n_factors + f];
        if (std::fabs(v_zif1) <= w_l1) {
          vif1 = 0.0;
        } else {
          vif1 = -1 * (v_zif1 - utils::sgn(v_zif1) * w_l1) /
                 (w_l2 + (w_beta + std::sqrt(v_nif1)) / w_alpha);
        }

        float &vif2 = mu2.vi[fi1 * n_factors + f];
        const float &v_nif2 = mu2.v_ni[fi1 * n_factors + f];
        const float &v_zif2 = mu2.v_zi[fi1 * n_factors + f];
        if (std::fabs(v_zif2) <= w_l1) {
          vif2 = 0.0;
        } else {
          vif2 = -1 * (v_zif2 - utils::sgn(v_zif2) * w_l1) /
                 (w_l2 + (w_beta + std::sqrt(v_nif2)) / w_alpha);
        }
        lck1.unlock();
        lck2.unlock();
      }
    }
  }
}

void update_linear_nz(std::vector<std::shared_ptr<ftrl_model_unit>> &params,
                      const std::vector<std::tuple<int, int, float>> &x, size_t feat_len,
                      float w_alpha, float mult) {
  for (int i = 0; i <= feat_len; i++) {
    ftrl_model_unit &mu = *(params[i]);
    const float xi = i < feat_len ? std::get<2>(x[i]) : 1.0f;
    std::unique_lock<std::mutex> lck(mu.mtx);
    const float w_gi = mult * xi;
    const float w_si = 1.0f / w_alpha * (sqrtf(mu.w_ni + w_gi * w_gi) - sqrtf(mu.w_ni));
    mu.w_zi += w_gi - w_si * mu.wi;
    mu.w_ni += w_gi * w_gi;
    lck.unlock();
  }
}

void update_fm_nz(std::vector<std::shared_ptr<ftrl_model_unit>> &params,
                  const std::vector<std::tuple<int, int, float>> &x, size_t feat_len, float w_alpha,
                  float mult, int n_factors, std::vector<float> &sum_vx) {
  for (int i = 0; i < feat_len; i++) {
    ftrl_model_unit &mu = *(params[i]);
    const float &xi = std::get<2>(x[i]);
    for (int f = 0; f < n_factors; f++) {
      std::unique_lock<std::mutex> lck(mu.mtx);
      const float &vif = mu.vi[f];
      float &v_nif = mu.v_ni[f];
      float &v_zif = mu.v_zi[f];
      const float &s_vx = sum_vx[f];
      const float v_gif = mult * (xi * s_vx - vif * xi * xi);
      const float v_sif = 1 / w_alpha * (sqrtf(v_nif + v_gif * v_gif) - sqrtf(v_nif));
      v_zif += v_gif - v_sif * vif;
      v_nif += v_gif * v_gif;
      lck.unlock();
    }
  }
}

void update_ffm_nz(std::vector<std::shared_ptr<ftrl_model_unit>> &params,
                   const std::vector<std::tuple<int, int, float>> &x, size_t feat_len,
                   float w_alpha, float mult, int n_factors) {
  for (int i = 0; i < feat_len; i++) {
    for (int j = i + 1; j < feat_len; j++) {
      const int fi1 = std::get<0>(x[i]);
      const int fi2 = std::get<0>(x[j]);
      ftrl_model_unit &mu1 = *(params[i]);
      ftrl_model_unit &mu2 = *(params[j]);
      const float &xi = std::get<2>(x[i]) * std::get<2>(x[j]);
      for (int f = 0; f < n_factors; f++) {
        std::unique_lock<std::mutex> lck1(mu1.mtx, std::defer_lock);
        std::unique_lock<std::mutex> lck2(mu2.mtx, std::defer_lock);
        std::lock(lck1, lck2);
        const float &vif1 = mu1.vi[fi2 * n_factors + f];
        float &v_nif1 = mu1.v_ni[fi2 * n_factors + f];
        float &v_zif1 = mu1.v_zi[fi2 * n_factors + f];
        const float &vif2 = mu2.vi[fi1 * n_factors + f];
        float &v_nif2 = mu2.v_ni[fi1 * n_factors + f];
        float &v_zif2 = mu2.v_zi[fi1 * n_factors + f];

        const float v_gif1 = mult * vif2 * xi;
        const float v_sif1 = 1.0f / w_alpha * (sqrtf(v_nif1 + v_gif1 * v_gif1) - sqrtf(v_nif1));
        v_zif1 += v_gif1 - v_sif1 * vif1;
        v_nif1 += v_gif1 * v_gif1;
        const float v_gif2 = mult * vif1 * xi;
        const float v_sif2 = 1.0f / w_alpha * (sqrtf(v_nif2 + v_gif2 * v_gif1) - sqrtf(v_nif2));
        v_zif2 += v_gif2 - v_sif2 * vif2;
        v_nif2 += v_gif2 * v_gif2;
        lck1.unlock();
        lck2.unlock();
      }
    }
  }
}

}  // namespace ftrl
