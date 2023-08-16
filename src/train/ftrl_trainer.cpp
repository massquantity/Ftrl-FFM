#include "train/ftrl_trainer.h"

#include <cmath>
#include <mutex>
#include <shared_mutex>

namespace ftrl {

void update_linear_weight(std::vector<std::shared_ptr<ftrl_model_unit>> &params, size_t feat_len,
                          float w_alpha, float w_beta, float w_l1, float w_l2) {
  for (int i = 0; i <= feat_len; i++) {
    auto mu = params[i];
    std::unique_lock lock(mu->mtx);
    if (std::fabs(mu->w_zi) <= w_l1) {
      mu->wi = 0.0;
    } else {
      mu->wi = -1 * (mu->w_zi - utils::sgn(mu->w_zi) * w_l1) /
               (w_l2 + (w_beta + std::sqrt(mu->w_ni)) / w_alpha);
    }
    lock.unlock();
  }
}

void update_fm_weight(std::vector<std::shared_ptr<ftrl_model_unit>> &params,
                      const std::vector<std::tuple<int, int, float>> &x, int n_factors,
                      size_t feat_len, float w_alpha, float w_beta, float w_l1, float w_l2) {
  for (int i = 0; i < feat_len; i++) {
    auto mu = params[i];
    std::unique_lock lock(mu->mtx);
    for (int f = 0; f < n_factors; f++) {
      const float v_nif = mu->v_ni[f];
      const float v_zif = mu->v_zi[f];
      if (std::fabs(v_zif) <= w_l1) {
        mu->vi[f] = 0.0;
      } else {
        mu->vi[f] = -1 * (v_zif - utils::sgn(v_zif) * w_l1) /
                    (w_l2 + (w_beta + std::sqrt(v_nif)) / w_alpha);
      }
    }
    lock.unlock();
  }
}

void update_ffm_weight(std::vector<std::shared_ptr<ftrl_model_unit>> &params,
                       const std::vector<std::tuple<int, int, float>> &x, int n_factors,
                       size_t feat_len, float w_alpha, float w_beta, float w_l1, float w_l2) {
  for (int i = 0; i < feat_len; i++) {
    for (int j = i + 1; j < feat_len; j++) {
      const int fi1 = std::get<0>(x[i]);
      const int fi2 = std::get<0>(x[j]);
      auto mu1 = params[i];
      auto mu2 = params[j];
      {
        const std::scoped_lock lock(mu1->mtx, mu2->mtx);
        for (int f = 0; f < n_factors; f++) {
          const int index1 = fi2 * n_factors + f;
          const float v_nif1 = mu1->v_ni[index1];
          const float v_zif1 = mu1->v_zi[index1];
          if (std::fabs(v_zif1) <= w_l1) {
            mu1->vi[index1] = 0.0;
          } else {
            mu1->vi[index1] = -1 * (v_zif1 - utils::sgn(v_zif1) * w_l1) /
                              (w_l2 + (w_beta + std::sqrt(v_nif1)) / w_alpha);
          }

          const int index2 = fi1 * n_factors + f;
          const float v_nif2 = mu2->v_ni[index2];
          const float v_zif2 = mu2->v_zi[index2];
          if (std::fabs(v_zif2) <= w_l1) {
            mu2->vi[index2] = 0.0;
          } else {
            mu2->vi[index2] = -1 * (v_zif2 - utils::sgn(v_zif2) * w_l1) /
                              (w_l2 + (w_beta + std::sqrt(v_nif2)) / w_alpha);
          }
        }
      }
    }
  }
}

void update_linear_nz(std::vector<std::shared_ptr<ftrl_model_unit>> &params,
                      const std::vector<std::tuple<int, int, float>> &x, size_t feat_len,
                      float w_alpha, float tmp_grad) {
  for (int i = 0; i <= feat_len; i++) {
    auto mu = params[i];
    const float xi = i < feat_len ? std::get<2>(x[i]) : 1.0f;
    float zi, ni;  // NOLINT
    std::shared_lock<std::shared_mutex> read_lock(mu->mtx);
    const float w_gi = tmp_grad * xi;
    const float w_si = (sqrtf(mu->w_ni + w_gi * w_gi) - sqrtf(mu->w_ni)) / w_alpha;
    zi = mu->w_zi + w_gi - w_si * mu->wi;
    ni = mu->w_ni + w_gi * w_gi;
    read_lock.unlock();

    std::unique_lock<std::shared_mutex> write_lock(mu->mtx);
    mu->w_zi = zi;
    mu->w_ni = ni;
    write_lock.unlock();
  }
}

void update_fm_nz(std::vector<std::shared_ptr<ftrl_model_unit>> &params,
                  const std::vector<std::tuple<int, int, float>> &x, size_t feat_len, float w_alpha,
                  float tmp_grad, int n_factors, std::vector<float> &sum_vx) {
  for (int i = 0; i < feat_len; i++) {
    auto mu = params[i];
    const float xi = std::get<2>(x[i]);
    std::vector<float> zi(n_factors), ni(n_factors);
    std::shared_lock<std::shared_mutex> read_lock(mu->mtx);
    for (int f = 0; f < n_factors; f++) {
      const float vif = mu->vi[f];
      const float v_nif = mu->v_ni[f];
      const float v_zif = mu->v_zi[f];
      const float s_vx = sum_vx[f];
      const float v_gif = tmp_grad * (xi * s_vx - vif * xi * xi);
      const float v_sif = (sqrtf(v_nif + v_gif * v_gif) - sqrtf(v_nif)) / w_alpha;
      zi[f] = v_zif + v_gif - v_sif * vif;
      ni[f] = v_nif + v_gif * v_gif;
    }
    read_lock.unlock();

    std::unique_lock<std::shared_mutex> write_lock(mu->mtx);
    for (int f = 0; f < n_factors; f++) {
      mu->v_zi[f] = zi[f];
      mu->v_ni[f] = ni[f];
    }
    write_lock.unlock();
  }
}

void update_ffm_nz(std::vector<std::shared_ptr<ftrl_model_unit>> &params,
                   const std::vector<std::tuple<int, int, float>> &x, size_t feat_len,
                   float w_alpha, float tmp_grad, int n_factors) {
  for (int i = 0; i < feat_len; i++) {
    for (int j = i + 1; j < feat_len; j++) {
      auto [field1, _feat1, x1] = x[i];
      auto [field2, _feat2, x2] = x[j];
      auto mu1 = params[i];
      auto mu2 = params[j];
      const float xi = x1 * x2;
      std::vector<float> zi1(n_factors), ni1(n_factors), ni2(n_factors), zi2(n_factors);
      {
        // https://stackoverflow.com/questions/54641216/can-scoped-lock-lock-a-shared-mutex-in-read-mode
        std::shared_lock<std::shared_mutex> read_lock1(mu1->mtx, std::defer_lock);
        std::shared_lock<std::shared_mutex> read_lock2(mu2->mtx, std::defer_lock);
        const std::scoped_lock read_lock(read_lock1, read_lock2);
        for (int f = 0; f < n_factors; f++) {
          const int index1 = field2 * n_factors + f;
          const float vif1 = mu1->vi[index1];
          const float v_nif1 = mu1->v_ni[index1];
          const float v_zif1 = mu1->v_zi[index1];
          const int index2 = field1 * n_factors + f;
          const float vif2 = mu2->vi[index2];
          const float v_nif2 = mu2->v_ni[index2];
          const float v_zif2 = mu2->v_zi[index2];

          const float v_gif1 = tmp_grad * vif2 * xi;
          const float v_sif1 = (sqrtf(v_nif1 + v_gif1 * v_gif1) - sqrtf(v_nif1)) / w_alpha;
          zi1[f] = v_zif1 + v_gif1 - v_sif1 * vif1;
          ni1[f] = v_nif1 + v_gif1 * v_gif1;

          const float v_gif2 = tmp_grad * vif1 * xi;
          const float v_sif2 = (sqrtf(v_nif2 + v_gif2 * v_gif1) - sqrtf(v_nif2)) / w_alpha;
          zi2[f] = v_zif2 + v_gif2 - v_sif2 * vif2;
          ni2[f] = v_nif2 + v_gif2 * v_gif2;
        }
      }

      {
        const std::scoped_lock write_lock(mu1->mtx, mu2->mtx);
        // std::unique_lock<std::shared_mutex> write_lock1(mu1->mtx, std::defer_lock);
        // std::unique_lock<std::shared_mutex> write_lock2(mu2->mtx, std::defer_lock);
        // std::lock(write_lock1, write_lock2);
        for (int f = 0; f < n_factors; f++) {
          const int index1 = field2 * n_factors + f;
          mu1->v_zi[index1] = zi1[f];
          mu1->v_ni[index1] = ni1[f];
          const int index2 = field1 * n_factors + f;
          mu2->v_zi[index2] = zi2[f];
          mu2->v_ni[index2] = ni2[f];
        }
      }
    }
  }
}

}  // namespace ftrl
