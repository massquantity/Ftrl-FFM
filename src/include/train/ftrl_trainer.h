#ifndef FTRL_FFM_FTRL_TRAINER_H
#define FTRL_FFM_FTRL_TRAINER_H

#include <memory>
#include <tuple>
#include <vector>

#include "model/ftrl_unit.h"

namespace ftrl {

void update_linear_weight(std::vector<std::shared_ptr<ftrl_model_unit>>& params, size_t feat_len,
                          float w_alpha, float w_beta, float w_l1, float w_l2);

void update_fm_weight(std::vector<std::shared_ptr<ftrl_model_unit>>& params,
                      const std::vector<std::tuple<int, int, float>>& x, int n_factors,
                      size_t feat_len, float w_alpha, float w_beta, float w_l1, float w_l2);

void update_ffm_weight(std::vector<std::shared_ptr<ftrl_model_unit>>& params,
                       const std::vector<std::tuple<int, int, float>>& x, int n_factors,
                       size_t feat_len, float w_alpha, float w_beta, float w_l1, float w_l2);

void update_linear_nz(std::vector<std::shared_ptr<ftrl_model_unit>>& params,
                      const std::vector<std::tuple<int, int, float>>& x, size_t feat_len,
                      float w_alpha, float mult);

void update_fm_nz(std::vector<std::shared_ptr<ftrl_model_unit>>& params,
                  const std::vector<std::tuple<int, int, float>>& x, size_t feat_len, float w_alpha,
                  float mult, int n_factors, std::vector<float>& sum_vx);

void update_ffm_nz(std::vector<std::shared_ptr<ftrl_model_unit>>& params,
                   const std::vector<std::tuple<int, int, float>>& x, size_t feat_len,
                   float w_alpha, float mult, int n_factors);

}  // namespace ftrl

#endif  // FTRL_FFM_FTRL_TRAINER_H
