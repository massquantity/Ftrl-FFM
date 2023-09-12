#ifndef FTRL_FFM_FM_H
#define FTRL_FFM_FM_H

#include <shared_mutex>
#include <vector>

#include "model/ftrl_model.h"

namespace ftrl {

class FM : public FtrlModel {
 public:
  explicit FM(const config_options &opt);
  float train(feat_vec &features, int label) override;
  float predict(feat_vec &features, bool output_prob) override;
  float compute_fm_logit(const feat_vec &features, bool update_model);
  void update_vector_w(const feat_vec &features);
  void update_vector_nz(const feat_vec &features, float tmp_grad);

  std::vector<std::vector<float>> vec_w;

 private:
  int n_factors;
  std::vector<float> sum_vx;
  std::vector<std::vector<float>> vec_w_n;
  std::vector<std::vector<float>> vec_w_z;
  std::vector<std::shared_mutex> vec_w_mutex;
};

}  // namespace ftrl

#endif  // FTRL_FFM_FM_H
