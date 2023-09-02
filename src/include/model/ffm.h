#ifndef FTRL_FFM_FFM_H
#define FTRL_FFM_FFM_H

#include <shared_mutex>

#include "ftrl_model.h"

namespace ftrl {

class FFM : public FtrlModel {
 public:
  explicit FFM(const config_options &opt);
  float train(feat_vec &features, int label) override;
  float predict(feat_vec &features, bool output_prob) override;
  float compute_ffm_logit(const feat_vec &features);
  void update_vector_w(const feat_vec &features);
  void update_vector_nz(const feat_vec &features, float tmp_grad);
  void remove_out_range(feat_vec &feats) override;

  std::vector<std::vector<float>> vec_w;

 private:
  int n_fields;
  int n_factors;
  std::vector<std::vector<float>> vec_w_n;
  std::vector<std::vector<float>> vec_w_z;
  std::vector<std::shared_mutex> vec_w_mutex;
};

}  // namespace ftrl

#endif  // FTRL_FFM_FFM_H
