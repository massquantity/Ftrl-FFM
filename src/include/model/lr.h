#ifndef FTRL_FFM_LR_H
#define FTRL_FFM_LR_H

#include "ftrl_model.h"

namespace ftrl {

class LR : public FtrlModel {
 public:
  explicit LR(const config_options &opt);
  float train(feat_vec &features, int label) override;
  float predict(feat_vec &features, bool output_prob) override;
};

}  // namespace ftrl

#endif  // FTRL_FFM_LR_H
