#ifndef FTRL_FFM_LR_H
#define FTRL_FFM_LR_H

#include <string_view>

#include "model/ftrl_model.h"

namespace ftrl {

class LR : public FtrlModel {
 public:
  explicit LR(const config_options &opt);
  float train(feat_vec &features, int label) override;
  float predict(feat_vec &features, bool output_prob) override;
  void save_compressed_model(std::string_view file_name, int compress_level);
  void load_compressed_model(std::string_view file_name);
};

}  // namespace ftrl

#endif  // FTRL_FFM_LR_H
