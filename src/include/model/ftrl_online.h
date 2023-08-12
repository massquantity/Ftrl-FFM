#ifndef FTRL_FFM_FTRL_ONLINE_H
#define FTRL_FFM_FTRL_ONLINE_H

#include <cmath>
#include <memory>
#include <vector>

#include "data/parser.h"
#include "model/ftrl_model.h"
#include "threading/pc_task.h"
#include "utils/cmd_option.h"
#include "utils/types.h"

namespace ftrl {

class FtrlOnline : public PcTask {
public:
  explicit FtrlOnline(const config_options &opt);
  void run_task(std::vector<std::string> &data_buffer, int t) override;
  double get_loss();
  [[maybe_unused]] bool load_model(std::ifstream &ifs);
  [[maybe_unused]] void output_model(std::ofstream &ofs);
  std::shared_ptr<FtrlModel> model_ptr;

private:
  float w_alpha, w_beta, w_l1, w_l2;
  int n_threads;
  std::vector<double> losses;
  std::vector<uint64> nums;
  std::shared_ptr<Parser> parser;
};

}

#endif //FTRL_FFM_FTRL_ONLINE_H