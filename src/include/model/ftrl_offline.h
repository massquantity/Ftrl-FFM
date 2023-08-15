#ifndef FTRL_FFM_FTRL_OFFLINE_H
#define FTRL_FFM_FTRL_OFFLINE_H

#include <algorithm>
#include <cassert>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "data/parser.h"
#include "data/sample.h"
#include "model/ftrl_model.h"
#include "threading/thread_pool.h"
#include "utils/cmd_option.h"

namespace ftrl {

class FtrlOffline {
 public:
  explicit FtrlOffline(const config_options &opt);
  double one_epoch_batch(std::vector<Sample> &samples, bool train);
  double one_epoch_pool(std::vector<Sample> &samples, bool train);
  std::shared_ptr<FtrlModel> pModel;

 private:
  float w_alpha, w_beta, w_l1, w_l2;
  int n_threads;
  std::shared_ptr<ThreadPool> thread_pool;
};

}  // namespace ftrl

#endif  // FTRL_FFM_FTRL_OFFLINE_H
