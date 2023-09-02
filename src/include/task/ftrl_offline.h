#ifndef FTRL_FFM_FTRL_OFFLINE_H
#define FTRL_FFM_FTRL_OFFLINE_H

#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

#include "concurrent/thread_pool.h"
#include "data/reader.h"
#include "data/sample.h"
#include "model/ftrl_model.h"
#include "utils/cmd_option.h"

namespace ftrl {

class FtrlOffline {
 public:
  explicit FtrlOffline(const config_options &opt);
  void train();
  void evaluate(int epoch = 0);
  double one_epoch(std::vector<Sample> &samples, bool train, bool use_pool);
  bool has_zero_weights();

  std::unique_ptr<FtrlModel> model_ptr;

 private:
  int n_epochs;
  int n_threads;
  std::unique_ptr<Reader> train_data_loader;
  std::unique_ptr<Reader> eval_data_loader;
  std::unique_ptr<ThreadPool> thread_pool;
};

}  // namespace ftrl

#endif  // FTRL_FFM_FTRL_OFFLINE_H
