#ifndef FTRL_FFM_EVALUATE_H
#define FTRL_FFM_EVALUATE_H

#include <fstream>
#include <memory>
#include <mutex>
#include <vector>

#include "data/parser.h"
#include "model/ftrl_model.h"
#include "threading/pc_task.h"
#include "utils/cmd_option.h"
#include "utils/types.h"

namespace ftrl {

class Evaluator : public PcTask {
 public:
  explicit Evaluator(const config_options &opt);
  void load_trained_model(std::shared_ptr<FtrlModel> &train_model);
  double predict(const feat_vec &feats);
  double get_loss();
  ~Evaluator() override;

 private:
  std::shared_ptr<FtrlModel> eval_model;
  int n_threads;
  std::shared_ptr<Parser> parser;
  void run_task(std::vector<std::string> &data_buffer, int t) override;
  std::unique_ptr<double[]> losses;
  std::unique_ptr<uint64[]> nums;
  // std::mutex eval_mtx;
};

}  // namespace ftrl

#endif  // FTRL_FFM_EVALUATE_H