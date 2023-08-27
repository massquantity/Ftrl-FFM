#ifndef FTRL_FFM_EVALUATE_H
#define FTRL_FFM_EVALUATE_H

#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "concurrent/pc_task.h"
#include "data/parser.h"
#include "model/ftrl_model.h"
#include "utils/cmd_option.h"
#include "utils/types.h"

namespace ftrl {

class Evaluator : public PcTask {
 public:
  explicit Evaluator(const config_options &opt);
  void load_trained_model(std::shared_ptr<FtrlModel> &train_model);
  double get_loss();
  ~Evaluator() override;

 private:
  void run_task(std::vector<std::string> &data_buffer, int t) override;

  std::shared_ptr<FtrlModel> eval_model;
  std::shared_ptr<Parser> parser;
  std::unique_ptr<double[]> losses;
  std::unique_ptr<uint64[]> nums;
  int n_threads;
};

}  // namespace ftrl

#endif  // FTRL_FFM_EVALUATE_H
