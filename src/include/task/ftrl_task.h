#ifndef FTRL_FFM_FTRL_TASK_H
#define FTRL_FFM_FTRL_TASK_H

#include <memory>
#include <utility>

#include "data/reader.h"
#include "eval/evaluate.h"
#include "model/ftrl_offline.h"
#include "model/ftrl_online.h"

namespace ftrl {

class FtrlTask {
 public:
  explicit FtrlTask(config_options _opt) : opt(std::move(_opt)) {}
  void init();

  void train() { opt.online ? train_online() : train_offline(); }
  [[maybe_unused]] void evaluate() { opt.online ? eval_online() : eval_offline(); }

 private:
  std::shared_ptr<FtrlOnline> online_model;
  std::shared_ptr<Evaluator> eval_online_model;
  std::shared_ptr<FtrlOffline> offline_model;
  std::shared_ptr<Reader> train_data;
  std::shared_ptr<Reader> eval_data;
  config_options opt;

  void train_online();
  void eval_online();
  void train_offline();
  void eval_offline();
};

}  // namespace ftrl

#endif  // FTRL_FFM_FTRL_TASK_H