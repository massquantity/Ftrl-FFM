#ifndef FTRL_FFM_FTRL_TASK_H
#define FTRL_FFM_FTRL_TASK_H

#include <chrono>
#include <memory>
#include "reader/reader.h"
#include "model/ftrl_online.h"
#include "model/ftrl_offline.h"
#include "eval/evaluate.h"

using namespace std::chrono;
using timer = std::chrono::steady_clock;

namespace ftrl {

class FtrlTask {
public:
  explicit FtrlTask(const trainer_option &_opt): opt(_opt) { } // NOLINT(modernize-pass-by-value)
  void init();
  void train();
  void evaluate();
  void trainOnline();
  void evalOnline();
  void trainOffline();
  void evalOffline();

private:
  std::shared_ptr<FtrlOnline> online_model;
  std::shared_ptr<Evaluator> eval_model;
  std::shared_ptr<FtrlOffline> offline_model;
  std::shared_ptr<Reader> train_data;
  std::shared_ptr<Reader> eval_data;
  trainer_option opt;
};

void FtrlTask::init() {
  if (opt.online) {
    online_model = std::make_shared<FtrlOnline>(opt);
  } else {
    offline_model = std::make_shared<FtrlOffline>(opt);
    train_data = std::make_shared<Reader>(opt.file_type);
    train_data->loadFromFile(opt.train_path, opt.thread_num);
    if (!opt.eval_path.empty()) {
      eval_data = std::make_shared<Reader>(opt.file_type);
      eval_data->loadFromFile(opt.eval_path, opt.thread_num);
    }
  }
}

void FtrlTask::train() {
  if (opt.online) {
    if (!opt.eval_path.empty()) {
      eval_model = std::make_shared<Evaluator>(opt);
    }
    trainOnline();
  } else {
    trainOffline();
  }
}

void FtrlTask::evaluate() {
  if (opt.online) {
    evalOnline();
  } else {
    evalOffline();
  }
}

void FtrlTask::trainOnline() {
  if (!opt.cmd) {
    online_model->openFile(opt.train_path);
    if (!opt.eval_path.empty()) {
      eval_model->openFile(opt.eval_path);
    }
    for (int i = 1; i <= opt.epoch; i++) {
      auto start0 = timer::now();
      online_model->run();
      online_model->rewindFile();
      double train_loss = online_model->get_loss();
      auto end0 = timer::now();
      auto dur0 = double(duration_cast<nanoseconds>(end0 - start0).count())
                  * nanoseconds::period::num / nanoseconds::period::den;
      printf("epoch %d train time: %.4lfs, train loss: %.4lf\n", i, dur0, train_loss);

      if (!opt.eval_path.empty()) {
        auto start1 = std::chrono::steady_clock::now();
        eval_model->loadTrainedModel(online_model->pModel);
        eval_model->run();
        eval_model->rewindFile();
        double eval_loss = eval_model->get_loss();
        auto end1 = std::chrono::steady_clock::now();
        auto dur1 = double(duration_cast<std::chrono::nanoseconds>(end1 - start1).count())
                    * nanoseconds::period::num / nanoseconds::period::den;
        printf("epoch %d eval time: %.4lfs, eval loss: %.4lf\n", i, dur1, eval_loss);
      }
    }
  } else {
    // todo
  }
}

void FtrlTask::evalOnline() {
  eval_model = std::make_shared<Evaluator>(opt);
  eval_model->openFile(opt.eval_path);
  auto start1 = std::chrono::steady_clock::now();
  eval_model->loadTrainedModel(online_model->pModel);
  eval_model->run();
  double eval_loss = eval_model->get_loss();
  auto end1 = std::chrono::steady_clock::now();
  auto dur1 = double(duration_cast<std::chrono::nanoseconds>(end1 - start1).count())
              * nanoseconds::period::num / nanoseconds::period::den;
  printf("eval time: %.4lfs, eval loss: %.4lf\n", dur1, eval_loss);
}

void FtrlTask::trainOffline() {
  for (int i = 1; i <= opt.epoch; i++) {
    auto start0 = timer::now();
    double train_loss = offline_model->oneEpochPool(train_data->data, true);
    auto end0 = timer::now();
    double dur0 = double(duration_cast<nanoseconds>(end0 - start0).count())
        * nanoseconds::period::num / nanoseconds::period::den;
    printf("epoch %d train time: %.4lfs, train loss: %.4lf\n", i, dur0, train_loss);
    if (eval_data != nullptr) {
      auto start1 = timer::now();
      double eval_loss = offline_model->oneEpochPool(train_data->data, false);
      auto end1 = timer::now();
      double dur1 = double(duration_cast<nanoseconds>(end1 - start1).count())
          * nanoseconds::period::num / nanoseconds::period::den;
      printf("epoch %d eval time: %.4lfs, eval loss: %.4lf\n", i, dur1, eval_loss);
    }
  }
}

void FtrlTask::evalOffline() {
  auto start1 = timer::now();
  double eval_loss = offline_model->oneEpochBatch(train_data->data, false);
  auto end1 = timer::now();
  double dur1 = double(duration_cast<nanoseconds>(end1 - start1).count())
                * nanoseconds::period::num / nanoseconds::period::den;
  printf("eval time: %.4lfs, eval loss: %.4lf\n", dur1, eval_loss);
}

}

#endif //FTRL_FFM_FTRL_TASK_H