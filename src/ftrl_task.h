#ifndef FTRL_FFM_FTRL_TASK_H
#define FTRL_FFM_FTRL_TASK_H

#include <chrono>
#include "model/ftrl_offline.h"
#include "reader/reader.h"

using namespace std::chrono;
using timer = std::chrono::steady_clock;

namespace ftrl {

class FtrlTask {
public:
  explicit FtrlTask(const trainer_option &_opt): opt(_opt) { }
  void init();
  double train();
  double evaluate();

private:
  std::shared_ptr<FtrlOffline> model;
  std::shared_ptr<Reader> trainData;
  std::shared_ptr<Reader> evalData;
  trainer_option opt;
};

void FtrlTask::init() {
  model = std::make_shared<FtrlOffline>(opt);
  trainData = std::make_shared<Reader>();
  trainData->loadFromFile(opt.train_path, opt.thread_num);
  if (!opt.eval_path.empty()) {
    evalData = std::make_shared<Reader>();
    evalData->loadFromFile(opt.eval_path, opt.thread_num);
  }
}

double FtrlTask::train() {
  for (int i = 1; i <= opt.epoch; i++) {
    auto start0 = timer::now();
    double train_loss = model->oneEpochBatch(trainData->data, true);
    auto end0 = timer::now();
    double dur0 = double(duration_cast<nanoseconds>(end0 - start0).count())
        * nanoseconds::period::num / nanoseconds::period::den;
    printf("epoch %d train time: %.4lfs, train loss: %.4lf\n", i, dur0, train_loss);
    if (evalData != nullptr) {
      auto start1 = timer::now();
      double eval_loss = model->oneEpochBatch(trainData->data, false);
      auto end1 = timer::now();
      // auto dur1 = std::chrono::duration<double>(end1 - start1).count();
      double dur1 = double(duration_cast<nanoseconds>(end1 - start1).count())
          * nanoseconds::period::num / nanoseconds::period::den;
      printf("epoch %d eval time: %.4lfs, eval loss: %.4lf\n", i, dur1, eval_loss);
    }
  }
}

}

#endif //FTRL_FFM_FTRL_TASK_H