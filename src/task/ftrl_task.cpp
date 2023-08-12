#include <chrono>

#include "task/ftrl_task.h"

using namespace std::chrono;
using timer = std::chrono::steady_clock;

namespace ftrl {

void FtrlTask::init() {
  if (opt.online) {
    online_model = std::make_shared<FtrlOnline>(opt);
    if (!opt.eval_path.empty()) {
      eval_online_model = std::make_shared<Evaluator>(opt);
    }
  } else {
    offline_model = std::make_shared<FtrlOffline>(opt);
    train_data = std::make_shared<Reader>(opt.file_type);
    train_data->load_from_file(opt.train_path, opt.thread_num);
    if (!opt.eval_path.empty()) {
      eval_data = std::make_shared<Reader>(opt.file_type);
      eval_data->load_from_file(opt.eval_path, opt.thread_num);
    }
  }
}

void FtrlTask::train_online() {
  if (!opt.cmd) {
    online_model->open_file(opt.train_path);
    if (!opt.eval_path.empty() && eval_online_model != nullptr) {
      eval_online_model->open_file(opt.eval_path);
    }
    for (int i = 1; i <= opt.epoch; i++) {
      auto train_start = timer::now();
      online_model->run();
      online_model->rewind_file();
      const double train_loss = online_model->get_loss();
      const double train_time = utils::compute_time(train_start);
      printf("epoch %d train time: %.4lfs, train loss: %.4lf\n", i, train_time, train_loss);

      if (!opt.eval_path.empty() && eval_online_model != nullptr) {
        auto eval_start = timer::now();
        eval_online_model->load_trained_model(online_model->model_ptr);
        eval_online_model->run();
        eval_online_model->rewind_file();
        const double eval_loss = eval_online_model->get_loss();
        const double eval_time = utils::compute_time(eval_start);
        printf("epoch %d eval time: %.4lfs, eval loss: %.4lf\n", i, eval_time, eval_loss);
      }
    }
  } else {
    // todo: online learning
  }
}

void FtrlTask::eval_online() {
  eval_online_model = std::make_shared<Evaluator>(opt);
  eval_online_model->open_file(opt.eval_path);
  auto start = timer::now();
  eval_online_model->load_trained_model(online_model->model_ptr);
  eval_online_model->run();
  const double eval_loss = eval_online_model->get_loss();
  auto eval_time = utils::compute_time(start);
  printf("eval time: %.4lfs, eval loss: %.4lf\n", eval_time, eval_loss);
}

void FtrlTask::train_offline() {
  for (int i = 1; i <= opt.epoch; i++) {
    auto train_start = timer::now();
    const double train_loss = offline_model->one_epoch_pool(train_data->data, true);
    const double train_time = utils::compute_time(train_start);
    printf("epoch %d train time: %.4lfs, train loss: %.4lf\n", i, train_time, train_loss);

    if (eval_data != nullptr) {
      auto eval_start = timer::now();
      const double eval_loss = offline_model->one_epoch_pool(eval_data->data, false);
      const double eval_time = utils::compute_time(eval_start);
      printf("epoch %d eval time: %.4lfs, eval loss: %.4lf\n", i, eval_time, eval_loss);
    }
  }
}

void FtrlTask::eval_offline() {
  auto eval_start = timer::now();
  const double eval_loss = offline_model->one_epoch_batch(eval_data->data, false);
  const double eval_time = utils::compute_time(eval_start);
  printf("eval time: %.4lfs, eval loss: %.4lf\n", eval_time, eval_loss);
}

}
