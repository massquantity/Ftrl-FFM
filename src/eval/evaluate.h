#ifndef FTRL_FFM_EVALUATE_H
#define FTRL_FFM_EVALUATE_H

#include <fstream>
#include <mutex>
#include <memory>
#include "../model/ftrl_model.h"
#include "../threading/pc_task.h"
#include "../reader/parser.h"
#include "loss.h"

namespace ftrl {

class Evaluator : public PcTask {
public:
  explicit Evaluator(const trainer_option &opt);
  void loadTrainedModel(std::shared_ptr<FtrlModel> &train_model);
  double predict(const std::vector<std::tuple<int, int, float>> &x);
  double get_loss();
  ~Evaluator();

private:
  std::shared_ptr<FtrlModel> eval_model;
  int n_threads;
  std::shared_ptr<Parser> parser;
  void run_task(std::vector<std::string> &data_buffer, int t) override;
  std::unique_ptr<double[]> losses;
  std::unique_ptr<long[]> nums;
  // std::mutex eval_mtx;
};

Evaluator::Evaluator(const trainer_option &opt)
    : PcTask(opt.thread_num, opt.cmd), n_threads(opt.thread_num) {
  int n = opt.thread_num;
  losses = std::unique_ptr<double[]>(new double[n]);
  nums = std::unique_ptr<long[]>(new long[n]);
  for (int i = 0; i < n; i++) {
    losses[i] = 0.0;
    nums[i] = 0;
  }
  if (opt.file_type == "libsvm") {
    parser = std::make_shared<LibsvmParser>();
  } else if (opt.file_type == "libffm") {
    parser = std::make_shared<FFMParser>();
  }
};

void Evaluator::run_task(std::vector<std::string> &data_buffer, int t) {
  double tmp_loss = 0.0;
  for (const auto &rawData : data_buffer) {
    Sample sample;
    parser->parse(rawData, sample);
    double pred = predict(sample.x);
    tmp_loss += loss(sample.y, pred);
  }
  losses[t] += tmp_loss;
  nums[t] += data_buffer.size();
}

void Evaluator::loadTrainedModel(std::shared_ptr<FtrlModel> &train_model) {
  eval_model = train_model;
}

double Evaluator::predict(const std::vector<std::tuple<int, int, float>> &x) {
  return eval_model->predict(x, false);
}

double Evaluator::get_loss() {
  double total_loss = 0.0;
  long total_num = 0;
  for (int i = 0; i < n_threads; i++) {
    total_loss += losses[i];
    total_num += nums[i];
  }
  for (int i = 0; i < n_threads; i++) {
    losses[i] = 0.0;
    nums[i] = 0;
  }
  return total_loss / total_num;
}

Evaluator::~Evaluator() {
  losses.release();
  nums.release();
}

}

#endif //FTRL_FFM_EVALUATE_H