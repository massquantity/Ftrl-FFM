#ifndef FTRL_FFM_EVALUATE_H
#define FTRL_FFM_EVALUATE_H

#include <fstream>
#include <mutex>
#include <memory>
#include "../model/ftrl_model.h"
#include "../threading/pc_task.h"
#include "../process/sample.h"
#include "loss.h"

using ftrl::ftrl_model;

class evaluator : public pc_task {
public:
  explicit evaluator(const trainer_option &opt);
  void loadTrainedModel(std::shared_ptr<ftrl_model> &trainModel);
  double predict(const std::vector<std::tuple<int, std::string, double>> &x);
  void print_metrics();
  ~evaluator();

private:
  std::shared_ptr<ftrl_model> eModel;
  int nthreads;
  void run_task(std::vector<std::string> &dataBuffer, int t) override;
  std::unique_ptr<long double[]> sum;
  std::unique_ptr<long[]> num;
  // std::mutex eval_mtx;
};

evaluator::evaluator(const trainer_option &opt): pc_task(opt.thread_num, opt.cmd),
    nthreads(opt.thread_num) {
  int n = opt.thread_num;
  sum = std::unique_ptr<long double[]>(new long double[n]);
  num = std::unique_ptr<long[]>(new long[n]);
  for (int i = 0; i < n; i++) {
    sum[i] = 0.0;
    num[i] = 0;
  }
};

void evaluator::run_task(std::vector<std::string> &dataBuffer, int t) {
  for (const auto &i : dataBuffer) {
    sample singleSample(i);
    double pred = predict(singleSample.x);
    sum[t] += loss(singleSample.y, pred);
    num[t] += 1;
  }
}

void evaluator::loadTrainedModel(std::shared_ptr<ftrl_model> &trainModel) {
  eModel = trainModel;
}

double evaluator::predict(const std::vector<std::tuple<int, std::string, double>> &x) {
  return eModel->predict(x, false);
}

void evaluator::print_metrics() {
  long double ssum = 0.0;
  long nnum = 0;
  for (int i = 0; i < nthreads; i++) {
    ssum += sum[i];
    nnum += num[i];
  }
  std::cout << "log loss: " << ssum / nnum << std::endl;
  for (int i = 0; i < nthreads; i++) {
    sum[i] = 0.0;
    num[i] = 0;
  }
}

evaluator::~evaluator() {
  sum.release();
  num.release();
}

#endif //FTRL_FFM_EVALUATE_H