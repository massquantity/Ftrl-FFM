#ifndef FTRL_FFM_FTRL_ONLINE_H
#define FTRL_FFM_FTRL_ONLINE_H

#include <cmath>
#include <vector>
#include <memory>
#include <tuple>
#include "ftrl_model.h"
#include "../reader/parser.h"
#include "../threading/pc_task.h"
#include "../utils/cmd_option.h"
#include "../eval/loss.h"

namespace ftrl {

class FtrlOnline : public PcTask {
public:
  explicit FtrlOnline(const trainer_option &opt);
  void run_task(std::vector<std::string> &data_buffer, int t) override;
  double get_loss();
  bool loadModel(std::ifstream &ifs);
  void outputModel(std::ofstream &ofs);
  std::shared_ptr<FtrlModel> pModel;

private:
  float w_alpha, w_beta, w_l1, w_l2;
  int n_threads;
  std::vector<double> losses;
  std::vector<long> nums;
  std::shared_ptr<Parser> parser;
};

FtrlOnline::FtrlOnline(const trainer_option &opt)
    : PcTask(opt.thread_num, opt.cmd), w_alpha(opt.w_alpha), w_beta(opt.w_beta),
      w_l1(opt.w_l1), w_l2(opt.w_l2), n_threads(opt.thread_num) {
  if (opt.model_type == "LR") {
    pModel = std::make_shared<FtrlModel>(
        opt.init_mean, opt.init_stddev, opt.model_type);
  } else if (opt.model_type == "FM") {
    pModel = std::make_shared<FtrlModel>(
        opt.init_mean, opt.init_stddev, opt.n_factors, opt.model_type);
  } else if (opt.model_type == "FFM") {
    pModel = std::make_shared<FtrlModel>(
        opt.init_mean, opt.init_stddev, opt.n_factors, opt.n_fields, opt.model_type);
  }
  if (opt.file_type == "libsvm") {
    parser = std::make_shared<LibsvmParser>();
  } else if (opt.file_type == "libffm") {
    parser = std::make_shared<FFMParser>();
  }
  losses.resize(n_threads);
  nums.resize(n_threads);
}

void FtrlOnline::run_task(std::vector<std::string> &data_buffer, int t) {
  double tmp_loss = 0.0;
  for (const auto &rawData : data_buffer) {
    Sample sample;
    parser->parse(rawData, sample);
    auto logit = pModel->train(sample.x, sample.y, w_alpha, w_beta, w_l1, w_l2);
    tmp_loss += loss(sample.y, logit);
  }
  losses[t] += tmp_loss;
  nums[t] += data_buffer.size();
}

double FtrlOnline::get_loss() {
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

bool FtrlOnline::loadModel(std::ifstream &ifs) {
  return pModel->loadModel(ifs);
}

void FtrlOnline::outputModel(std::ofstream &ofs) {
  return pModel->outputModel(ofs);
}

}

#endif //FTRL_FFM_FTRL_ONLINE_H