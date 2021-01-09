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

namespace ftrl {

class FtrlOnline : public PcTask {
public:
  explicit FtrlOnline(const trainer_option &opt);
  void run_task(std::vector<std::string> &dataBuffer, int t) override;
  bool loadModel(std::ifstream &ifs);
  void outputModel(std::ofstream &ofs);
  std::shared_ptr<ftrl_model> pModel;

private:
  float w_alpha, w_beta, w_l1, w_l2;
};

FtrlOnline::FtrlOnline(const trainer_option &opt)
    : PcTask(opt.thread_num, opt.cmd), w_alpha(opt.w_alpha), w_beta(opt.w_beta),
      w_l1(opt.w_l1), w_l2(opt.w_l2) {
  if (opt.model_type == "LR") {
    pModel = std::make_shared<ftrl_model>(opt.init_mean, opt.init_stddev, opt.model_type);
  } else if (opt.model_type == "FM") {
    pModel = std::make_shared<ftrl_model>(
        opt.init_mean, opt.init_stddev, opt.n_factors, opt.model_type);
  } else if (opt.model_type == "FFM") {
    pModel = std::make_shared<ftrl_model>(
        opt.init_mean, opt.init_stddev, opt.n_factors, opt.n_fields, opt.model_type);
  }
}

void FtrlOnline::run_task(std::vector<std::string> &dataBuffer, int t) {
  for (const auto &rawData : dataBuffer) {
    Sample sample;
    Parser::parse(rawData, sample);
    pModel->train(sample.x, sample.y, w_alpha, w_beta, w_l1, w_l2);
  }
}

bool FtrlOnline::loadModel(std::ifstream &ifs) {
  return pModel->loadModel(ifs);
}

void FtrlOnline::outputModel(std::ofstream &ofs) {
  return pModel->outputModel(ofs);
}

}

#endif //FTRL_FFM_FTRL_ONLINE_H