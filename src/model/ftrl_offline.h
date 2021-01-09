#ifndef FTRL_FFM_FTRL_OFFLINE_H
#define FTRL_FFM_FTRL_OFFLINE_H

#include <algorithm>
#include <numeric>
#include <random>
#include <omp.h>
#include "ftrl_model.h"
#include "../utils/cmd_option.h"
#include "../reader/parser.h"
#include "../eval/loss.h"

namespace ftrl {

class FtrlOffline {
public:
  explicit FtrlOffline(const trainer_option &opt);
  double trainOneEpoch(std::vector<Sample> &samples);
  double evaluate(std::vector<Sample> &samples);
  std::shared_ptr<ftrl_model> pModel;

private:
  float w_alpha, w_beta, w_l1, w_l2;
};

FtrlOffline::FtrlOffline(const trainer_option &opt)
    : w_alpha(opt.w_alpha), w_beta(opt.w_beta), w_l1(opt.w_l1), w_l2(opt.w_l2) {
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

double FtrlOffline::trainOneEpoch(std::vector<Sample> &samples) {
  size_t len = samples.size();
  double total_loss = 0.0;
  // long total_count = 0;
  std::vector<int> indices(len);
  std::iota(indices.begin(), indices.end(), 0);
  shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));
#pragma omp parallel for schedule(static) reduction(+: total_loss)
  for (int i = 0; i < len; i++) {
    Sample &sample = samples[i];
    float logit = pModel->train(sample.x, sample.y, w_alpha, w_beta, w_l1, w_l2);
    total_loss += loss(sample.y, logit);
  }
  return total_loss / len;
}

double FtrlOffline::evaluate(std::vector<Sample> &samples) {
  double total_loss = 0.0;
  long total_count = 0;
  std::vector<int> indices(samples.size());
  std::iota(indices.begin(), indices.end(), 0);
  shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));
  for (int i : indices) {
    Sample &sample = samples[i];
    float pred = pModel->predict(sample.x, false);
    total_loss += loss(sample.y, pred);
    total_count += 1;
  }
  return total_loss / total_count;
}

}

#endif //FTRL_FFM_FTRL_OFFLINE_H