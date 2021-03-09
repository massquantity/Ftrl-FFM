#ifndef FTRL_FFM_FTRL_OFFLINE_H
#define FTRL_FFM_FTRL_OFFLINE_H

#include <algorithm>
#include <numeric>
#include <random>
#include <omp.h>
#include <cassert>
#include <memory>
#include <thread>
#include "ftrl_model.h"
#include "../utils/cmd_option.h"
#include "../reader/parser.h"
#include "../eval/loss.h"

namespace ftrl {

class FtrlOffline {
public:
  explicit FtrlOffline(const trainer_option &opt);
  double oneEpoch(std::vector<Sample> &samples, bool train = true);
  double oneEpochBatch(std::vector<Sample> &samples, bool train = true);
  std::shared_ptr<ftrl_model> pModel;

private:
  float w_alpha, w_beta, w_l1, w_l2;
  int n_threads;
};

FtrlOffline::FtrlOffline(const trainer_option &opt)
    : w_alpha(opt.w_alpha), w_beta(opt.w_beta), w_l1(opt.w_l1),
      w_l2(opt.w_l2), n_threads(opt.thread_num) {
  if (opt.model_type == "LR") {
    pModel = std::make_shared<ftrl_model>(
        opt.init_mean, opt.init_stddev, opt.model_type);
  } else if (opt.model_type == "FM") {
    pModel = std::make_shared<ftrl_model>(
        opt.init_mean, opt.init_stddev, opt.n_factors, opt.model_type);
  } else if (opt.model_type == "FFM") {
    pModel = std::make_shared<ftrl_model>(
        opt.init_mean, opt.init_stddev, opt.n_factors, opt.n_fields, opt.model_type);
  }
}

double FtrlOffline::oneEpoch(std::vector<Sample> &samples, bool train) {
  size_t len = samples.size();
  double total_loss = 0.0;
  std::vector<int> indices(len);
  std::iota(indices.begin(), indices.end(), 0);
  if (train) {
    shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));
  }
  omp_set_num_threads(n_threads);
// #pragma omp parallel for schedule(static) reduction(+: total_loss, total_count) shared(samples, pModel, w_alpha, w_beta, w_l1, w_l2) private(i, sample) num_threads(4)
  if (train) {
#pragma omp parallel for simd reduction(+: total_loss) schedule(static)
    for (size_t i = 0; i < len; i++) {
      Sample &sample = samples[i];
      float logit = pModel->train(sample.x, sample.y, w_alpha, w_beta, w_l1, w_l2);
      total_loss += loss(sample.y, logit);
    }
  } else {
#pragma omp parallel for simd reduction(+: total_loss) schedule(static)
    for (size_t i = 0; i < len; i++) {
      Sample &sample = samples[i];
      float logit = pModel->predict(sample.x, false);
      total_loss += loss(sample.y, logit);
    }
  }
  return total_loss / len;
}

double FtrlOffline::oneEpochBatch(std::vector<Sample> &samples, bool train) {
  size_t len = samples.size();
  std::vector<int> indices(len);
  std::iota(indices.begin(), indices.end(), 0);
  if (train) {
    shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));
  }
  std::vector<double> losses(n_threads);
  auto one_thread = [&](size_t idx, size_t start, size_t end) {
    double tmp_loss = 0.0;
    for (auto i = start; i < end; i++) {
      Sample &sample = samples[i];
      auto logit = train ?
          pModel->train(sample.x, sample.y, w_alpha, w_beta, w_l1, w_l2) :
          pModel->predict(sample.x, false);
      tmp_loss += loss(sample.y, logit);
    }
    losses[idx] = tmp_loss;
  };

  std::vector<std::thread> total_threads;
  size_t unit = std::ceil(len / n_threads);
  assert(unit > 0);
  for (size_t i = 0; i < n_threads; i++) {
    size_t start = i * unit;
    size_t end = std::min(start + unit, len);
    total_threads.emplace_back(std::thread([=] {
      one_thread(i, start, end);
    }));
  }
  for (auto &t : total_threads) {
    t.join();
  }
  double total_loss = std::accumulate(losses.begin(), losses.end(), 0.0);
  return total_loss / len;
}

}

#endif //FTRL_FFM_FTRL_OFFLINE_H