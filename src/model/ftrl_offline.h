#ifndef FTRL_FFM_FTRL_OFFLINE_H
#define FTRL_FFM_FTRL_OFFLINE_H

#include <algorithm>
#include <cassert>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <omp.h>

#include "ftrl_model.h"
#include "../eval/loss.h"
#include "../reader/parser.h"
#include "../threading/thread_pool.h"
#include "../utils/cmd_option.h"

namespace ftrl {

class FtrlOffline {
public:
  explicit FtrlOffline(const config_options &opt);
  [[maybe_unused]] double one_epoch_openmp(std::vector<Sample> &samples, bool train);
  double one_epoch_batch(std::vector<Sample> &samples, bool train);
  double one_epoch_pool(std::vector<Sample> &samples, bool train);
  std::shared_ptr<FtrlModel> pModel;

private:
  float w_alpha, w_beta, w_l1, w_l2;
  int n_threads;
  std::shared_ptr<ThreadPool> thread_pool;
  bool use_pool = true;
};

FtrlOffline::FtrlOffline(const config_options &opt)
    : w_alpha(opt.w_alpha), w_beta(opt.w_beta), w_l1(opt.w_l1),
      w_l2(opt.w_l2), n_threads(opt.thread_num) {
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
  if (use_pool) {
    thread_pool = std::make_shared<ThreadPool>(n_threads);
  }
}

[[maybe_unused]] double FtrlOffline::one_epoch_openmp(std::vector<Sample> &samples, bool train) {
  size_t len = samples.size();
  double total_loss = 0.0;
  std::vector<int> indices(len);
  std::iota(indices.begin(), indices.end(), 0);
  if (train) {
    shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));
  }
  omp_set_num_threads(n_threads);
// #pragma omp parallel for schedule(static) reduction(+: total_loss, total_count) shared(samples, model_ptr, w_alpha, w_beta, w_l1, w_l2) private(i, sample) num_threads(4)
  if (train) {
#pragma omp parallel for simd reduction(+: total_loss) schedule(static)
    for (size_t i = 0; i < len; i++) {
      const Sample &sample = samples[i];
      const float logit = pModel->train(sample.x, sample.y, w_alpha, w_beta, w_l1, w_l2);
      total_loss += loss(sample.y, logit);
    }
  } else {
#pragma omp parallel for simd reduction(+: total_loss) schedule(static)
    for (size_t i = 0; i < len; i++) {
      const Sample &sample = samples[i];
      const float logit = pModel->predict(sample.x, false);
      total_loss += loss(sample.y, logit);
    }
  }
  return total_loss / static_cast<double>(len);
}

double FtrlOffline::one_epoch_batch(std::vector<Sample> &samples, bool train) {
  const size_t total_num = samples.size();
  std::vector<int> indices(total_num);
  std::iota(indices.begin(), indices.end(), 0);
  if (train) {
    shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));
  }
  std::vector<double> losses(n_threads);
  auto one_thread = [&](size_t idx, size_t start, size_t end) {
    double tmp_loss = 0.0;
    for (auto i = start; i < end; i++) {
      const Sample &sample = samples[i];
      auto logit = train ?
          pModel->train(sample.x, sample.y, w_alpha, w_beta, w_l1, w_l2) :
          pModel->predict(sample.x, false);
      tmp_loss += loss(sample.y, logit);
    }
    losses[idx] = tmp_loss;
  };

  std::vector<std::thread> total_threads;
  const size_t unit = std::ceil(total_num / n_threads);
  assert(unit > 0);
  for (size_t i = 0; i < n_threads; i++) {
    const size_t start = i * unit;
    const size_t end = std::min(start + unit, total_num);
    total_threads.emplace_back([=] { one_thread(i, start, end); });
  }
  for (auto &t : total_threads) {
    t.join();
  }
  const double total_loss = std::accumulate(losses.begin(), losses.end(), 0.0);
  return total_loss / static_cast<double>(total_num);
}

double FtrlOffline::one_epoch_pool(std::vector<Sample> &samples, bool train) {
  const size_t total_num = samples.size();
  std::vector<int> indices(total_num);
  std::iota(indices.begin(), indices.end(), 0);
  if (train) {
    shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));
  }
  std::vector<double> losses(n_threads);
  auto one_thread = [&](size_t idx, size_t start, size_t end) {
    double tmp_loss = 0.0;
    for (auto i = start; i < end; i++) {
      const Sample &sample = samples[i];
      auto logit = train ?
                   pModel->train(sample.x, sample.y, w_alpha, w_beta, w_l1, w_l2) :
                   pModel->predict(sample.x, false);
      tmp_loss += loss(sample.y, logit);
    }
    losses[idx] = tmp_loss;
  };

  const size_t unit = std::ceil(total_num / n_threads);
  assert(unit > 0);
  for (int i = 0; i < n_threads; i++) {
    const size_t start = i * unit;
    const size_t end = std::min(start + unit, total_num);
    thread_pool->enqueue(std::bind(one_thread, i, start, end));
  }
  thread_pool->synchronize(n_threads);
  const double total_loss = std::accumulate(losses.begin(), losses.end(), 0.0);
  return total_loss / static_cast<double>(total_num);
}

}

#endif //FTRL_FFM_FTRL_OFFLINE_H