#include "task/ftrl_offline.h"

#include <cassert>
#include <random>
#include <thread>

#include "eval/loss.h"
#include "model/ffm.h"
#include "model/fm.h"
#include "model/lr.h"

namespace ftrl {

using timer = std::chrono::steady_clock;

FtrlOffline::FtrlOffline(const config_options &opt)
    : n_epochs(opt.epoch), n_threads(opt.thread_num) {
  if (opt.model_type == "LR") {
    model_ptr = std::make_unique<LR>(opt);
  } else if (opt.model_type == "FM") {
    model_ptr = std::make_unique<FM>(opt);
  } else if (opt.model_type == "FFM") {
    model_ptr = std::make_unique<FFM>(opt);
  } else {
    std::cout << "Invalid model_type: " << opt.model_type;
    std::cout << ", expect `LR`, `FM` or `FFM`." << std::endl;
    throw std::invalid_argument("invalid model_type");
  }

  train_data_loader = std::make_unique<Reader>(opt.file_type);
  train_data_loader->load_from_file(opt.train_path, n_threads);
  if (!opt.eval_path.empty()) {
    eval_data_loader = std::make_unique<Reader>(opt.file_type);
    eval_data_loader->load_from_file(opt.eval_path, n_threads);
  }

  thread_pool = std::make_unique<ThreadPool>(n_threads);
}

void FtrlOffline::train() {
  for (int i = 1; i <= n_epochs; i++) {
    auto train_start = timer::now();
    const double train_loss = one_epoch(train_data_loader->data, true, true);
    const double train_time = utils::compute_time(train_start);
    printf("epoch %d train time: %.4lfs, train loss: %.4lf\n", i, train_time, train_loss);
    if (eval_data_loader != nullptr) {
      evaluate(i);
    }
  }
}

void FtrlOffline::evaluate(int epoch) {
  auto eval_start = timer::now();
  const double eval_loss = one_epoch(eval_data_loader->data, false, false);
  const double eval_time = utils::compute_time(eval_start);
  printf("epoch %d eval time: %.4lfs, eval loss: %.4lf\n", epoch, eval_time, eval_loss);
}

double FtrlOffline::one_epoch(std::vector<Sample> &samples, bool train, bool use_pool) {
  const size_t total_num = samples.size();
  const size_t unit = std::ceil(static_cast<double>(total_num) / n_threads);
  assert(unit > 0);
  std::vector<int> indices(total_num);
  std::iota(indices.begin(), indices.end(), 0);
  if (train) {
    shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
  }
  std::vector<double> losses(n_threads);

  auto one_thread = [&](size_t idx, size_t start, size_t end) {
    double tmp_loss = 0.0;
    for (auto i = start; i < end; i++) {
      Sample &sample = samples[indices[i]];
      auto logit =
          train ? model_ptr->train(sample.x, sample.y) : model_ptr->predict(sample.x, false);
      tmp_loss += loss(sample.y, logit);
    }
    losses[idx] = tmp_loss;
  };

  if (use_pool) {
    for (size_t i = 0; i < n_threads; i++) {
      const size_t start = i * unit;
      const size_t end = std::min(start + unit, total_num);
      thread_pool->enqueue(one_thread, i, start, end);
    }
    thread_pool->synchronize(n_threads);
  } else {
    std::vector<std::thread> total_threads;
    for (size_t i = 0; i < n_threads; i++) {
      const size_t start = i * unit;
      const size_t end = std::min(start + unit, total_num);
      total_threads.emplace_back([=] { one_thread(i, start, end); });
    }
    std::for_each(total_threads.begin(), total_threads.end(), [](auto &t) { t.join(); });
  }
  const double total_loss = std::accumulate(losses.begin(), losses.end(), 0.0);
  return total_loss / static_cast<double>(total_num);
}

}  // namespace ftrl
