#include "model/ftrl_offline.h"

#include "eval/loss.h"

namespace ftrl {

FtrlOffline::FtrlOffline(const config_options &opt)
    : w_alpha(opt.w_alpha),
      w_beta(opt.w_beta),
      w_l1(opt.w_l1),
      w_l2(opt.w_l2),
      n_threads(opt.thread_num) {
  if (opt.model_type == "LR") {
    model_ptr = std::make_shared<FtrlModel>(opt.init_mean, opt.init_stddev, opt.model_type);
  } else if (opt.model_type == "FM") {
    model_ptr =
        std::make_shared<FtrlModel>(opt.init_mean, opt.init_stddev, opt.n_factors, opt.model_type);
  } else if (opt.model_type == "FFM") {
    model_ptr = std::make_shared<FtrlModel>(opt.init_mean, opt.init_stddev, opt.n_factors,
                                            opt.n_fields, opt.model_type);
  } else {
    std::cout << "Invalid model_type: " << opt.model_type;
    std::cout << ", expect `LR`, `FM` or `FFM`." << std::endl;
    throw std::invalid_argument("invalid model_type");
  }

  thread_pool = std::make_shared<ThreadPool>(n_threads);
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
      const Sample &sample = samples[indices[i]];
      auto logit = train ? model_ptr->train(sample.x, sample.y, w_alpha, w_beta, w_l1, w_l2)
                         : model_ptr->predict(sample.x, false);
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
