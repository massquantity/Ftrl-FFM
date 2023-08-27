#include "eval/evaluate.h"

#include "eval/loss.h"

namespace ftrl {

Evaluator::Evaluator(const config_options &opt)
    : PcTask(opt.thread_num, opt.cmd), n_threads(opt.thread_num) {
  const int n = opt.thread_num;
  losses = std::unique_ptr<double[]>(new double[n]);
  nums = std::unique_ptr<uint64[]>(new uint64[n]);
  for (int i = 0; i < n; i++) {
    losses[i] = 0.0;
    nums[i] = 0;
  }
  if (opt.file_type == "libsvm") {
    parser = std::make_shared<LibsvmParser>();
  } else if (opt.file_type == "libffm") {
    parser = std::make_shared<FFMParser>();
  }
}

void Evaluator::run_task(std::vector<std::string> &data_buffer, int t) {
  double tmp_loss = 0.0;
  for (const auto &rawData : data_buffer) {
    Sample sample;
    parser->parse(rawData, sample);
    const double pred = eval_model->predict(sample.x, false);
    tmp_loss += loss(sample.y, pred);
  }
  losses[t] += tmp_loss;
  nums[t] += data_buffer.size();
}

void Evaluator::load_trained_model(std::shared_ptr<FtrlModel> &train_model) {
  eval_model = train_model;
}

double Evaluator::get_loss() {
  double total_loss = 0.0;
  uint64 total_num = 0;
  for (int i = 0; i < n_threads; i++) {
    total_loss += losses[i];
    total_num += nums[i];
    losses[i] = 0.0;
    nums[i] = 0;
  }
  return total_loss / static_cast<double>(total_num);
}

Evaluator::~Evaluator() {
  losses.release();
  nums.release();
}

}  // namespace ftrl
