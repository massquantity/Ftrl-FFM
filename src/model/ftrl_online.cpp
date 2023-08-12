#include "eval/loss.h"
#include "model/ftrl_online.h"

namespace ftrl {

FtrlOnline::FtrlOnline(const config_options &opt)
    : PcTask(opt.thread_num, opt.cmd), w_alpha(opt.w_alpha), w_beta(opt.w_beta),
      w_l1(opt.w_l1), w_l2(opt.w_l2), n_threads(opt.thread_num) {
  if (opt.model_type == "LR") {
    model_ptr = std::make_shared<FtrlModel>(
        opt.init_mean, opt.init_stddev, opt.model_type);
  } else if (opt.model_type == "FM") {
    model_ptr = std::make_shared<FtrlModel>(
        opt.init_mean, opt.init_stddev, opt.n_factors, opt.model_type);
  } else if (opt.model_type == "FFM") {
    model_ptr = std::make_shared<FtrlModel>(
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
    auto logit = model_ptr->train(sample.x, sample.y, w_alpha, w_beta, w_l1, w_l2);
    tmp_loss += loss(sample.y, logit);
  }
  losses[t] += tmp_loss;
  nums[t] += data_buffer.size();
}

double FtrlOnline::get_loss() {
  double total_loss = 0.0;
  uint64 total_num = 0;
  for (int i = 0; i < n_threads; i++) {
    total_loss += losses[i];
    total_num += nums[i];
  }
  for (int i = 0; i < n_threads; i++) {
    losses[i] = 0.0;
    nums[i] = 0;
  }
  return total_loss / static_cast<double>(total_num);
}

[[maybe_unused]] bool FtrlOnline::load_model(std::ifstream &ifs) {
  return model_ptr->load_model(ifs);
}

[[maybe_unused]] void FtrlOnline::output_model(std::ofstream &ofs) {
  return model_ptr->output_model(ofs);
}

}
