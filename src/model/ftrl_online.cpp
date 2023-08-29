#include "model/ftrl_online.h"

#include "eval/loss.h"
#include "model/ffm.h"
#include "model/fm.h"
#include "model/lr.h"

namespace ftrl {

FtrlOnline::FtrlOnline(const config_options &opt)
    : PcTask(opt.thread_num, opt.cmd), n_threads(opt.thread_num) {
  if (opt.model_type == "LR") {
    model_ptr = std::make_shared<LR>(opt);
  } else if (opt.model_type == "FM") {
    model_ptr = std::make_shared<FM>(opt);
  } else if (opt.model_type == "FFM") {
    model_ptr = std::make_shared<FFM>(opt);
  } else {
    std::cout << "Invalid model_type: " << opt.model_type;
    std::cout << ", expect `LR`, `FM` or `FFM`." << std::endl;
    throw std::invalid_argument("invalid model_type");
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
    auto logit = model_ptr->train(sample.x, sample.y);
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

}  // namespace ftrl
