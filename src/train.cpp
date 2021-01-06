#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include "model/ftrl_trainer.h"
#include "eval/evaluate.h"
using ftrl::ftrl_trainer;

std::vector<std::string> argv_to_args(int argc, char *argv[]) {
  std::vector<std::string> args;
  for (int i = 1; i < argc; i++) {
    args.emplace_back(std::string(argv[i]));
  }
  return args;
}

int main(int argc, char *argv[]) {
  std::istream::sync_with_stdio(false);
  std::ostream::sync_with_stdio(false);
  srand(time(NULL));
  trainer_option opt;
  try {
    opt.parse_option(argv_to_args(argc, argv));
  } catch (const std::invalid_argument &e) {
    std::cout << "invalid argument: " << e.what() << std::endl;
    std::cout << train_help() << std::endl;
    return EXIT_FAILURE;
  }

  ftrl_trainer trainer(opt);
  evaluator eval(opt);
  if (!opt.cmd) {
    trainer.openFile(opt.train_path);
    if (!opt.eval_path.empty()) {
      eval.openFile(opt.eval_path);
    }
    for (int i = 1; i <= opt.epoch; i++) {
      auto start0 = std::chrono::steady_clock::now();
      trainer.run();
      trainer.rewindFile();
      auto end0 = std::chrono::steady_clock::now();
      auto dur0 = std::chrono::duration_cast<std::chrono::nanoseconds>(end0 - start0).count();
      std::cout << "epoch " << i << " train time: " <<
        (double)dur0 * std::chrono::nanoseconds::period::num /
        std::chrono::nanoseconds::period::den << "s" << std::endl;

      if (!opt.eval_path.empty()) {
        // sleep(1);
        auto start = std::chrono::steady_clock::now();
        eval.loadTrainedModel(trainer.pModel);
        eval.run();
        eval.rewindFile();
        eval.print_metrics();
        auto end = std::chrono::steady_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << "eval time: " <<
          (double)dur * std::chrono::nanoseconds::period::num /
          std::chrono::nanoseconds::period::den << "s" << std::endl;
      }
    }
  }

  std::ofstream f_model(opt.model_path.c_str(), std::ofstream::out);
  trainer.outputModel(f_model);
  f_model.close();
  return 0;
}
