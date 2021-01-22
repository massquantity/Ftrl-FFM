#include "ftrl_task.h"
using ftrl::FtrlTask;

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
  srand(time(nullptr));
  trainer_option opt;
  try {
    opt.parse_option(argv_to_args(argc, argv));
  } catch (const std::invalid_argument &e) {
    std::cout << "invalid argument: " << e.what() << std::endl;
    std::cout << train_help() << std::endl;
    exit(EXIT_FAILURE);
  }

  FtrlTask task(opt);
  task.init();
  task.train();
}
