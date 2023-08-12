#include "task/ftrl_task.h"

int main(int argc, char *argv[]) {
  std::istream::sync_with_stdio(false);
  std::ostream::sync_with_stdio(false);
  std::srand(std::time(nullptr)); // NOLINT
  config_options opt;
  try {
    opt.parse_option(argc, argv);
  } catch (const std::invalid_argument &e) {
    std::cout << "invalid argument: " << e.what() << std::endl;
    std::cout << train_help() << std::endl;
    exit(EXIT_FAILURE);  // NOLINT
  }

  ftrl::FtrlTask task(opt);
  task.init();
  task.train();
}
