#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>

#include <fmt/color.h>
#include <fmt/core.h>

#include "task/ftrl_offline.h"
#include "task/ftrl_online.h"
#include "utils/cmd_option.h"

int main(int argc, char *argv[]) {
  std::istream::sync_with_stdio(false);
  std::ostream::sync_with_stdio(false);
  std::srand(std::time(nullptr));  // NOLINT
  config_options opt;
  try {
    opt.parse_option(argc, argv);
  } catch (const std::invalid_argument &e) {
    fmt::println(stderr, "invalid argument: {}", e.what());
    fmt::print(fmt::emphasis::bold | fmt::emphasis::italic, cmd_help);
    exit(EXIT_FAILURE);            // NOLINT
  }

  if (opt.online) {
    ftrl::FtrlOnline task(opt);
    task.train();
  } else {
    ftrl::FtrlOffline task(opt);
    task.train();
  }
  return 0;
}
