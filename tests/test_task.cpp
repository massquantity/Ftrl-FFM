#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "common.h"
#include "task/ftrl_offline.h"
#include "task/ftrl_online.h"
#include "utils/cmd_option.h"

namespace ftrl {

static config_options get_command_args() {
  config_options opt;
  opt.train_path = test_file_path;
  opt.n_fields = 4;
  opt.n_feats = 50;
  opt.n_factors = 4;
  opt.epoch = 2;
  opt.thread_num = 4;
  opt.model_type = "FFM";
  opt.file_type = "libffm";
  opt.online = false;
  return opt;
}

TEST_CASE("ftrl online training has zero weights") {
  write_test_data(test_file_path);
  ftrl::FtrlOnline online_task(get_command_args());
  online_task.train();
  CHECK(online_task.has_zero_weights());
  remove_test_file(test_file_path);
}

TEST_CASE("ftrl offline training has zero weights") {
  write_test_data(test_file_path);
  ftrl::FtrlOffline offline_task(get_command_args());
  offline_task.train();
  CHECK(offline_task.has_zero_weights());
  remove_test_file(test_file_path);
}

}  // namespace ftrl
