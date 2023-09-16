#include "utils/cmd_option.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <fstream>
#include <iostream>
#include <vector>

#include <fmt/core.h>
#include <fmt/format.h>

#include "utils//utils.h"
#include "utils/types.h"

static std::vector<std::string> argv_to_args(int argc, char *argv[]) {
  std::vector<std::string> args;
  for (int i = 1; i < argc; i++) {
    args.emplace_back(argv[i]);
  }
  return args;
}

static inline bool assign_bool(std::string arg) {
  std::transform(arg.begin(), arg.end(), arg.begin(), [&](char c) { return tolower(c); });
  return arg == "true" || arg == "1";
}

static inline std::string upper(std::string arg) {
  std::transform(arg.begin(), arg.end(), arg.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return arg;
}

static std::string detect_file_type(const std::string &file_path) {
  std::ifstream ifs(file_path);
  if (!ifs.good()) {
    std::cerr << "fail to open " << file_path << std::endl;
    exit(EXIT_FAILURE);  // NOLINT
  }
  std::string line;
  std::getline(ifs, line);
  ifs.close();

  std::vector<std::string> split_line;
  utils::split_string(line, " ", split_line);
  const std::string_view example_feature = split_line[1];
  const int64 colon_count = std::count_if(example_feature.cbegin(), example_feature.cend(),
                                          [](const char c) { return c == ':'; });

  if (colon_count == 1) {
    return "libsvm";
  } else if (colon_count == 2) {
    return "libffm";
  } else {
    std::cerr << "unknown file format..." << std::endl;
    exit(EXIT_FAILURE);  // NOLINT
  }
}

void config_options::parse_option(int argc, char *argv[]) {
  std::vector<std::string> args = argv_to_args(argc, argv);
  const size_t len = args.size();
  assert(len % 2 == 0);
  int i = 0;
  while (i < len) {
    if (args[i] == "--model_path") {
      model_path = args[i + 1];
    } else if (args[i] == "--model_type") {
      model_type = upper(args[i + 1]);
    } else if (args[i] == "--online") {
      online = assign_bool(args[i + 1]);
    } else if (args[i] == "--n_fields") {
      n_fields = std::stoi(args[i + 1]);
    } else if (args[i] == "--n_feats") {
      n_feats = std::stoi(args[i + 1]);
    } else if (args[i] == "--n_factors") {
      n_factors = std::stoi(args[i + 1]);
    } else if (args[i] == "--train_data") {
      train_path = args[i + 1];
    } else if (args[i] == "--eval_data") {
      eval_path = args[i + 1];
    } else if (args[i] == "--init_mean") {
      init_mean = std::stof(args[i + 1]);
    } else if (args[i] == "--init_stddev") {
      init_stddev = std::stof(args[i + 1]);
    } else if (args[i] == "--w_alpha") {
      w_alpha = std::stof(args[i + 1]);
    } else if (args[i] == "--w_beta") {
      w_beta = std::stof(args[i + 1]);
    } else if (args[i] == "--w_l1") {
      w_l1 = std::stof(args[i + 1]);
    } else if (args[i] == "--w_l2") {
      w_l2 = std::stof(args[i + 1]);
    } else if (args[i] == "--n_threads") {
      thread_num = std::stoi(args[i + 1]);
    } else if (args[i] == "--n_epochs") {
      epoch = std::stoi(args[i + 1]);
    } else if (args[i] == "--cmd") {
      cmd = assign_bool(args[i + 1]);
    } else {
      auto out = fmt::memory_buffer();
      fmt::format_to(std::back_inserter(out), "unknown argument: {}\n", args[i]);
      throw std::invalid_argument(out.data());
    }
    i += 2;
  }

  file_type = detect_file_type(train_path);
  if (model_type == "FFM" && file_type != "libffm") {
    std::cerr << "FFM model requires libffm data format..." << std::endl;
    exit(EXIT_FAILURE);  // NOLINT
  }
}
