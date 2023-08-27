#ifndef FTRL_FFM_CMD_OPTION_H
#define FTRL_FFM_CMD_OPTION_H

#include <algorithm>
#include <cassert>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// todo: change options
static const constexpr char *cmd_help =
    "\nUsage: ./lr_train [<options>]   OR   cat sample | ./lr_train [<options>]"
    "\n"
    "\n"
    "options:\n"
    "-model_path <model_path>: set the output model path\n"
    "-train_data <data_path>: set the train data path\n"
    "-eval_data <data_path>: set the eval data path\n"
    "-init_mean <mean>: mean for parameter initialization\tdefault:0.0\n"
    "-init_stddev <stddev>: stddev for parameter initialization\tdefault:0.01\n"
    "-w_alpha <w_alpha>: alpha is one of the learning rate parameters\tdefault:0.1\n"
    "-w_beta <w_beta>: beta is one of the learning rate parameters\tdefault:1.0\n"
    "-w_l1 <w_L1_reg>: L1 regularization parameter of w\tdefault:0.1\n"
    "-w_l2 <w_L2_reg>: L2 regularization parameter of w\tdefault:5.0\n"
    "-nthreads <threads_num>: set the number of threads\tdefault:1\n"
    "-epoch <epochs>: how many epochs to train\tdefault:1\n"
    "-cmd <command line input>: whether to input data using command line\tdefault:false\n";

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

static inline void split_string(const std::string &line, const std::string &delimiter,
                                std::vector<std::string> &v) {
  std::string::size_type begin = line.find_first_not_of(delimiter, 0);
  std::string::size_type end = line.find_first_of(delimiter, begin);
  while (begin != std::string::npos || end != std::string::npos) {
    v.push_back(line.substr(begin, end - begin));
    begin = line.find_first_not_of(delimiter, end);
    end = line.find_first_of(delimiter, begin);
  }
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
  split_string(line, " ", split_line);
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

struct config_options {  // NOLINT(altera-struct-pack-align)
  std::string model_path;
  std::string train_path;
  std::string eval_path;
  std::string model_type;
  std::string file_type;
  float init_mean;
  float init_stddev;
  float w_alpha;
  float w_beta;
  float w_l1;
  float w_l2;
  int thread_num;
  int epoch;
  int n_factors;
  int n_fields;
  bool cmd;
  bool online;

  config_options()
      : init_mean(0.0),
        init_stddev(0.02),
        w_alpha(1e-4),
        w_beta(1.0),
        w_l1(0.1),
        w_l2(5.0),
        thread_num(1),
        epoch(1),
        cmd(false),
        n_factors(16),
        n_fields(0),
        model_type("FFM"),
        online(true) {}

  void parse_option(int argc, char *argv[]) {
    std::vector<std::string> args = argv_to_args(argc, argv);
    const size_t len = args.size();
    assert(len % 2 == 0);
    int i = 0;
    while (i < len) {
      if (args[i] == "-model_path") {
        model_path = args[i + 1];
      } else if (args[i] == "-model_type") {
        model_type = upper(args[i + 1]);
      } else if (args[i] == "-online") {
        online = assign_bool(args[i + 1]);
      } else if (args[i] == "-dim") {
        n_factors = stoi(args[i + 1]);
      } else if (args[i] == "-n_fields") {
        n_fields = stoi(args[i + 1]);
      } else if (args[i] == "-n_factors") {
        n_factors = stoi(args[i + 1]);
      } else if (args[i] == "-train_data") {
        train_path = args[i + 1];
      } else if (args[i] == "-eval_data") {
        eval_path = args[i + 1];
      } else if (args[i] == "-init_mean") {
        init_mean = stof(args[i + 1]);
      } else if (args[i] == "-init_stddev") {
        init_stddev = stof(args[i + 1]);
      } else if (args[i] == "-w_alpha") {
        w_alpha = stof(args[i + 1]);
      } else if (args[i] == "-w_beta") {
        w_beta = stof(args[i + 1]);
      } else if (args[i] == "-w_l1") {
        w_l1 = stof(args[i + 1]);
      } else if (args[i] == "-w_l2") {
        w_l2 = stof(args[i + 1]);
      } else if (args[i] == "-n_threads") {
        thread_num = stoi(args[i + 1]);
      } else if (args[i] == "-n_epochs") {
        epoch = stoi(args[i + 1]);
      } else if (args[i] == "-cmd") {
        cmd = assign_bool(args[i + 1]);
      } else {
        std::cerr << "unknown argument: " << args[i] << std::endl;
        throw std::invalid_argument("invalid command. \n");
      }
      i += 2;
    }

    file_type = detect_file_type(train_path);
    if (model_type == "FFM" && file_type != "libffm") {
      std::cerr << "FFM model requires libffm data format..." << std::endl;
      exit(EXIT_FAILURE);  // NOLINT
    }
  }
};

#endif  // FTRL_FFM_CMD_OPTION_H
