#ifndef FTRL_FFM_CMD_OPTION_H
#define FTRL_FFM_CMD_OPTION_H

#include <cassert>
#include <string>
#include "../reader/file_utils.h"

std::string train_help() {
  return std::string(
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
      "-cmd <command line input>: whether to input data using command line\tdefault:false\n"
  );
}

static bool assignBool(std::string arg) {
  std::transform(arg.begin(), arg.end(), arg.begin(),
                 [&](char c) { return tolower(c); });
  return arg == "true" || arg == "1";
}

static std::vector<std::string> argv_to_args(int argc, char *argv[]) {
  std::vector<std::string> args;
  for (int i = 1; i < argc; i++) {
    args.emplace_back(std::string(argv[i]));
  }
  return args;
}

struct trainer_option {
  std::string model_path, train_path, eval_path, model_type, file_type;
  float init_mean, init_stddev;
  float w_alpha, w_beta, w_l1, w_l2;
  int thread_num, epoch, n_factors, n_fields;
  bool cmd, online;
  trainer_option(): init_mean(0.0), init_stddev(0.01), w_alpha(0.1), w_beta(1.0),
                    w_l1(0.1), w_l2(5.0), thread_num(1), epoch(1), cmd(false),
                    n_factors(8), n_fields(0), model_type("FFM"), online(true) { }

  void parse_option(int argc, char *argv[]) {
    std::vector<std::string> args = argv_to_args(argc, argv);
    size_t len = args.size();
    assert( len % 2 == 0);
    int i = 0;
    while (i < len) {
      if (args[i] == "-model_path") {
        model_path = args[i + 1];
      } else if (args[i] == "-model_type") {
        model_type = args[i + 1];
      } else if (args[i] == "-online") {
        online = assignBool(args[i + 1]);
      } else if (args[i] == "-dim") {
        n_factors = stoi(args[i + 1]);
      } else if (args[i] == "-n_fields") {
        n_fields = stoi(args[i + 1]);
      } else if (args[i] == "-train_data") {
        train_path = args[i + 1];
      } else if (args[i] =="-eval_data") {
        eval_path = args[i + 1];
      } else if (args[i] =="-init_mean") {
        init_mean = stof(args[i + 1]);
      } else if (args[i] =="-init_stddev") {
        init_stddev = stof(args[i + 1]);
      } else if (args[i] =="-w_alpha") {
        w_alpha = stof(args[i + 1]);
      } else if (args[i] =="-w_beta") {
        w_beta = stof(args[i + 1]);
      } else if (args[i] =="-w_l1") {
        w_l1 = stof(args[i + 1]);
      } else if (args[i] =="-w_l2") {
        w_l2 = stof(args[i + 1]);
      } else if (args[i] =="-nthreads") {
        thread_num = stoi(args[i + 1]);
      } else if (args[i] == "-epoch") {
        epoch = stoi(args[i + 1]);
      } else if (args[i] =="-cmd") {
        cmd = assignBool(args[i + 1]);
      } else {
        std::cerr << "unknown argument: " << args[i] << std::endl;
        throw std::invalid_argument("invalid command. \n");
      }
      i += 2;
    }

    file_type = getFileType(train_path);
    if (model_type == "FFM" && file_type != "libffm") {
      std::cerr << "FFM model needs libffm data format..." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
};

#endif //FTRL_FFM_CMD_OPTION_H