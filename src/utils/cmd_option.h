#ifndef FTRL_FFM_CMD_OPTION_H
#define FTRL_FFM_CMD_OPTION_H

#include <string>

std::string train_help() {
  return std::string(
      "\nUsage: ./lr_train [<options>]   OR   cat sample | ./lr_train [<options>]"
      "\n"
      "\n"
      "options:\n"
      "-m <model_path>: set the output model path\n"
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

struct trainer_option {
  std::string model_path, train_path, eval_path;
  double init_mean, init_stddev;
  double w_alpha, w_beta, w_l1, w_l2;
  int thread_num, epoch, n_factors, n_fields;
  bool cmd, k0, k1;
  trainer_option(): init_mean(0.0), init_stddev(0.01), w_alpha(0.1), w_beta(1.0),
                    w_l1(0.1), w_l2(5.0), thread_num(1), epoch(1), cmd(false),
                    n_factors(8), n_fields(15), k0(true), k1(true) { }

  void parse_option(const std::vector<std::string> &args) {
    int argc = args.size();
    if (argc == 0) throw std::invalid_argument("invalid argumnent\n");
    for (int i = 0; i < argc; i++) {
      if (args[i] == "-m") {
        if (i == argc - 1)
          throw std::invalid_argument("invalid argumnent \"-m\". \n");
        model_path = args[++i];
      }
      else if (args[i] == "-dim") {
        if (i == argc - 1)
          throw std::invalid_argument("invalid argument \"-dim\". \n");
        std::vector<std::string> strVec;
        std::string tmpStr = args[++i];
        utils::splitString(tmpStr, ",", strVec);
        if (strVec.size() != 3)
          throw std::invalid_argument("invalid argument \"-dim\". \n");
        k0 = (stoi(strVec[0]) != 0);
        k1 = (stoi(strVec[1]) != 0);
        n_factors = stoi(strVec[2]);
      }
      else if (args[i] == "n_fields") {
        if (i == argc - 1)
          throw std::invalid_argument("invalid command \"-n_fields\". \n");
        n_fields = stoi(args[++i]);
      }
      else if (args[i] == "-train_data") {
        if (i == argc - 1)
          throw std::invalid_argument("invalid command \"-train_data\".\n");
        train_path = args[++i];
      }
      else if (args[i] =="-eval_data") {
        if (i == argc - 1)
          throw std::invalid_argument("invalid command \"-eval_data\".\n");
        eval_path = args[++i];
      }
      else if (args[i] =="-init_mean") {
        if (i == argc - 1)
          throw std::invalid_argument("invalid command \"-init_mean\".\n");
        init_mean = stod(args[++i]);
      }
      else if (args[i] =="-init_stddev") {
        if (i == argc - 1)
          throw std::invalid_argument("invalid command \"-init_stddev\".\n");
        init_stddev = stod(args[++i]);
      }
      else if (args[i] =="-w_alpha") {
        if (i == argc - 1)
          throw std::invalid_argument("invalid command \"-w_alpha\".\n");
        w_alpha = stod(args[++i]);
      }
      else if (args[i] =="-w_beta") {
        if (i == argc - 1)
          throw std::invalid_argument("invalid command \"-w_beta\".\n");
        w_beta = stod(args[++i]);
      }
      else if (args[i] =="-w_l1") {
        if (i == argc - 1)
          throw std::invalid_argument("invalid command \"-w_l1\".\n");
        w_l1 = stod(args[++i]);
      }
      else if (args[i] =="-w_l2") {
        if (i == argc - 1)
          throw std::invalid_argument("invalid command \"-w_l2\".\n");
        w_l2 = stod(args[++i]);
      }
      else if (args[i] =="-nthreads") {
        if (i == argc - 1)
          throw std::invalid_argument("invalid command \"-nthreads\".\n");
        thread_num = stoi(args[++i]);
      }
      else if (args[i] == "-epoch") {
        if (i == argc - 1)
          throw std::invalid_argument("invalid command \"-epoch\".\n");
        epoch = stoi(args[++i]);
      }
      else if (args[i] =="-cmd") {
        if (i == argc - 1)
          throw std::invalid_argument("invalid command \"cmd\".\n");
        i++;
        if      (args[i] == "true")  cmd = true;
        else if (args[i] == "false") cmd = false;
        else {
          std::cout << "cmd argument must be one of these: true, false" << std::endl;
          exit(-1);
        }
      }
      else
        throw std::invalid_argument("invalid command. \n");
    }
  }
};

#endif //FTRL_FFM_CMD_OPTION_H