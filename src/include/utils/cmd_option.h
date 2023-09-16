#ifndef FTRL_FFM_CMD_OPTION_H
#define FTRL_FFM_CMD_OPTION_H

#include <string>
#include <string_view>

static const constexpr std::string_view cmd_help =
    "\nUsage: ./main [<options>]"
    "\n"
    "\n"
    "options:\n"
    "--model_path <model_path>: set the output model path\n"
    "--train_data <data_path>: set the train data path\n"
    "--eval_data <data_path>: set the eval data path\n"
    "--model_type <model_type>: LR, FM or FFM\n"
    "--init_mean <mean>: mean for parameter initialization\tdefault:0.0\n"
    "--init_stddev <stddev>: stddev for parameter initialization\tdefault:0.02\n"
    "--n_fields <n_fields>: number of fields in FFM\tdefault:8\n"
    "--n_feats <n_feats>: number of total features\tdefault:10000\n"
    "--n_factors <n_factors>: number of embed size in FM and FFM\tdefault:16\n"
    "--w_alpha <w_alpha>: alpha is one of the learning rate parameters\tdefault:1e-4\n"
    "--w_beta <w_beta>: beta is one of the learning rate parameters\tdefault:1.0\n"
    "--w_l1 <w_L1_reg>: L1 regularization parameter of w\tdefault:0.1\n"
    "--w_l2 <w_L2_reg>: L2 regularization parameter of w\tdefault:5.0\n"
    "--n_threads <threads_num>: set the number of threads\tdefault:1\n"
    "--epoch <epochs>: how many epochs to train\tdefault:1\n"
    "--online <online>: whether to online training mode\tdefault:true\n";

struct config_options {  // NOLINT
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
  int n_fields;
  int n_feats;
  int n_factors;
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
        n_fields(8),
        n_feats(10000),
        n_factors(16),
        model_type("FFM"),
        online(true) {}

  void parse_option(int argc, char *argv[]);
};

#endif  // FTRL_FFM_CMD_OPTION_H
