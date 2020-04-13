#ifndef FIRST_CMD_OPTIONS_H
#define FIRST_CMD_OPTIONS_H

#include <string>
#include <cstdio>
#include <vector>
#include <iostream>
using namespace std;

string train_help()
{
    return string(
            "\nUsage: ./lr_train [<options>]   OR   cat sample | ./lr_train [<options>]"
            "\n"
            "\n"
            "options:\n"
            "-m <model_path>: set the output model path\n"
            "-train_data <data_path>: set the train data path\n"
            "-eval_data <data_path>: set the evaluate data path\n"
            "-init_mean <mean>: mean for parameter initialization\tdefault:0.0\n"
            "-init_stddev <stddev>: stddev for parameter initialization\tdefault:0.01\n"
            "-w_alpha <w_alpha>: alpha is one of the learning rate parameters\tdefault:0.05\n"
            "-w_beta <w_beta>: beta is one of the learning rate parameters\tdefault:1.0\n"
            "-w_l1 <w_L1_reg>: L1 regularization parameter of w\tdefault:0.1\n"
            "-w_l2 <w_L2_reg>: L2 regularization parameter of w\tdefault:5.0\n"
            "-nthreads <threads_num>: set the number of threads\tdefault:1\n"
            "-epoch <epochs>: how many epochs to train\tdefault:1\n"
            "-cmd <command line input>: whether to input data using command line\tdefault:false\n"
    );
}

struct trainer_option
{
    string model_path, train_path, eval_path;
    double init_mean, init_stddev;
    double w_alpha, w_beta, w_l1, w_l2;
    int thread_num, epoch;
    bool cmd;
    trainer_option(): train_path(""), eval_path(""), init_mean(0.0), init_stddev(0.01),
                      w_alpha(0.05), w_beta(1.0), w_l1(0.1), w_l2(5.0),
                      thread_num(1), epoch(1), cmd(false) { }

    void parse_option(const vector<string> &args)
    {
        int argc = args.size();
        if (0 == argc) throw invalid_argument("invalud command.\n");
        for (int i = 0; i < argc; ++i) {
            if (args[i].compare("-m") == 0) {
                if (i == argc - 1)
                    throw invalid_argument("invalid command \"-m\".\n");
                model_path = args[++i];
            }
            else if (args[i].compare("-train_data") == 0) {
                if (i == argc - 1)
                    throw invalid_argument("invalid command \"-train_data\".\n");
                train_path = args[++i];
            }
            else if (args[i].compare("-eval_data") == 0) {
                if (i == argc - 1)
                    throw invalid_argument("invalid command \"-eval_data\".\n");
                eval_path = args[++i];
            }
            else if (args[i].compare("-init_mean") == 0) {
                if (i == argc - 1)
                    throw invalid_argument("invalid command \"-init_mean\".\n");
                init_mean = stod(args[++i]);
            }
            else if (args[i].compare("-init_stddev") == 0) {
                if (i == argc - 1)
                    throw invalid_argument("invalid command \"-init_stddev\".\n");
                init_stddev = stod(args[++i]);
            }
            else if (args[i].compare("-w_alpha") == 0) {
                if (i == argc - 1)
                    throw invalid_argument("invalid command \"-w_alpha\".\n");
                w_alpha = stod(args[++i]);
            }
            else if (args[i].compare("-w_beta") == 0) {
                if (i == argc - 1)
                    throw invalid_argument("invalid command \"-w_beta\".\n");
                w_beta = stod(args[++i]);
            }
            else if (args[i].compare("-w_l1") == 0) {
                if (i == argc - 1)
                    throw invalid_argument("invalid command \"-w_l1\".\n");
                w_l1 = stod(args[++i]);
            }
            else if (args[i].compare("-w_l2") == 0) {
                if (i == argc - 1)
                    throw invalid_argument("invalid command \"-w_l2\".\n");
                w_l2 = stod(args[++i]);
            }
            else if (args[i].compare("-nthreads") == 0) {
                if (i == argc - 1)
                    throw invalid_argument("invalid command \"-nthreads\".\n");
                thread_num = stoi(args[++i]);
            }
            else if (args[i].compare("-epoch") == 0) {
                if (i == argc - 1)
                    throw invalid_argument("invalid command \"-epoch\".\n");
                epoch = stoi(args[++i]);
            }
            else if (args[i].compare("-cmd") == 0) {
                if (i == argc - 1)
                    throw invalid_argument("invalid command \"cmd\".\n");
                i++;
                if      (args[i] == "true")  cmd = true;
                else if (args[i] == "false") cmd = false;
                else {
                    cout << "cmd argument must be one of these: true, false" << endl;
                    exit(-1);
                }
                //    cmd = (1 == stoi(args[i]));
            }
            else
                throw invalid_argument("invalid command.\n");
        }
    }
};

string predict_help()
{
    return string(
            "\nUsage: ./lr_pred [<options>]  OR  cat sample | ./lr_predict [<options>]"
            "\n"
            "\n"
            "options:\n"
            "-m <model_path>: set the model path\n"
            "-data <data_path>: set the input data path\n"
            "-o <output_path>: set the output result path\n"
            "-nthreads <threads_num>: set the number of threads\tdefault:1\n"
            "-cmd <command line input>: whether to input data using command line\tdefault:false\n"
    );
}


struct predict_option
{
    string model_path, data_path, output_path;
    int thread_num;
    bool cmd;
    predict_option(): data_path(""), model_path(""), output_path(""),
                      thread_num(1), cmd(false) { }

    void parse_option(const vector<string> &args)
    {
        int argc = args.size();
        if (0 == argc) throw invalid_argument("invalud command.\n");
        for (int i = 0; i < argc; ++i) {
            if (args[i].compare("-m") == 0) {
                if (i == argc - 1)
                    throw invalid_argument("invalid command \"-m\".\n");
                model_path = args[++i];
            }
            else if (args[i].compare("-data") == 0) {
                if (i == argc - 1)
                    throw invalid_argument("invalid command \"-data\".\n");
                data_path = args[++i];
            }
            else if (args[i].compare("-o") == 0) {
                if (i == argc - 1)
                    throw invalid_argument("invalid command \"-o\".\n");
                output_path = args[++i];
            }
            else if (args[i].compare("-nthreads") == 0) {
                if (i == argc - 1)
                    throw invalid_argument("invalid command \"-nthreads\".\n");
                thread_num = stoi(args[++i]);
            }
            else if (args[i].compare("-cmd") == 0) {
                if (i == argc - 1)
                    throw invalid_argument("invalid command \"cmd\".\n");
                i++;
                if      (args[i] == "true") cmd = true;
                else if (args[i] == "false") cmd = false;
                else {
                    cout << "cmd argument must be one of these: true, false" << endl;
                    exit(-1);
                }
                //    cmd = (1 == stoi(args[i]));
            }
            else
                throw invalid_argument("invalid command.\n");
        }
    }
};

#endif //FIRST_CMD_OPTIONS_H
