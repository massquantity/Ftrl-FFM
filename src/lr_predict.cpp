#include <iostream>
#include <fstream>
#include "model/ftrl_predictor.h"
#include "threading/pc_threading.h"
using namespace std;

vector<string> argv_to_args(int argc, char *argv[])
{
    vector<string> args;
    for (int i = 1; i < argc; ++i)
        args.emplace_back(string(argv[i]));
    return args;
}

int main(int argc, char *argv[])
{
    std::istream::sync_with_stdio(false);
    std::ostream::sync_with_stdio(false);
    predict_option opt;
    try {
        opt.parse_option(argv_to_args(argc, argv));
    } catch (const invalid_argument &e) {
        cout << "invalid_argument:" << e.what() << endl;
        cout << predict_help() << endl;
        return EXIT_FAILURE;
    }

    ifstream f_model(opt.model_path.c_str(), ifstream::in);
    ofstream f_outpout(opt.output_path.c_str(), ofstream::out);
    ftrl_predicter predictor(f_model, f_outpout);
    pc_threading pct;
    if (!opt.cmd) {
        pct.openFile(opt.data_path);
        pct.init(predictor, opt.thread_num, opt.cmd);
        pct.run();
    }
    else {
        pct.init(predictor, opt.thread_num, opt.cmd);
        pct.run();
    }
    f_model.close();
    f_outpout.close();
}






