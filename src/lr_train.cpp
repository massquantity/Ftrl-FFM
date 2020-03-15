#include <iostream>
#include <fstream>
#include <chrono>
#include <unistd.h>
#include "model/ftrl_trainer.h"
#include "threading/pc_threading.h"
#include "eval/evaluator.h"
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
    srand(time(NULL));
    trainer_option opt;
    try {
        opt.parse_option(argv_to_args(argc, argv));
    }
    catch(const invalid_argument &e) {
        cout << "invalid_argument:" << e.what() << endl;
        cout << train_help() << endl;
        return EXIT_FAILURE;
    }

    ftrl_trainer trainer(opt);
    pc_threading pct;
    pc_threading pct_eval;
    if (!opt.cmd) {
        int count = 0;
        pct.openFile(opt.train_path);
        if (!opt.eval_path.empty())
            pct_eval.openFile(opt.eval_path);
        while (0 < opt.epoch--) {
            auto start0 = chrono::steady_clock::now();
            pct.init(trainer, opt.thread_num, opt.cmd);
            pct.run();
            pct.rewindFile();
            auto end0 = chrono::steady_clock::now();
            auto dur0 = chrono::duration_cast<chrono::nanoseconds>(end0 - start0).count();
            cout << "epoch " << ++count << " train time: " <<
                (double)dur0 * chrono::nanoseconds::period::num /
                chrono::nanoseconds::period::den << "s" << endl;
/*
            auto start1 = chrono::steady_clock::now();
            ifstream f_eval("../lr_files/test2.txt", ifstream::in);
            trainer.evaluate(f_eval);
            f_eval.close();
            auto end1 = chrono::steady_clock::now();
            auto dur1 = chrono::duration_cast<chrono::nanoseconds>(end1 - start1).count();
            cout << "eval1 time: " << (double)dur1 * chrono::nanoseconds::period::num /
                    chrono::nanoseconds::period::den << "s" << endl;
*/
            if (!opt.eval_path.empty()) {
            //    sleep(1);
                auto start = chrono::steady_clock::now();
                evaluator eval(trainer.getModel(), opt.thread_num);
                pct_eval.init(eval, opt.thread_num, opt.cmd);
                pct_eval.run();
                pct_eval.rewindFile();
                eval.print_metrics();
                auto end = chrono::steady_clock::now();
                auto dur = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                cout << "eval time: " << (double)dur * chrono::nanoseconds::period::num /
                                          chrono::nanoseconds::period::den << "s" << endl;
            }
        }
    }
    else {
        pct.init(trainer, opt.thread_num, opt.cmd);
        pct.run();
    }

    ofstream f_model(opt.model_path.c_str(), ofstream::out);
    trainer.outputModel(f_model);
    f_model.close();
    return 0;
}





