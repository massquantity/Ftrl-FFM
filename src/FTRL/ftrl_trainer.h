#ifndef FIRST_FTRL_TRAINER_H
#define FIRST_FTRL_TRAINER_H

#include <cstdio>
#include <queue>
#include <vector>
#include "ftrl_model.h"
#include "utils.h"
#include "lr_sample.h"
#include "pc_task.h"

struct trainer_option
{
    string model_path, data_path;
    double init_mean, init_stddev;
    double w_alpha, w_beta, w_l1, w_l2;
    int thread_num, epoch;
    bool cmd;
    trainer_option(): data_path(""), init_mean(0.0), init_stddev(0.01),
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
            else if (args[i].compare("-data") == 0) {
                if (i == argc - 1)
                    throw invalid_argument("invalid command \"-data\".\n");
                data_path = args[++i];
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
                cmd = (1 == stoi(args[i]));
            }
            else
                throw invalid_argument("invalid command.\n");
        }
    }
};


class ftrl_trainer : public pc_task
{
public:
    ftrl_trainer(const trainer_option &opt);
    void run_task(vector<string> &dataBuffer) override;
    bool loadModel(ifstream &ifs);
    void outputModel(ofstream &ofs);
private:
    void train(int y, const vector<pair<string, double> > &x);
    ftrl_model *pModel;
    double w_alpha, w_beta, w_l1, w_l2;
};

ftrl_trainer::ftrl_trainer(const trainer_option &opt)
{
    w_alpha = opt.w_alpha;
    w_beta = opt.w_beta;
    w_l1 = opt.w_l1;
    w_l2 = opt.w_l2;
    pModel = new ftrl_model(opt.init_mean, opt.init_stddev);
}

void ftrl_trainer::run_task(vector<string> &dataBuffer)
{
    for (int i = 0; i < dataBuffer.size(); ++i) {
        lr_sample sample(dataBuffer[i]);
    //    for (auto &j : sample.x)
    //        cout << j.first << " " << j.second << endl;
        train(sample.y, sample.x);
    }
}

bool ftrl_trainer::loadModel(ifstream &ifs)
{
    return pModel->loadModel(ifs);
}

void ftrl_trainer::outputModel(ofstream &ofs)
{
    return pModel->outputModel(ofs);
}

void ftrl_trainer::train(int y, const vector<pair<string, double> > &x)
{
//    for (int k = 0; k < x.size(); ++k)
//        cout << x[k].first << endl;
    ftrl_model_unit *thetaBias = pModel->getOrInitModelUnitBias();
    vector<ftrl_model_unit *> theta(x.size(), nullptr);
    int xLen = x.size();
    for (int i = 0; i < xLen; ++i) {
        const string &index = x[i].first;
        theta[i] = pModel->getOrInitModelUnit(index);
    }
    for (int i = 0; i <= xLen; ++i) {
        ftrl_model_unit &mu = i < xLen ? *(theta[i]) : *thetaBias;
        mu.mtx.lock();
        if (fabs(mu.w_zi) <= w_l1)
            mu.wi = 0.0;
        else {
            mu.wi = (-1) *
                    (1 / (w_l2 + (w_beta + sqrt(mu.w_ni)) / w_alpha)) *
                    (mu.w_zi - utils::sgn(mu.w_zi) * w_l1);
        }
        mu.mtx.unlock();
    }

    double bias = thetaBias->wi;
    double p = pModel->predict(x, bias, theta);
    double mult = y * (1 / (1 + exp(-p * y)) - 1);
    for (int i = 0; i <= xLen; ++i) {
        ftrl_model_unit &mu = i < xLen ? *(theta[i]) : *thetaBias;
        double xi = i < xLen ? x[i].second : 1.0;
        mu.mtx.lock();
        double w_gi = mult * xi;
        double w_si = 1 / w_alpha * (sqrt(mu.w_ni + w_gi * w_gi) - sqrt(mu.w_ni));
        mu.w_zi += w_gi - w_si * mu.wi;
        mu.w_ni += w_gi * w_gi;
        mu.mtx.unlock();
    }

//    pModel->debugPrintModel();
}










#endif //FIRST_FTRL_TRAINER_H
