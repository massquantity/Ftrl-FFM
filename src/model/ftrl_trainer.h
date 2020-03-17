#ifndef FIRST_FTRL_TRAINER_H
#define FIRST_FTRL_TRAINER_H

#include <cstdio>
#include <queue>
#include <vector>
#include <thread>
#include "ftrl_model.h"
#include "../utils/utils.h"
#include "../sample/lr_sample.h"
#include "../threading/pc_task.h"
#include "../utils/cmd_options.h"


class ftrl_trainer : public pc_task
{
public:
    ftrl_trainer(const trainer_option &opt);
    void run_task(vector<string> &dataBuffer, int t) override;
    bool loadModel(ifstream &ifs);
    void outputModel(ofstream &ofs);
    void evaluate(ifstream &efs);
//    void evaluate(vector<string> &dataBuffer);
//    void print_metrics();
    ftrl_model * getModel();
private:
    void train(int y, const vector<pair<string, double> > &x);
    double loss(int y, const vector<pair<string, double> > &x);
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

void ftrl_trainer::evaluate(ifstream &efs)
{
    long double sum = 0.0;
    long num = 0;
    string line;
    while (getline(efs, line)) {
        lr_sample sample(line);
        sum += loss(sample.y, sample.x);
        num += 1;
    }
    cout << "log loss: " << sum / num << endl;
}

double ftrl_trainer::loss(int y, const vector<pair<string, double> > &x)
{
    double y_hat = pModel->predict(x, pModel->muBias->wi, pModel->muMap, false);
    return -log(1 / (1 + exp(-y * y_hat)));
}

ftrl_model * ftrl_trainer::getModel()
{
    return pModel;
}

void ftrl_trainer::run_task(vector<string> &dataBuffer, int t)
{
    for (const auto & i : dataBuffer) {
        lr_sample sample(i);
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
    double p = pModel->forecast(x, bias, theta);
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
