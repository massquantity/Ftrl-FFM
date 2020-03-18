#ifndef FIRST_EVALUATOR_H
#define FIRST_EVALUATOR_H

#include <fstream>
#include "../model/ftrl_model.h"
#include "../threading/pc_task.h"
#include "../sample/lr_sample.h"

using namespace std;

class evaluator : public pc_task
{
public:
    evaluator(ftrl_model *trainModel, int n_threads);
    void print_metrics();
private:
    ftrl_model *eModel;
    void run_task(vector<string> &dataBuffer, int t) override;
    double loss(int y, const vector<pair<string, double> > &x);
    int nthreads;
    long double *sum;
    long *num;
//    mutex eval_mtx;
};

evaluator::evaluator(ftrl_model *trainModel, int n_threads)
{
    nthreads = n_threads;
    eModel = trainModel;
    sum = new long double[nthreads]{0.0};
    num = new long[nthreads]{0};
}

void evaluator::run_task(vector<string> &dataBuffer, int t)
{
    string line;
    for (const auto & i : dataBuffer) {
        lr_sample sample(i);
        double single_loss = loss(sample.y, sample.x);
    //    eval_mtx.lock();
        sum[t] += single_loss;
        num[t] += 1;
    //    eval_mtx.unlock();
    }
}

void evaluator::print_metrics()
{
    long double ssum = 0.0;
    long nnum = 0;
    for (int i = 0; i < nthreads; ++i) {
        ssum += sum[i];
        nnum += num[i];
    }
    cout << "log loss: " << ssum / nnum << endl;
    delete [] sum;
    delete [] num;
}

double evaluator::loss(int y, const vector<pair<string, double> > &x)
{
    double y_hat = eModel->predict(x, eModel->muBias->wi, eModel->muMap, false);
    return -log(1 / (1 + exp(-y * y_hat)));
}


#endif //FIRST_EVALUATOR_H
