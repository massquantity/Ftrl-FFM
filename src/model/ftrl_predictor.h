#ifndef FIRST_FTRL_PREDICTOR_H
#define FIRST_FTRL_PREDICTOR_H


#include "ftrl_model.h"
#include "../sample/lr_sample.h"
#include "../threading/pc_task.h"
#include "../utils/cmd_options.h"

class ftrl_predicter : public pc_task
{
public:
    ftrl_predicter(ifstream &_fModel, ofstream &_fPredict);
    void run_task(vector<string> &dataBuffer, int t) override;
private:
    ftrl_model *pModel;
    ofstream &fPredict;
    mutex outMtx;
};

ftrl_predicter::ftrl_predicter(ifstream &_fModel, ofstream &_fPredict): fPredict(_fPredict)
{
    pModel = new ftrl_model();
    if (!pModel->loadModel(_fModel)) {
        cout << "load model failed..." << endl;
        exit(-1);
    }
}

void ftrl_predicter::run_task(vector<string> &dataBuffer, int t)
{
    vector<string> outputVec(dataBuffer.size());
    for (int i = 0; i < dataBuffer.size(); ++i) {
        lr_sample sample(dataBuffer[i]);
        double score = pModel->predict(sample.x, pModel->muBias->wi, pModel->muMap);
        outputVec[i] = to_string(sample.y) + " " + to_string(score);
    }
    outMtx.lock();
    for (const auto & res : outputVec) {
        fPredict << res << endl;
    }
    outMtx.unlock();
}




#endif //FIRST_FTRL_PREDICTOR_H
