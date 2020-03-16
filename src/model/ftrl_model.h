#ifndef FIRST_FTRL_MODEL_H
#define FIRST_FTRL_MODEL_H

#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <mutex>
#include <cmath>
#include <unordered_map>
#include "../utils/utils.h"
using namespace std;

class ftrl_model_unit
{
public:
    double wi;
    double w_ni;
    double w_zi;
    mutex mtx;

    ftrl_model_unit()
    {
        wi = 0.0;
        w_ni = 0.0;
        w_zi = 0.0;
    }

    ftrl_model_unit(double mean, double stddev)
    {
        wi = utils::gaussian(mean, stddev);
        w_ni = 0.0;
        w_zi = 0.0;
    }

    ftrl_model_unit(const vector<string> &modelLine)
    {
        wi = stod(modelLine[1]);
        w_ni = 0.0;
        w_zi = 0.0;
    }

    friend inline ostream & operator << (ostream &os, const ftrl_model_unit &mu)
    {
        os << mu.wi;
        return os;
    }
};


class ftrl_model
{
public:
    double init_mean, init_stddev;
    ftrl_model_unit *muBias = nullptr;
    unordered_map<string, ftrl_model_unit *> muMap;
    ftrl_model();
    ftrl_model(double _mean, double _stddev);
    ftrl_model_unit *getOrInitModelUnit(const string & index);
    ftrl_model_unit *getOrInitModelUnitBias();
    double predict(const vector<pair<string, double> > &x, double bias,
                   unordered_map<string, ftrl_model_unit *> &theta, bool sigmoid = true);
    double forecast(const vector<pair<string, double> > &x, double bias,
                   vector<ftrl_model_unit *> &theta);  // for inner training
    void outputModel(ofstream &out);
    void debugPrintModel();
    bool loadModel(ifstream &in);

private:
    mutex mtx_weight;
    mutex mtx_bias;
};

ftrl_model::ftrl_model()
{
    init_mean = 0.0;
    init_stddev = 0.0;
}

ftrl_model::ftrl_model( double _mean, double _stddev)
{
    init_mean = _mean;
    init_stddev = _stddev;
}

ftrl_model_unit * ftrl_model::getOrInitModelUnit(const string &index)
{
    auto iter = muMap.find(index);
    if (iter == muMap.end()) {
        mtx_weight.lock();
        ftrl_model_unit *pMU = new ftrl_model_unit(init_mean, init_stddev);
        muMap.insert(make_pair(index, pMU));
        mtx_weight.unlock();
        return pMU;
    } else {
        return iter->second;
    }
}

ftrl_model_unit * ftrl_model::getOrInitModelUnitBias()
{
    if (muBias == nullptr) {
        mtx_bias.lock();
        muBias = new ftrl_model_unit();
        mtx_bias.unlock();
    }
    return muBias;
}

double ftrl_model::predict(const vector<pair<string, double> > &x, double bias,
                           unordered_map<string, ftrl_model_unit *> &theta, bool sigmoid)
{
    double result = 0.0;
    result += bias;
    for (const auto & feat : x) {
        auto iter = theta.find(feat.first);
        if (iter != theta.end())
            result += iter->second->wi * feat.second;
    }
    return sigmoid ? (1.0 / (1.0 + exp(-result))) : result;
}

double ftrl_model::forecast(const vector<pair<string, double> > &x, double bias,
                           vector<ftrl_model_unit *> &theta)
{
    double result = 0.0;
    result += bias;
    for (int i = 0; i < x.size(); ++i)
        result += theta[i]->wi * x[i].second;
    return result;
}

void ftrl_model::outputModel(ofstream &out)
{
    out << "bias " << *muBias << endl;
    for (auto &elem : muMap) {
        out << elem.first << " " << *(elem.second) << endl;
    }
}

void ftrl_model::debugPrintModel()
{
    cout << "bias " << *muBias << endl;
    for (auto iter = muMap.begin(); iter != muMap.end(); iter++)
        cout << iter->first << " " << *(iter->second) << endl;
}

bool ftrl_model::loadModel(ifstream &in)
{
    string line;
    if (!getline(in, line))  // first get Bias
        return false;
    vector<string> vec;
    utils::splitString(line, " ", &vec);
    muBias = new ftrl_model_unit(vec);
    while (getline(in, line)) {
        vec.clear();
        utils::splitString(line, " ", &vec);
        string &index = vec[0];
        ftrl_model_unit *pMU = new ftrl_model_unit(vec);
        muMap[index] = pMU;
    }
    return true;
}




#endif //FIRST_FTRL_MODEL_H
