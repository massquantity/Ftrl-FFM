#ifndef FTRL_FFM_FTRL_MODEL_H
#define FTRL_FFM_FTRL_MODEL_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include "../utils/utils.h"

namespace ftrl {

struct ftrl_model_unit {
  double wi;
  double w_ni;
  double w_zi;
  std::vector<double> vi;
  std::vector<double> v_ni;
  std::vector<double> v_zi;
  int fi;
  std::mutex mtx;

  ftrl_model_unit() {  // init bias
    wi = 0.0;
    w_ni = 0.0;
    w_zi = 0.0;
  }

  ftrl_model_unit(double mean, double stddev, int n_factors, int n_fields, int field_id) {
    wi = utils::gaussian(mean, stddev);
    w_ni = 0.0;
    w_zi = 0.0;
    fi = field_id;
    int v_size = n_factors * n_fields;
    vi.resize(v_size);
    v_ni.resize(v_size);
    v_zi.resize(v_size);
    for (int f = 0; f < v_size; f++) {
      vi[f] = utils::gaussian(mean, stddev);
      v_ni[f] = 0.0;
      v_zi[f] = 0.0;
    }
  }

  explicit ftrl_model_unit(const std::string &value) {  // todo
    wi = stod(value);
    w_ni = 0.;
    w_zi = 0.;
  }

  void reinit_vi(double mean, double stddev) {
    for (double &f : vi) {
      f = utils::gaussian(mean, stddev);
    }
  }

  friend inline std::ostream &operator<<(std::ostream &os,  // todo
                                         const ftrl_model_unit &mu) {
    os << mu.wi;
    return os;
  }
};

class ftrl_model {
public:
  int n_factors, n_fields;
  double init_mean, init_stddev;
  std::vector<double> sum_vx;
  std::shared_ptr<ftrl_model_unit> mBias;
  std::unordered_map<std::string, std::shared_ptr<ftrl_model_unit>> mWeight;
  explicit ftrl_model(int _n_factors, int _n_fields);
  ftrl_model(double _mean, double _stddev, int _n_factors, int _n_fields);
  std::shared_ptr<ftrl_model_unit>& getOrInitWeight(int field_id,
                                                    const std::string &index);
  std::shared_ptr<ftrl_model_unit>& getOrInitBias();
  double predict(const std::vector<std::tuple<int, std::string, double>> &x,
                 bool sigmoid = true);
  double logit(const std::vector<std::tuple<int, std::string, double>> &x,
               bool update_model);
  void outputModel(std::ofstream &ofs);
  void debugPrintModel();
  bool loadModel(std::ifstream &ifs);

private:
  std::mutex mtxWeight;
  std::mutex mtxBias;
};

ftrl_model::ftrl_model(int _n_factors, int _n_fields) {
  init_mean = 0.;
  init_stddev = 0.;
  n_factors = _n_factors;
  n_fields = _n_fields;
  sum_vx.resize(n_factors);
  for (int i = 0; i < n_factors; i++) {
    sum_vx[i] = 0.0;
  }
}

ftrl_model::ftrl_model(double _mean, double _stddev, int _n_factors, int _n_fields) {
  init_mean = _mean;
  init_stddev = _stddev;
  n_factors = _n_factors;
  n_fields = _n_fields;
  sum_vx.resize(n_factors);
  for (int i = 0; i < n_factors; i++) {
    sum_vx[i] = 0.0;
  }
}

std::shared_ptr<ftrl_model_unit>& ftrl_model::getOrInitWeight(int field_id,
                                                              const std::string &index) {
  auto iter = mWeight.find(index);
  if (iter == mWeight.end()) {
    std::lock_guard<std::mutex> lock(mtxWeight);
    mWeight[index] = std::make_shared<ftrl_model_unit>(
        init_mean, init_stddev, n_factors, n_fields, field_id);
  }
  return mWeight[index];
}

std::shared_ptr<ftrl_model_unit>& ftrl_model::getOrInitBias() {
  if (mBias == nullptr) {
    std::lock_guard<std::mutex> lock(mtxBias);
    mBias = std::make_shared<ftrl_model_unit>();
  }
  return mBias;
}

double ftrl_model::predict(const std::vector<std::tuple<int, std::string, double>> &x,
                           bool sigmoid) {
  double result = logit(x, false);
  return sigmoid ? (1.0 / (1.0 + exp(-result))) : result;
}

double ftrl_model::logit(const std::vector<std::tuple<int, std::string, double>> &x,
                         bool update_model) {
  double result = 0.0;
  result += mBias->wi;
  for (const auto &feat : x) {
    auto iter = mWeight.find(std::get<1>(feat));
    if (iter != mWeight.end())
      result += (iter->second->wi * std::get<2>(feat));
  }

  if (update_model) {   // use sum_vx for later update model
    double sum_sqr, vx;
    for (int f = 0; f < n_factors; f++) {
      sum_vx[f] = 0.0;
      sum_sqr = 0.0;
      for (const auto &feat : x) {
        int fi = std::get<0>(feat);
        auto iter = mWeight.find(std::get<1>(feat));
        if (iter != mWeight.end()) {
          vx = iter->second->vi[fi * n_factors + f] * std::get<2>(feat);
          sum_vx[f] += vx;
          sum_sqr += vx * vx;
        }
      }
      result += 0.5 * (sum_vx[f] * sum_vx[f] - sum_sqr);
    }
  } else {
    double s_vx, sum_sqr, vx;
    for (int f = 0; f < n_factors; f++) {
      s_vx = sum_sqr = 0.0;
      for (const auto &feat : x) {
        int fi = std::get<0>(feat);
        auto iter = mWeight.find(std::get<1>(feat));
        if (iter != mWeight.end()) {
          vx = iter->second->vi[fi * n_factors + f] * std::get<2>(feat);
          s_vx += vx;
          sum_sqr += vx * vx;
        }
      }
      result += 0.5 * (s_vx * s_vx - sum_sqr);
    }
  }

  size_t xLen = x.size();
  for (int i = 0; i < xLen; i++) {
    for (int j = i + 1; j < xLen; j++) {
      int fi1 = std::get<0>(x[i]);
      int fi2 = std::get<0>(x[j]);
      const std::string &index1 = std::get<1>(x[i]);
      const std::string &index2 = std::get<1>(x[j]);
      double x1 = std::get<2>(x[i]);
      double x2 = std::get<2>(x[j]);
      double dot = 0.0;
      auto iter1 = mWeight.find(index1);
      auto iter2 = mWeight.find(index2);
      double val1, val2;
      if (iter1 != mWeight.end() && iter2 != mWeight.end()) {
        for (int f = 0; f < n_factors; f++) {
          val1 = iter1->second->vi[fi2 * n_factors + f];
          val2 = iter2->second->vi[fi1 * n_factors + f];
          dot += val1 * val2;
        }
        result += dot * x1 * x2;
      }
    }
  }
  return result;
}

void ftrl_model::outputModel(std::ofstream &ofs) {
  std::ostringstream ost;
  ost << "bias " << *mBias << std::endl;
  for (auto &elem : mWeight) {
    ost << elem.first << " " << *(elem.second) << std::endl;
  }
  ofs << ost.str();
}

void ftrl_model::debugPrintModel()
{
  std::cout << "bias " << *mBias << std::endl;
  for (auto iter = mWeight.begin(); iter != mWeight.end(); iter++)
    std::cout << iter->first << " " << *(iter->second) << std::endl;
}

bool ftrl_model::loadModel(std::ifstream &ifs) {
  std::string line;
  if (!getline(ifs, line))  // first get bias
    return false;
  std::vector<std::string> vec;
  utils::splitString(line, " ", vec);
  mBias = std::make_shared<ftrl_model_unit>(vec[1]);
  while (getline(ifs, line)) {
    vec.clear();
    utils::splitString(line, " ", vec);
    const std::string &index = vec[0];
    mWeight[index] = std::make_shared<ftrl_model_unit>(vec[1]);
  }
  return true;
}

}

#endif //FTRL_FFM_FTRL_MODEL_H