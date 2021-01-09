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
#include "ftrl_unit.h"
#include "../learn/ftrl_learn.h"
#include "../utils/utils.h"

namespace ftrl {

class ftrl_model {
public:
  std::string model_type = "FFM";
  int n_factors = 0, n_fields = 0;
  float init_mean, init_stddev;
  std::vector<float> sum_vx;
  std::shared_ptr<ftrl_model_unit> mBias;
  std::unordered_map<int, std::shared_ptr<ftrl_model_unit>> mWeight;

  ftrl_model(float _mean, float _stddev, std::string _model_type);
  ftrl_model(float _mean, float _stddev, int _n_factors,
             std::string _model_type);
  ftrl_model(float _mean, float _stddev, int _n_factors, int _n_fields,
             std::string _model_type);
  std::shared_ptr<ftrl_model_unit>& getOrInitWeight(int index);
  std::shared_ptr<ftrl_model_unit>& getOrInitBias();

  float predict(const std::vector<std::tuple<int, int, float>> &x,
                bool sigmoid = true);

  float logit(const std::vector<std::tuple<int, int, float>> &x,
              bool update_model);

  float train(const std::vector<std::tuple<int, int, float>> &x, int y,
             float w_alpha, float w_beta, float w_l1, float w_l2);

  void outputModel(std::ofstream &ofs);
  void debugPrintModel();
  bool loadModel(std::ifstream &ifs);

private:
  std::mutex mtxWeight;
  std::mutex mtxBias;
};

ftrl_model::ftrl_model(float _mean, float _stddev, std::string _model_type)
    : init_mean(_mean), init_stddev(_stddev), model_type(_model_type) {
}

ftrl_model::ftrl_model(float _mean, float _stddev, int _n_factors,
                       std::string _model_type)
                       : ftrl_model(_mean, _stddev, _model_type) {
  n_factors = _n_factors;
  sum_vx.resize(n_factors);
  for (int i = 0; i < n_factors; i++) {
    sum_vx[i] = 0.0;
  }
}

ftrl_model::ftrl_model(float _mean, float _stddev, int _n_factors, int _n_fields,
                       std::string _model_type)
                       : ftrl_model(_mean, _stddev, _n_factors, _model_type) {
  n_fields = _n_fields;
}

std::shared_ptr<ftrl_model_unit>& ftrl_model::getOrInitWeight(int index) {
  auto iter = mWeight.find(index);
  if (iter == mWeight.end()) {
    std::lock_guard<std::mutex> lock(mtxWeight);
    if (model_type == "LR") {
      mWeight.insert(std::make_pair(index, std::make_shared<ftrl_model_unit>(
          init_mean, init_stddev)));
    } else if (model_type == "FM") {
      mWeight.insert(std::make_pair(index, std::make_shared<ftrl_model_unit>(
          init_mean, init_stddev, n_factors)));
    } else if (model_type == "FFM") {
      mWeight.insert(std::make_pair(index, std::make_shared<ftrl_model_unit>(
          init_mean, init_stddev, n_factors, n_fields)));
    }
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

float ftrl_model::predict(
    const std::vector<std::tuple<int, int, float>> &x,
    bool sigmoid) {
  float result = logit(x, false);
  return sigmoid ? (1.0 / (1.0 + exp(-result))) : result;
}

float ftrl_model::logit(
    const std::vector<std::tuple<int, int, float>> &x,
    bool update_model) {
  double result = 0.0;
  result += mBias->wi;
  for (const auto &feat : x) {
    auto iter = mWeight.find(std::get<1>(feat));
    if (iter != mWeight.end())
      result += (iter->second->wi * std::get<2>(feat));
  }

  if (model_type == "FM") {
    // store sum_vx for later update model
    if (update_model) {
      float sum_sqr, vx;
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
      float s_vx, sum_sqr, vx;
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
  }

  if (model_type == "FFM") {
    size_t xLen = x.size();
    for (int i = 0; i < xLen; i++) {
      for (int j = i + 1; j < xLen; j++) {
        int fi1 = std::get<0>(x[i]);
        int fi2 = std::get<0>(x[j]);
        int index1 = std::get<1>(x[i]);
        int index2 = std::get<1>(x[j]);
        float x1 = std::get<2>(x[i]);
        float x2 = std::get<2>(x[j]);
        float dot = 0.0;
        auto iter1 = mWeight.find(index1);
        auto iter2 = mWeight.find(index2);
        float val1, val2;
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
  }
  return result;
}

float ftrl_model::train(const std::vector<std::tuple<int, int, float>> &x, int y,
                       float w_alpha, float w_beta, float w_l1, float w_l2) {
  size_t xLen = x.size();
  std::vector<std::shared_ptr<ftrl_model_unit>> params(xLen + 1);
  for (int i = 0; i < xLen; i++) {
    int index = std::get<1>(x[i]);
    params[i] = getOrInitWeight(index);
  }
  params[xLen] = getOrInitBias();

  update_linear_weight(params, xLen, w_alpha, w_beta, w_l1, w_l2);
  if (model_type == "FM") {
    update_fm_weight(params, x, n_factors, xLen, w_alpha, w_beta, w_l1, w_l2);
  }

  if (model_type == "FFM") {
    update_ffm_weight(params, x, n_factors, xLen, w_alpha, w_beta, w_l1, w_l2);
  }

  float p = logit(x, true);
  float mult = y * (1 / (1 + exp(-p * y)) - 1);
  update_linear_nz(params, x, xLen, w_alpha, mult);
  if (model_type == "FM") {
    update_fm_nz(params, x, xLen, w_alpha, mult, n_factors, sum_vx);
  }
  if (model_type == "FFM") {
    update_ffm_nz(params, x, xLen, w_alpha, mult, n_factors);
  }
  return p;
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
    int index = stoi(vec[0]);
    mWeight[index] = std::make_shared<ftrl_model_unit>(vec[1]);
  }
  return true;
}

}

#endif //FTRL_FFM_FTRL_MODEL_H