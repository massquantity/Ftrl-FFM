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
  int n_factors, n_fields;
  float init_mean, init_stddev;
  std::vector<float> sum_vx;
  std::shared_ptr<ftrl_model_unit> model_bias;
  std::unordered_map<int, std::shared_ptr<ftrl_model_unit>> model_weight;

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
  std::mutex weight_mutex;
  std::mutex bias_mutex;
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
  auto iter = model_weight.find(index);
  if (iter == model_weight.end()) {
    std::lock_guard<std::mutex> lock(weight_mutex);
    if (model_type == "LR") {
      model_weight.insert(std::make_pair(index, std::make_shared<ftrl_model_unit>(
          init_mean, init_stddev)));
    } else if (model_type == "FM") {
      model_weight.insert(std::make_pair(index, std::make_shared<ftrl_model_unit>(
          init_mean, init_stddev, n_factors)));
    } else if (model_type == "FFM") {
      model_weight.insert(std::make_pair(index, std::make_shared<ftrl_model_unit>(
          init_mean, init_stddev, n_factors, n_fields)));
    }
  }
  return model_weight[index];
}

std::shared_ptr<ftrl_model_unit>& ftrl_model::getOrInitBias() {
  if (model_bias == nullptr) {
    std::lock_guard<std::mutex> lock(bias_mutex);
    model_bias = std::make_shared<ftrl_model_unit>();
  }
  return model_bias;
}

float ftrl_model::predict(
    const std::vector<std::tuple<int, int, float>> &x,
    bool sigmoid) {
  float result = logit(x, false);
  return sigmoid ? (1.0f / (1.0f + std::exp(-result))) : result;
}

float ftrl_model::logit(
    const std::vector<std::tuple<int, int, float>> &x,
    bool update_model) {
  double result = 0.0;
  result += model_bias->wi;
  for (const auto &feat : x) {
    auto iter = model_weight.find(std::get<1>(feat));
    if (iter != model_weight.end())
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
          auto iter = model_weight.find(std::get<1>(feat));
          if (iter != model_weight.end()) {
            vx = iter->second->vi[f] * std::get<2>(feat);
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
          auto iter = model_weight.find(std::get<1>(feat));
          if (iter != model_weight.end()) {
            vx = iter->second->vi[f] * std::get<2>(feat);
            s_vx += vx;
            sum_sqr += vx * vx;
          }
        }
        result += 0.5 * (s_vx * s_vx - sum_sqr);
      }
    }
  }

  if (model_type == "FFM") {
    size_t x_len = x.size();
    for (int i = 0; i < x_len; i++) {
      for (int j = i + 1; j < x_len; j++) {
        int fi1 = std::get<0>(x[i]);
        int fi2 = std::get<0>(x[j]);
        int index1 = std::get<1>(x[i]);
        int index2 = std::get<1>(x[j]);
        float x1 = std::get<2>(x[i]);
        float x2 = std::get<2>(x[j]);
        float dot = 0.0;
        auto iter1 = model_weight.find(index1);
        auto iter2 = model_weight.find(index2);
        float val1, val2;
        if (iter1 != model_weight.end() && iter2 != model_weight.end()) {
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
  size_t x_len = x.size();
  std::vector<std::shared_ptr<ftrl_model_unit>> params(x_len + 1);
  for (int i = 0; i < x_len; i++) {
    int index = std::get<1>(x[i]);
    params[i] = getOrInitWeight(index);
  }
  params[x_len] = getOrInitBias();

  update_linear_weight(params, x_len, w_alpha, w_beta, w_l1, w_l2);
  if (model_type == "FM") {
    update_fm_weight(params, x, n_factors, x_len, w_alpha, w_beta, w_l1, w_l2);
  }

  if (model_type == "FFM") {
    update_ffm_weight(params, x, n_factors, x_len, w_alpha, w_beta, w_l1, w_l2);
  }

  float p = logit(x, true);
  float mult = y * (1 / (1 + std::exp(-p * y)) - 1);
  update_linear_nz(params, x, x_len, w_alpha, mult);
  if (model_type == "FM") {
    update_fm_nz(params, x, x_len, w_alpha, mult, n_factors, sum_vx);
  }
  if (model_type == "FFM") {
    update_ffm_nz(params, x, x_len, w_alpha, mult, n_factors);
  }
  return p;
}

void ftrl_model::outputModel(std::ofstream &ofs) {
  std::ostringstream ost;
  ost << "bias " << *model_bias << std::endl;
  for (auto &elem : model_weight) {
    ost << elem.first << " " << *(elem.second) << std::endl;
  }
  ofs << ost.str();
}

void ftrl_model::debugPrintModel()
{
  std::cout << "bias " << *model_bias << std::endl;
  for (auto iter = model_weight.begin(); iter != model_weight.end(); iter++)
    std::cout << iter->first << " " << *(iter->second) << std::endl;
}

bool ftrl_model::loadModel(std::ifstream &ifs) {
  /*
  std::string line;
  if (!getline(ifs, line))  // first get bias
    return false;
  std::vector<std::string> vec;
  utils::splitString(line, " ", vec);
  model_bias = std::make_shared<ftrl_model_unit>(vec[1]);
  while (getline(ifs, line)) {
    vec.clear();
    utils::splitString(line, " ", vec);
    int index = stoi(vec[0]);
    model_weight[index] = std::make_shared<ftrl_model_unit>(vec[1]);
  } */
  return true;
}

}

#endif //FTRL_FFM_FTRL_MODEL_H
#pragma clang diagnostic pop