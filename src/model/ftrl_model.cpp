#include "model/ftrl_model.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "train/ftrl_trainer.h"

namespace ftrl {

FtrlModel::FtrlModel(float _mean, float _stddev, std::string_view _model_type)
    : init_mean(_mean), init_stddev(_stddev) {
  if (_model_type == "LR") {
    model_type = ModelType::LR;
  } else if (_model_type == "FM") {
    model_type = ModelType::FM;
  } else if (_model_type == "FFM") {
    model_type = ModelType::FFM;
  }
}

FtrlModel::FtrlModel(float _mean, float _stddev, int _n_factors, std::string_view _model_type)
    : FtrlModel(_mean, _stddev, _model_type) {
  n_factors = _n_factors;  // NOLINT
  sum_vx.resize(n_factors);
  for (int i = 0; i < n_factors; i++) {
    sum_vx[i] = 0.0;
  }
}

FtrlModel::FtrlModel(float _mean, float _stddev, int _n_factors, int _n_fields,
                     std::string_view _model_type)
    : FtrlModel(_mean, _stddev, _n_factors, _model_type) {
  n_fields = _n_fields;  // NOLINT
}

std::shared_ptr<ftrl_model_unit> &FtrlModel::get_or_init_weight(int index) {
  if (auto iter = model_weight.find(index); iter == model_weight.end()) {
    std::scoped_lock<std::mutex> lock(weight_mutex);  // NOLINT
    switch (model_type) {
      case ModelType::LR:
        model_weight.insert(
            std::make_pair(index, std::make_shared<ftrl_model_unit>(init_mean, init_stddev)));
        break;
      case ModelType::FM:
        model_weight.insert(std::make_pair(
            index, std::make_shared<ftrl_model_unit>(init_mean, init_stddev, n_factors)));
        break;
      case ModelType::FFM:
        model_weight.insert(std::make_pair(
            index, std::make_shared<ftrl_model_unit>(init_mean, init_stddev, n_factors, n_fields)));
        break;
    }
  }
  return model_weight[index];
}

std::shared_ptr<ftrl_model_unit> &FtrlModel::get_or_init_bias() {
  if (model_bias == nullptr) {
    std::scoped_lock<std::mutex> lock(bias_mutex);  // NOLINT
    model_bias = std::make_shared<ftrl_model_unit>();
  }
  return model_bias;
}

float FtrlModel::predict(const feat_vec &feats, bool sigmoid) {
  const float result = compute_logit(feats, false);
  return sigmoid ? (1.0f / (1.0f + std::exp(-result))) : result;
}

float FtrlModel::compute_logit(const feat_vec &feats, bool update_model) {
  double result = model_bias->wi;
  for (const auto &feat : feats) {
    auto iter = model_weight.find(std::get<1>(feat));
    if (iter != model_weight.end()) {
      result += (iter->second->wi * std::get<2>(feat));
    }
  }

  if (model_type == ModelType::FM) {
    // store sum_vx for later update model
    if (update_model) {
      float sum_sqr, vx;  // NOLINT
      for (int f = 0; f < n_factors; f++) {
        sum_vx[f] = 0.0;
        sum_sqr = 0.0;
        for (const auto &feat : feats) {
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
      float s_vx, sum_sqr, vx;  // NOLINT
      for (int f = 0; f < n_factors; f++) {
        s_vx = sum_sqr = 0.0;
        for (const auto &feat : feats) {
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

  if (model_type == ModelType::FFM) {
    const size_t x_len = feats.size();
    for (int i = 0; i < x_len; i++) {
      for (int j = i + 1; j < x_len; j++) {
        const int fi1 = std::get<0>(feats[i]);
        const int fi2 = std::get<0>(feats[j]);
        const int index1 = std::get<1>(feats[i]);
        const int index2 = std::get<1>(feats[j]);
        const float x1 = std::get<2>(feats[i]);
        const float x2 = std::get<2>(feats[j]);
        float dot = 0.0;
        auto iter1 = model_weight.find(index1);
        auto iter2 = model_weight.find(index2);
        float val1, val2;  // NOLINT
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
  return static_cast<float>(result);
}

float FtrlModel::train(const feat_vec &feats, int label, float w_alpha, float w_beta, float w_l1,
                       float w_l2) {
  const size_t feat_len = feats.size();
  std::vector<std::shared_ptr<ftrl_model_unit>> params(feat_len + 1);
  for (int i = 0; i < feat_len; i++) {
    const int index = std::get<1>(feats[i]);
    params[i] = get_or_init_weight(index);
  }
  params[feat_len] = get_or_init_bias();

  update_linear_weight(params, feat_len, w_alpha, w_beta, w_l1, w_l2);
  if (model_type == ModelType::FM) {
    update_fm_weight(params, feats, n_factors, feat_len, w_alpha, w_beta, w_l1, w_l2);
  }

  if (model_type == ModelType::FFM) {
    update_ffm_weight(params, feats, n_factors, feat_len, w_alpha, w_beta, w_l1, w_l2);
  }

  const float p = compute_logit(feats, true);
  const float mult = label * (1 / (1 + std::exp(-p * label)) - 1);  // NOLINT
  update_linear_nz(params, feats, feat_len, w_alpha, mult);
  if (model_type == ModelType::FM) {
    update_fm_nz(params, feats, feat_len, w_alpha, mult, n_factors, sum_vx);
  }
  if (model_type == ModelType::FFM) {
    update_ffm_nz(params, feats, feat_len, w_alpha, mult, n_factors);
  }
  return p;
}

void FtrlModel::output_model(std::ofstream &ofs) {
  std::ostringstream ost;
  ost << "bias " << *model_bias << std::endl;
  for (auto &elem : model_weight) {
    ost << elem.first << " " << *(elem.second) << std::endl;
  }
  ofs << ost.str();
}

[[maybe_unused]] void FtrlModel::debug_print_model() {
  std::cout << "bias " << *model_bias << std::endl;
  for (auto &iter : model_weight) {
    std::cout << iter.first << " " << *(iter.second) << std::endl;
  }
}

bool FtrlModel::load_model(std::ifstream &ifs) {
  /*
  std::string line;
  if (!getline(ifs, line))  // first get bias
    return false;
  std::vector<std::string> vec;
  utils::splitString(line, " ", vec);
  model_bias = std::make_shared<ftrl_model_unit>(vec[1]);
  while (getline(ifs, line)) {
    vec.clear();
    utils::split_string(line, " ", vec);
    int index = stoi(vec[0]);
    model_weight[index] = std::make_shared<ftrl_model_unit>(vec[1]);
  } */
  return true;
}

}  // namespace ftrl
