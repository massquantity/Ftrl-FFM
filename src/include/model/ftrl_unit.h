#ifndef FTRL_FFM_FTRL_UNIT_H
#define FTRL_FFM_FTRL_UNIT_H

#include <iostream>
#include <shared_mutex>
#include <string>
#include <vector>

#include "utils/utils.h"

namespace ftrl {

struct ftrl_model_unit {
  float wi;
  float w_ni;
  float w_zi;
  std::vector<float> vi;
  std::vector<float> v_ni;
  std::vector<float> v_zi;
  std::shared_mutex mtx;

  // init bias
  ftrl_model_unit() : wi(0.0), w_ni(0.0), w_zi(0.0) {}

  // init linear
  ftrl_model_unit(float mean, float stddev)
      : wi(utils::gaussian(mean, stddev)), w_ni(0.0), w_zi(0.0) {}

  // init fm
  ftrl_model_unit(float mean, float stddev, int n_factors) : ftrl_model_unit(mean, stddev) {
    vi.resize(n_factors);
    v_ni.resize(n_factors);
    v_zi.resize(n_factors);
    for (int f = 0; f < n_factors; f++) {
      vi[f] = utils::gaussian(mean, stddev);
      v_ni[f] = 0.0;
      v_zi[f] = 0.0;
    }
  }

  // init ffm
  ftrl_model_unit(float mean, float stddev, int n_factors, int n_fields)
      : ftrl_model_unit(mean, stddev) {
    const int v_size = n_factors * n_fields;
    vi.resize(v_size);
    v_ni.resize(v_size);
    v_zi.resize(v_size);
    for (int f = 0; f < v_size; f++) {
      vi[f] = utils::gaussian(mean, stddev);
      v_ni[f] = 0.0;
      v_zi[f] = 0.0;
    }
  }

  // todo
  friend inline std::ostream &operator<<(std::ostream &os, const ftrl_model_unit &mu) {
    os << mu.wi;
    return os;
  }
};

}  // namespace ftrl

#endif  // FTRL_FFM_FTRL_UNIT_H
