#ifndef FTRL_FFM_LOSS_H
#define FTRL_FFM_LOSS_H

#include <cmath>

#include "utils/utils.h"

inline double loss(int y, double logit) {
  // return std::log1p(std::exp(-y * pred));
  const auto s = utils::sigmoid<double>(logit);
  return -y * std::log(s) - (1 - y) * std::log(1 - s);
}

#endif  // FTRL_FFM_LOSS_H
