#ifndef FTRL_FFM_LOSS_H
#define FTRL_FFM_LOSS_H

#include <cmath>

inline double loss(int y, double pred) {
  return std::log1p(std::exp(-y * pred));
}

#endif //FTRL_FFM_LOSS_H