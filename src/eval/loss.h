#ifndef FTRL_FFM_LOSS_H
#define FTRL_FFM_LOSS_H

#include <cmath>

double loss(int y, double pred) {
  return log1p(exp(-y * pred));
}

#endif //FTRL_FFM_LOSS_H