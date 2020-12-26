#ifndef FTRL_FFM_LOSS_H
#define FTRL_FFM_LOSS_H

double loss(int y, double pred) {
  return log(1 + exp(-y * pred));
}

#endif //FTRL_FFM_LOSS_H