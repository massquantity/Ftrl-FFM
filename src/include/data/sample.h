#ifndef FTRL_FFM_SAMPLE_H
#define FTRL_FFM_SAMPLE_H

#include "utils/types.h"

struct Sample { // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
  feat_vec x;
  int y;
} __attribute__((aligned(32)));

#endif //FTRL_FFM_SAMPLE_H
