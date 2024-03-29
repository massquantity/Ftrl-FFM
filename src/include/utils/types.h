#ifndef FTRL_FFM_TYPES_H
#define FTRL_FFM_TYPES_H

#include <cstdint>
#include <tuple>
#include <vector>

typedef int8_t  int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;

typedef uint8_t  uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

typedef std::tuple<int, int, float> feat;
typedef std::vector<feat> feat_vec;

enum class ModelType : uint8 {
  LR = 0,
  FM = 1,
  FFM = 2
};

#endif  // FTRL_FFM_TYPES_H
