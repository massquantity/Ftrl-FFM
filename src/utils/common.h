#ifndef FTRL_FFM_COMMON_H
#define FTRL_FFM_COMMON_H

typedef int8_t  int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;

typedef uint8_t  uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

typedef std::vector<std::tuple<int, int, float>> feat_vec;

static const std::string splitter = " ";
static const std::string innerSplitter = ":";

#endif //FTRL_FFM_COMMON_H