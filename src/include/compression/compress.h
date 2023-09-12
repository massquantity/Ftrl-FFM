#ifndef FTRL_FFM_COMPRESS_H
#define FTRL_FFM_COMPRESS_H

#include <string_view>
#include <vector>

namespace ftrl {

void compress_weights(const float *weights, size_t weight_size, std::string_view file_name,
                      int compress_level);

float *decompress_weights(std::string_view file_name);

}  // namespace ftrl

#endif  // FTRL_FFM_COMPRESS_H
