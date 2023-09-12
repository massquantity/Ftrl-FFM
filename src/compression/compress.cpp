#include "compression/compress.h"

#include <zstd.h>

#include <cstdlib>

#include "utils/types.h"

extern "C" {
#include "compression/file_ops.h"
}

namespace ftrl {

void compress_weights(const float *weights, size_t weight_size, std::string_view file_name,
                      int compress_level) {
  const size_t buffer_size = ZSTD_compressBound(weight_size);
  void *const buffer = malloc_orDie(buffer_size);
  const size_t compress_size =
      ZSTD_compress(buffer, buffer_size, weights, weight_size, compress_level);
  CHECK_ZSTD(compress_size);  // NOLINT

  saveFile_orDie(file_name.data(), buffer, compress_size);
  printf("saving to %s, ", file_name.data());
  printf("before: %zu -> after: %zu\n", weight_size, compress_size);
  free(buffer);
}

float *decompress_weights(std::string_view file_name) {
  size_t file_size = 0;
  void *const file_buffer = mallocAndLoadFile_orDie(file_name.data(), &file_size);
  const uint64 buffer_size = ZSTD_getFrameContentSize(file_buffer, file_size);
  // NOLINTNEXTLINE
  CHECK(buffer_size != ZSTD_CONTENTSIZE_ERROR, "%s: not compressed by zstd!", file_name.data());
  // NOLINTNEXTLINE
  CHECK(buffer_size != ZSTD_CONTENTSIZE_UNKNOWN, "%s: original size unknown!", file_name.data());

  ZSTD_DCtx *dctx = ZSTD_createDCtx();
  void *const buffer = malloc_orDie(buffer_size);
  const size_t decompress_size =
      ZSTD_decompressDCtx(dctx, buffer, buffer_size, file_buffer, file_size);
  CHECK_ZSTD(decompress_size);                            // NOLINT
  CHECK(decompress_size == buffer_size);                  // NOLINT
  float *buffer_ptr = reinterpret_cast<float *>(buffer);  // NOLINT

  printf("loading from %s, ", file_name.data());
  printf("before: %zu -> after: %lu \n", file_size, buffer_size);
  ZSTD_freeDCtx(dctx);
  free(file_buffer);
  return buffer_ptr;
}

}  // namespace ftrl
