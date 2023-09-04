#ifndef FTRL_FFM_READER_H
#define FTRL_FFM_READER_H

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "data/parser.h"
#include "data/sample.h"
#include "utils/types.h"

namespace ftrl {

class Reader {
 public:
  explicit Reader(const std::string &file_type);
  void load_from_file(std::string_view file_name, int n_threads);
  [[nodiscard]] size_t get_size() const { return data_size; }

  size_t data_size{0};
  std::vector<Sample> data;
  std::shared_ptr<Parser> parser;

 private:
  std::vector<int64> get_data_partition(std::string_view file_name, int n_threads);
};

}  // namespace ftrl

#endif  // FTRL_FFM_READER_H
