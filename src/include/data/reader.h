#ifndef FTRL_FFM_READER_H
#define FTRL_FFM_READER_H

#include <memory>
#include <string>
#include <vector>

#include "data/parser.h"
#include "data/sample.h"

namespace ftrl {

class Reader {
public:
  explicit Reader(const std::string &file_type);
  void load_from_file(const std::string &file_name, int n_threads);
  [[maybe_unused]] [[nodiscard]] size_t getSize() const { return dataSize; }
  size_t dataSize{0};
  std::vector<Sample> data;
  std::shared_ptr<Parser> parser;
};

}

#endif //FTRL_FFM_READER_H