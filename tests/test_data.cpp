#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "common.h"
#include "data/reader.h"

namespace ftrl {

TEST_CASE("file reader and parser") {
  write_test_data(test_file_path);
  auto reader = Reader("libffm");
  reader.load_from_file(test_file_path, 4);
  CHECK(reader.get_size() == 10);
  CHECK(reader.data[0].y == 0);
  CHECK(reader.data[0].x[0] == std::tuple{0, 1, 1});
  CHECK(reader.data[0].x[3] == std::tuple{3, 31, 1});
  remove_test_file(test_file_path);
}

}  // namespace ftrl
