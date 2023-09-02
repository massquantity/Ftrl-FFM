#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <iostream>

#include "common.h"
#include "eval/loss.h"
#include "utils/utils.h"

namespace ftrl {

TEST_CASE("sign function") {
  CHECK(utils::sgn<int>(1) == 1);
  CHECK(utils::sgn<int>(0) == -1);
  CHECK(utils::sgn<int>(-2) == -1);
  CHECK(utils::sgn<double>(-2) == doctest::Approx(-1.0));
}

TEST_CASE("sigmoid") {
  CHECK(utils::sigmoid<float>(0) == doctest::Approx(0.5));
  CHECK(utils::sigmoid<float>(1) == doctest::Approx(0.7311).epsilon(1e-4));
  CHECK(utils::sigmoid<float>(-2) == doctest::Approx(0.1192).epsilon(1e-4));
}

TEST_CASE("init weights") {
  const std::size_t num = 100;
  const int n_fields = 10;
  const int n_factors = 4;
  const float mean = 0.0;
  const float stddev = 0.01;
  auto weights = utils::init_weights(num, n_fields, n_factors, mean, stddev);
  CHECK(weights.size() == num);
  CHECK(weights[0].size() == n_fields * n_factors);
  std::for_each(weights.cbegin(), weights.cend(), [](auto v) {
    CHECK_FALSE(std::none_of(v.cbegin(), v.cend(), [](float i) { return i > -0.05 && i < 0.05; }));
  });
}

TEST_CASE("cross entropy loss") {
  CHECK(loss(1, 2) == doctest::Approx(0.1269).epsilon(1e-4));
  CHECK(loss(0, 1) == doctest::Approx(1.3133).epsilon(1e-4));
}

TEST_CASE("test data samples reading and writing") {
  write_test_data(test_file_path);
  std::stringstream s;
  const std::ifstream ifs(test_file_path.data());
  s << ifs.rdbuf();
  CHECK(s.str() == test_samples);
  remove_test_file(test_file_path);
}

}  // namespace ftrl
