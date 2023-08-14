#ifndef FTRL_FFM_UTILS_H
#define FTRL_FFM_UTILS_H

#include <chrono>
#include <random>
#include <utility>

#include "utils/types.h"

struct utils {
  static inline float sgn(float x) { return x > 0 ? 1 : -1; }

  [[maybe_unused]] static float uniform() {
    return rand() / ((float)RAND_MAX + 1.);  // NOLINT
  }

  static float gaussian(float mean = 0.0, float stddev = 0.02) {
    std::random_device rd;  // NOLINT
    std::mt19937 gen(rd());
    std::normal_distribution<> dist{mean, stddev};
    return static_cast<float>(dist(gen));
  }

  using clock_time = std::chrono::time_point<std::chrono::steady_clock>;
  static constexpr int64 numerator = std::chrono::nanoseconds::period::num;
  static constexpr int64 denominator = std::chrono::nanoseconds::period::den;

  static inline constexpr auto convert_time = [](auto &&time) {
    auto exact_time = static_cast<double>(std::forward<decltype(time)>(time));
    return exact_time * numerator / denominator;
  };

  static decltype(auto) compute_time(clock_time start_time) {
    // std::this_thread::sleep_for(std::chrono::microseconds(1020023));
    auto end_time = std::chrono::steady_clock::now();
    auto nano_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    return convert_time(nano_time);
  }
};

#endif  // FTRL_FFM_UTILS_H