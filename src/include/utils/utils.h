#ifndef FTRL_FFM_UTILS_H
#define FTRL_FFM_UTILS_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

#include "utils/types.h"

struct utils {
  template <class T>
  static inline T sgn(T x) {
    return x > 0 ? 1 : -1;
  }

  template <class T>
  static inline T sigmoid(T x) {
    return 1 / (1 + std::exp(-x));
  }

  template <class T>
  [[maybe_unused]] static T uniform() {
    return rand() / (RAND_MAX + 1.0);  // NOLINT
  }

  template <class T>
  static T gaussian(T mean, T stddev) {
    std::random_device rd;  // NOLINT
    std::mt19937 gen(rd());
    std::normal_distribution<> dist{mean, stddev};
    return dist(gen);
  }

  template <class T>
  static std::vector<T> init_weights(std::size_t num, T mean, T stddev) {
    std::vector<T> weights(num);
    std::generate(weights.begin(), weights.end(), [&] { return gaussian(mean, stddev); });
    return weights;
  }

  template <class T>
  static decltype(auto) init_weights(std::size_t num, int n_factors, T mean, T stddev) {
    std::vector<std::vector<T>> weights(num);
    for (int i = 0; i < num; i++) {
      std::vector<T> v(n_factors);
      std::generate(v.begin(), v.end(), [&] { return gaussian(mean, stddev); });
      weights[i] = std::move(v);
    }
    return weights;
  }

  template <class T>
  static decltype(auto) init_weights(std::size_t num, int n_fields, int n_factors, T mean,
                                     T stddev) {
    const int v_size = n_factors * n_fields;
    return init_weights(num, v_size, mean, stddev);
  }

  using clock_time = std::chrono::time_point<std::chrono::steady_clock>;
  static constexpr int64 numerator = std::chrono::nanoseconds::period::num;
  static constexpr int64 denominator = std::chrono::nanoseconds::period::den;

  static inline constexpr auto convert_time = [](auto &&time) {
    auto exact_time = static_cast<double>(std::forward<decltype(time)>(time));
    return exact_time * numerator / denominator;
  };

  static decltype(auto) compute_time(const clock_time &start_time) {
    // std::this_thread::sleep_for(std::chrono::microseconds(1020023));
    auto end_time = std::chrono::steady_clock::now();
    auto nano_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    return convert_time(nano_time);
  }
};

#endif  // FTRL_FFM_UTILS_H
