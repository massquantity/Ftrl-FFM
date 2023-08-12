#ifndef FTRL_FFM_UTILS_H
#define FTRL_FFM_UTILS_H

#include <chrono>
#include <random>

using namespace std::chrono;

struct utils {

  static float sgn(float x) {
    return x > 0 ? 1 : -1;
  }

  static float uniform() {
    return rand() / ((float)RAND_MAX + 1.);  // NOLINT
  }

  static float gaussian(float mean = 0.0, float stddev = 0.02) {
    std::random_device rd;  // NOLINT
    std::mt19937 gen(rd());
    std::normal_distribution<> dist{mean, stddev};
    return static_cast<float>(dist(gen));
  }

  static decltype(auto) compute_time(time_point<steady_clock> start_time) {
    // auto start_time = std::chrono::steady_clock::now();
    // std::this_thread::sleep_for(std::chrono::microseconds(1020023));
    auto end_time = std::chrono::steady_clock::now();
    auto nano_time = duration_cast<nanoseconds>(end_time - start_time).count();
    return static_cast<double>(nano_time) * nanoseconds::period::num / nanoseconds::period::den;
  }
};

#endif //FTRL_FFM_UTILS_H