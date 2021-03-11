#ifndef FTRL_FFM_UTILS_H
#define FTRL_FFM_UTILS_H

#include <string>
#include <vector>
#include <random>

struct utils {

  static int sgn(float x) {
    return x > 0 ? 1 : -1;
  }

  static float uniform() {
    return rand() / ((float)RAND_MAX + 1.);
  }

  static float gaussian(float mean = 0., float stddev = 0.01) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist{mean, stddev};
    return dist(gen);
  }
};

template<typename T, typename... U>
std::unique_ptr<T> make_unique(U&&... params) {
  return std::unique_ptr<T>(new T(std::forward<U>(params)...));
}

#endif //FTRL_FFM_UTILS_H