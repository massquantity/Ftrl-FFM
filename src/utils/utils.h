#ifndef FTRL_FFM_UTILS_H
#define FTRL_FFM_UTILS_H

#include <string>
#include <vector>
#include <random>

const double kPrecision = 1e-11;

struct utils {
  static void splitString(const std::string &line,
                          const std::string &delimiter,
                          std::vector<std::string> &v) {
    std::string::size_type begin = line.find_first_not_of(delimiter, 0);
    std::string::size_type end = line.find_first_of(delimiter, begin);
    while (begin != std::string::npos || end != std::string::npos) {
      v.push_back(line.substr(begin, end - begin));
      begin = line.find_first_not_of(delimiter, end);
      end = line.find_first_of(delimiter, begin);
    }
  }

  static int sgn(double x) {
    return x > kPrecision ? 1 : -1;
  }

  static double uniform() {
    return rand() / ((double)RAND_MAX + 1.);
  }

  static double gaussian(double mean = 0., double stddev = 0.01) {
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