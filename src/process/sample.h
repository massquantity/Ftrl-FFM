#ifndef FTRL_FFM_SAMPLE_H
#define FTRL_FFM_SAMPLE_H

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

const std::string splitter = " ";
const std::string innerSplitter = ":";

class sample {
public:
  int y;
  std::vector<std::tuple<int, std::string, double>> x;

  sample(const std::string &line) {
    this->x.clear();
    std::string::size_type begin = line.find_first_not_of(splitter, 0);
    std::string::size_type end = line.find_first_of(splitter, begin);
    int label = atoi(line.substr(begin, end - begin).c_str());
    y = label > 0 ? 1 : -1;

    std::string::size_type lineLength = line.size();
    while (end < lineLength) {
      begin = line.find_first_not_of(splitter, end);
      if (begin == std::string::npos) break;
      end = line.find_first_of(innerSplitter, begin);
      if (end == std::string::npos) {
        std::cout << "wrong input: " << line << std::endl;
        throw std::out_of_range(line);
      }
      int field = stoi(line.substr(begin, end - begin));

      begin = end + 1;
      if (begin >= lineLength) {
        std::cout << "wrong input: " << line << std::endl;
        throw std::out_of_range(line);
      }
      end = line.find_first_of(innerSplitter, begin);
      if (end == std::string::npos) {
        std::cout << "wrong input: " << line << std::endl;
        throw std::out_of_range(line);
      }
      std::string feat = line.substr(begin, end - begin);

      begin = end + 1;
      if (begin >= lineLength) {
        std::cout << "wrong input: " << line << std::endl;
        throw std::out_of_range(line);
      }
      end = line.find_first_of(splitter, begin);
      double value = stod(line.substr(begin, end - begin));
      if (value != 0.) {
        x.emplace_back(std::make_tuple(field, feat, value));
      }
    }
  }
};

#endif //FTRL_FFM_SAMPLE_H