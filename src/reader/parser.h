#ifndef FTRL_FFM_PARSER_H
#define FTRL_FFM_PARSER_H

#include "../utils/common.h"
using std::string;

namespace ftrl {

struct Sample {
  feat_vec x;
  int y;
};

class Parser {
public:
  static void parse(const string &line, Sample &sample) {
    sample.x.clear();
    string::size_type begin = line.find_first_not_of(splitter, 0);
    string::size_type end = line.find_first_of(splitter, begin);
    int label = atoi(line.substr(begin, end - begin).c_str());
    sample.y = label > 0 ? 1 : -1;

    string::size_type lineLength = line.size();
    while (end < lineLength) {
      begin = line.find_first_not_of(splitter, end);
      if (begin == string::npos) break;
      end = line.find_first_of(innerSplitter, begin);
      if (end == string::npos) {
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
      if (end == string::npos) {
        std::cout << "wrong input: " << line << std::endl;
        throw std::out_of_range(line);
      }
      int feat = stoi(line.substr(begin, end - begin));

      begin = end + 1;
      if (begin >= lineLength) {
        std::cout << "wrong input: " << line << std::endl;
        throw std::out_of_range(line);
      }
      end = line.find_first_of(splitter, begin);
      float value = stof(line.substr(begin, end - begin));
      if (value != 0.0) {
        sample.x.emplace_back(std::make_tuple(field, feat, value));
      }
    }
  }
};

}

#endif //FTRL_FFM_PARSER_H