#ifndef FTRL_FFM_PARSER_H
#define FTRL_FFM_PARSER_H

#include <iostream>
#include <cstring>
#include <regex>

#include "../utils/common.h"

namespace ftrl {

struct Sample {
  std::vector<feat> x;
  int y;
} __attribute__((aligned(32)));

class Parser {
public:
  Parser() = default;
  virtual ~Parser() = default;
  virtual void parse(const std::string &line, Sample &sample) = 0;
};

class LibsvmParser : public Parser {
  void parse(const std::string &line, Sample &sample) override {
    sample.x.clear();
    std::string::size_type begin = line.find_first_not_of(splitter, 0);
    std::string::size_type end = line.find_first_of(splitter, begin);
    const int label = std::stoi(line.substr(begin, end - begin));
    sample.y = label > 0 ? 1 : -1;

    const std::string::size_type lineLength = line.size();
    while (end < lineLength) {  // NOLINT
      const int field = 0;  // dummy field
      begin = line.find_first_not_of(splitter, end);
      if (begin == std::string::npos) break;
      end = line.find_first_of(innerSplitter, begin);
      if (end == std::string::npos) {
        std::cout << "wrong input: " << line << std::endl;
        throw std::out_of_range(line);
      }
      const int feat = stoi(line.substr(begin, end - begin));

      begin = end + 1;
      if (begin >= lineLength) {
        std::cout << "wrong input: " << line << std::endl;
        throw std::out_of_range(line);
      }
      end = line.find_first_of(splitter, begin);
      const float value = stof(line.substr(begin, end - begin));
      if (value != 0.0) {
        sample.x.emplace_back(field, feat, value);
      }
    }
  }
};

class FFMParser : public Parser {
public:
  [[maybe_unused]] void parseFFM(const std::string &line, Sample &sample) {
    sample.x.clear();
    const std::string::size_type begin = line.find_first_not_of(splitter, 0);
    const std::string::size_type end = line.find_first_of(splitter, begin);
    const int label = std::stoi(line.substr(begin, end - begin));
    sample.y = label > 0 ? 1 : -1;

    std::sregex_iterator it {line.begin(), line.end(), reg};
    const std::sregex_iterator end_it;
    for (; it != end_it; ++it) {
      const int field = stoi(it->str(1));
      const int feat = stoi(it->str(2));
      const float value = stof(it->str(3));
      if (value != 0.0) {
        sample.x.emplace_back(field, feat, value);
      }
    }
  }

  void parse(const std::string &line, Sample &sample) override {
    sample.x.clear();
    std::string::size_type begin = line.find_first_not_of(splitter, 0);
    std::string::size_type end = line.find_first_of(splitter, begin);
    const int label = std::stoi(line.substr(begin, end - begin));
    sample.y = label > 0 ? 1 : -1;

    const std::string::size_type lineLength = line.size();
    while (end < lineLength) {  // NOLINT
      begin = line.find_first_not_of(splitter, end);
      if (begin == std::string::npos) break;
      end = line.find_first_of(innerSplitter, begin);
      if (end == std::string::npos) {
        std::cout << "wrong input: " << line << std::endl;
        throw std::out_of_range(line);
      }
      const int field = std::stoi(line.substr(begin, end - begin));

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
      const int feat = std::stoi(line.substr(begin, end - begin));

      begin = end + 1;
      if (begin >= lineLength) {
        std::cout << "wrong input: " << line << std::endl;
        throw std::out_of_range(line);
      }
      end = line.find_first_of(splitter, begin);
      const float value = std::stof(line.substr(begin, end - begin));
      if (value != 0.0) {
        sample.x.emplace_back(field, feat, value);
      }
    }
  }

  [[maybe_unused]] void parseCFFM(const std::string &line, Sample &sample) {
    sample.x.clear();
    const std::string::size_type begin = line.find_first_not_of(splitter, 0);
    const std::string::size_type end = line.find_first_of(splitter, begin);
    const int label = atoi(line.substr(begin, end - begin).c_str());  // NOLINT
    sample.y = label > 0 ? 1 : -1;

    // NOLINTBEGIN
    char *buffer = new char[line.size() + 1];
    std::strcpy(buffer, line.c_str());
    buffer[line.size()] = '\0';
    strtok(buffer, " ");
    while (true) {
      char *field_char = strtok(nullptr, ":");
      char *feat_char = strtok(nullptr, ":");
      char *value_char = strtok(nullptr, " ");
      if (field_char == nullptr) break;
      const int field = atoi(field_char);
      const int feat = atoi(feat_char);
      const float value = atof(value_char);
      if (value != 0.0) {
        sample.x.emplace_back(std::make_tuple(field, feat, value));
      }
    }
    // NOLINTEND
    delete [] buffer;
  }

private:
  std::regex reg {"([[:digit:]]+):([[:digit:]]+):([[:digit:]]+\\.?[[:digit:]]*)"};
};

}

#endif //FTRL_FFM_PARSER_H