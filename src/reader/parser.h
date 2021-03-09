#ifndef FTRL_FFM_PARSER_H
#define FTRL_FFM_PARSER_H

#include <iostream>
#include <cstring>
#include <regex>
#include "../utils/common.h"
using std::string;

namespace ftrl {

struct Sample {
  feat_vec x;
  int y;
};

class Parser {
public:
  Parser() = default;
  virtual ~Parser() = default;
  virtual void parse(const string &line, Sample &sample) = 0;
};

class LibsvmParser : public Parser {
  void parse(const string &line, Sample &sample) override {
    sample.x.clear();
    string::size_type begin = line.find_first_not_of(splitter, 0);
    string::size_type end = line.find_first_of(splitter, begin);
    int label = atoi(line.substr(begin, end - begin).c_str());
    sample.y = label > 0 ? 1 : -1;

    string::size_type lineLength = line.size();
    while (end < lineLength) {
      int field = 0;  // dummy field
      begin = line.find_first_not_of(splitter, end);
      if (begin == string::npos) break;
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

class FFMParser : public Parser {
public:
  void parseFFM(const string &line, Sample &sample) {
    sample.x.clear();
    string::size_type begin = line.find_first_not_of(splitter, 0);
    string::size_type end = line.find_first_of(splitter, begin);
    int label = stoi(line.substr(begin, end - begin));
    sample.y = label > 0 ? 1 : -1;

    std::sregex_iterator it {line.begin(), line.end(), reg};
    std::sregex_iterator end_it;
    for (; it != end_it; ++it) {
      int field = stoi(it->str(1));
      int feat = stoi(it->str(2));
      float value = stof(it->str(3));
      if (value != 0.0) {
        sample.x.emplace_back(std::make_tuple(field, feat, value));
      }
    }
  }

  void parse(const string &line, Sample &sample) override {
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

  void parseCFFM(const string &line, Sample &sample) {
    sample.x.clear();
    string::size_type begin = line.find_first_not_of(splitter, 0);
    string::size_type end = line.find_first_of(splitter, begin);
    int label = atoi(line.substr(begin, end - begin).c_str());
    sample.y = label > 0 ? 1 : -1;

    char *buffer = new char[line.size() + 1];
    std::strcpy(buffer, line.c_str());
    buffer[line.size()] = '\0';
    strtok(buffer, " ");
    while (true) {
      char *field_char = strtok(nullptr, ":");
      char *feat_char = strtok(nullptr, ":");
      char *value_char = strtok(nullptr, " ");
      if (field_char == nullptr) break;
      int field = atoi(field_char);
      int feat = atoi(feat_char);
      float value = atof(value_char);
      if (value != 0.0) {
        sample.x.emplace_back(std::make_tuple(field, feat, value));
      }
    }
    delete [] buffer;
  }

private:
  std::regex reg {"([[:digit:]]+):([[:digit:]]+):([[:digit:]]+\\.?[[:digit:]]*)"};
};

}

#endif //FTRL_FFM_PARSER_H