#include "data/parser.h"

#include <cstring>
#include <iostream>

namespace ftrl {

static inline constexpr std::string_view SPLITTER = " ";
static inline constexpr std::string_view INNER_SPLITTER = ":";

void LibsvmParser::parse(const std::string &line, Sample &sample) {
  sample.x.clear();
  std::string::size_type begin = line.find_first_not_of(SPLITTER, 0);
  std::string::size_type end = line.find_first_of(SPLITTER, begin);
  const int label = std::stoi(line.substr(begin, end - begin));
  sample.y = label > 0 ? 1 : 0;

  const std::string::size_type line_length = line.size();
  while (end < line_length) {
    const int field = 0;  // dummy field
    begin = line.find_first_not_of(SPLITTER, end);
    if (begin == std::string::npos) break;
    end = line.find_first_of(INNER_SPLITTER, begin);
    if (end == std::string::npos) {
      std::cout << "wrong input: " << line << std::endl;
      throw std::out_of_range(line);
    }
    const int feat = stoi(line.substr(begin, end - begin));

    begin = end + 1;
    if (begin >= line_length) {
      std::cout << "wrong input: " << line << std::endl;
      throw std::out_of_range(line);
    }
    end = line.find_first_of(SPLITTER, begin);
    const float value = stof(line.substr(begin, end - begin));
    if (value != 0.0) {
      sample.x.emplace_back(field, feat, value);
    }
  }
}

[[maybe_unused]] void FFMParser::parseFFM(const std::string &line, Sample &sample) {
  sample.x.clear();
  const std::string::size_type begin = line.find_first_not_of(SPLITTER, 0);
  const std::string::size_type end = line.find_first_of(SPLITTER, begin);
  const int label = std::stoi(line.substr(begin, end - begin));
  sample.y = label > 0 ? 1 : 0;

  std::sregex_iterator it{line.begin(), line.end(), reg};
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

void FFMParser::parse(const std::string &line, Sample &sample) {
  sample.x.clear();
  std::string::size_type begin = line.find_first_not_of(SPLITTER, 0);
  std::string::size_type end = line.find_first_of(SPLITTER, begin);
  const int label = std::stoi(line.substr(begin, end - begin));
  sample.y = label > 0 ? 1 : 0;

  const std::string::size_type lineLength = line.size();
  while (end < lineLength) {
    begin = line.find_first_not_of(SPLITTER, end);
    if (begin == std::string::npos) break;
    end = line.find_first_of(INNER_SPLITTER, begin);
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
    end = line.find_first_of(INNER_SPLITTER, begin);
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
    end = line.find_first_of(SPLITTER, begin);
    const float value = std::stof(line.substr(begin, end - begin));
    if (value != 0.0) {
      sample.x.emplace_back(field, feat, value);
    }
  }
}

[[maybe_unused]] void FFMParser::parseCFFM(const std::string &line, Sample &sample) {
  sample.x.clear();
  const std::string::size_type begin = line.find_first_not_of(SPLITTER, 0);
  const std::string::size_type end = line.find_first_of(SPLITTER, begin);
  const int label = atoi(line.substr(begin, end - begin).c_str());  // NOLINT
  sample.y = label > 0 ? 1 : 0;

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
  delete[] buffer;
}

}  // namespace ftrl
