#ifndef FTRL_FFM_FILE_UTILS_H
#define FTRL_FFM_FILE_UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

void split_string(const std::string &line,
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

std::string detect_file_type(const std::string &file_path) {
  std::ifstream ifs(file_path);
  if (!ifs.good()) {
    std::cerr << "fail to open " << file_path << std::endl;
    exit(EXIT_FAILURE);  // NOLINT
  }
  std::string line;
  std::getline(ifs, line);
  ifs.close();
  std::vector<std::string> split_line;
  split_string(line, " ", split_line);
  int count = 0;
  for (const char c : split_line[1]) {
    if (c == ':') {
      count++;
    }
  }
  if (count == 1) {
    return "libsvm";
  }
  else if (count == 2) {
    return "libffm";
  }
  else {
    std::cerr << "unknown file format..." << std::endl;
    exit(EXIT_FAILURE);  // NOLINT
  }
}

#endif //FTRL_FFM_FILE_UTILS_H