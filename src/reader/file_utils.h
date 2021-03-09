#ifndef FTRL_FFM_FILE_UTILS_H
#define FTRL_FFM_FILE_UTILS_H

inline void splitString(const std::string &line,
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

inline std::string getFileType(const std::string filePath) {
  std::ifstream ifs(filePath);
  if (!ifs.good()) {
    std::cerr << "fail to open " << filePath << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string line;
  getline(ifs, line);
  ifs.close();
  std::vector<std::string> split_line;
  splitString(line, " ", split_line);
  int count = 0;
  for (char c : split_line[1]) {
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
    exit(EXIT_FAILURE);
  }
}

#endif //FTRL_FFM_FILE_UTILS_H