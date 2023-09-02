#ifndef FTRL_FFM_COMMON_H
#define FTRL_FFM_COMMON_H

#include <cstdio>
#include <fstream>
#include <string_view>

namespace ftrl {

const std::string_view test_file_path = "./test_file.txt";

const std::string_view test_samples =
    "0 0:1:1 1:13:1 2:21:1 3:31:1\n"
    "1 0:4:1 1:11:1 2:23:1 3:32:1\n"
    "1 0:2:1 1:13:1 2:25:1 3:34:1\n"
    "0 0:1:1 1:14:1 2:21:1 3:32:1\n"
    "0 0:2:1 1:15:1 2:22:1 3:34:1\n"
    "1 0:4:1 1:11:1 2:21:1 3:35:1\n"
    "1 0:5:1 1:12:1 2:23:1 3:31:1\n"
    "1 0:5:1 1:12:1 2:25:1 3:38:1\n"
    "0 0:2:1 1:11:1 2:24:1 3:37:1\n"
    "1 0:1:1 1:15:1 2:22:1 3:35:1";

void write_test_data(std::string_view file_path) {
  std::ofstream ofs(file_path.data());
  if (!ofs.good()) {
    std::cerr << "Failed to write file " << file_path << std::endl;
    return;
  }
  ofs << test_samples;
  ofs.close();
}

void remove_test_file(std::string_view file_path) {
  if (std::remove(file_path.data()) != 0) {
    std::cerr << "Failed to remove file " << file_path << std::endl;
  } else {
    std::cout << "Test file removed!" << std::endl;
  }
}

}  // namespace ftrl

#endif  // FTRL_FFM_COMMON_H
