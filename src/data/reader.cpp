#include "data/reader.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <regex>
#include <thread>

namespace ftrl {

Reader::Reader(const std::string &file_type) {
  if (file_type == "libsvm") {
    parser = std::make_shared<LibsvmParser>();
  } else if (file_type == "libffm") {
    parser = std::make_shared<FFMParser>();
  }
}

void Reader::load_from_file(const std::string &file_name, int n_threads) {
  printf("Loading data from file: %s\n", file_name.c_str());
  auto start = std::chrono::steady_clock::now();
  std::ifstream ifs(file_name);
  if (!ifs.good()) {
    std::cerr << "fail to open " << file_name << std::endl;
    exit(EXIT_FAILURE);  // NOLINT
  }

  auto file_len = [&](std::ifstream &f) {
    f.seekg(0L, std::ios_base::end);
    return f.tellg();
  };
  auto len = file_len(ifs);
  std::vector<off_t> partitions(n_threads + 1);
  partitions[0] = 0;
  partitions[n_threads] = len;

  std::string unused;
  for (size_t i = 1; i < n_threads; i++) {
    const uint64 pos = len / n_threads * i;
    ifs.clear();
    ifs.seekg(pos, std::ios_base::beg);  // NOLINT
    std::getline(ifs, unused);
    partitions[i] = ifs.tellg();
  }
  ifs.close();

  // todo: read from memory
  // std::vector<Sample> partition_data[n_threads];
  std::vector<std::vector<Sample>> partition_data(n_threads);
  std::vector<std::thread> data_threads;
  for (size_t i = 0; i < n_threads; i++) {
    data_threads.emplace_back([i, &file_name, &partitions, &partition_data, this] {
      std::ifstream ifs_t(file_name);
      ifs_t.seekg(partitions[i]);
      std::string line;
      while (ifs_t.tellg() < partitions[i + 1] && std::getline(ifs_t, line)) {
        Sample sample;
        parser->parse(line, sample);
        partition_data[i].emplace_back(sample);
      }
      ifs_t.close();
    });
  }
  for (auto &t : data_threads) {
    t.join();
  }

  const size_t totalSize = std::accumulate(
      partition_data.begin(), partition_data.end(), size_t(0),
      [](size_t acc, std::vector<Sample> &samples) { return acc + samples.size(); });
  data.resize(totalSize);
  dataSize = totalSize;
  size_t cur = 0;
  for (const auto &pd : partition_data) {
    std::move(pd.begin(), pd.end(), data.begin() + cur);  // NOLINT
    cur += pd.size();
  }
  printf("Total number of samples loaded: %zu\n", totalSize);
  auto end = std::chrono::steady_clock::now();
  auto dur = std::chrono::duration<double>(end - start).count();
  printf("time elapsed: %.4lfs\n", dur);
}

}  // namespace ftrl
