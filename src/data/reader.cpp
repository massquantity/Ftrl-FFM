#include "data/reader.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>

#include "concurrent/asynchronous.h"
#include "utils/utils.h"

namespace ftrl {

Reader::Reader(const std::string &file_type) {
  if (file_type == "libsvm") {
    parser = std::make_shared<LibsvmParser>();
  } else if (file_type == "libffm") {
    parser = std::make_shared<FFMParser>();
  }
}

std::vector<int64> Reader::get_data_partition(std::string_view file_name, int n_threads) {
  std::ifstream ifs(file_name.data());
  if (!ifs.good()) {
    std::cerr << "fail to open " << file_name << std::endl;
    exit(EXIT_FAILURE);  // NOLINT
  }

  auto get_file_len = [](std::ifstream &f) {
    f.seekg(0L, std::ios_base::end);
    return f.tellg();
  };
  auto len = get_file_len(ifs);
  std::vector<int64> partitions(n_threads + 1);
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
  return std::move(partitions);
}

void Reader::load_from_file(std::string_view file_name, int n_threads) {
  auto parse_func = [&file_name, this](int64 start, int64 end, std::vector<Sample> &samples) {
    std::vector<Sample> tmp_samples;
    std::ifstream ifs_t(file_name.data());
    ifs_t.seekg(start);
    std::string line;
    while (ifs_t.tellg() < end && std::getline(ifs_t, line)) {
      Sample sample;
      parser->parse(line, sample);
      tmp_samples.emplace_back(sample);
    }
    ifs_t.close();
    samples = std::move(tmp_samples);
  };

  printf("Loading data from file: %s\n", file_name.data());
  auto start = std::chrono::steady_clock::now();
  std::vector<int64> partitions = get_data_partition(file_name, n_threads);
  std::vector<std::vector<Sample>> partition_data(n_threads);
  std::vector<std::future<void>> futures;
  for (size_t i = 0; i < n_threads; i++) {
    auto samples = std::ref(partition_data[i]);
    auto fut = execute_async(parse_func, partitions[i], partitions[i + 1], samples);
    futures.emplace_back(std::move(fut));
  }
  std::for_each(futures.cbegin(), futures.cend(), [](const auto &f) { f.wait(); });

  const size_t total_size = std::accumulate(
      partition_data.begin(), partition_data.end(), size_t(0),
      [](size_t acc, std::vector<Sample> &samples) { return acc + samples.size(); });
  printf("Total number of samples loaded: %zu\n", total_size);

  size_t cur = 0;
  data.resize(total_size);
  for (const auto &pd : partition_data) {
    std::move(pd.cbegin(), pd.cend(), data.begin() + cur);  // NOLINT
    cur += pd.size();
  }
  this->dataSize = total_size;
  const double data_time = utils::compute_time(start);
  printf("parsing data time: %.4lfs\n", data_time);
}

}  // namespace ftrl
