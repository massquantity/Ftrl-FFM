#ifndef FTRL_FFM_READER_H
#define FTRL_FFM_READER_H

#include <algorithm>
#include <fstream>
#include <thread>
#include "parser.h"

namespace ftrl {

class Reader {
public:
  void loadFromFile(const std::string &fileName, int numThreads = 1);
  size_t getSize() const { return dataSize; }
  size_t dataSize;
  std::vector<Sample> data;
};

void Reader::loadFromFile(const std::string &fileName, int numThreads) {
  printf("Loading data from file: %s\n", fileName.c_str());
  auto start = std::chrono::steady_clock::now();
  std::ifstream ifs(fileName);
  if (!ifs.good()) {
    std::cerr << "fail to open " << fileName << std::endl;
    exit(EXIT_FAILURE);
  }

  auto fileLen = [&](std::ifstream &f) {
    f.seekg(0L, std::ios_base::end);
    return f.tellg();
  };
  auto len = fileLen(ifs);
  std::vector<off_t> partitions(numThreads + 1);
  partitions[0] = 0;
  partitions[numThreads] = len;

  std::string unused;
  for (size_t i = 1; i < numThreads; i++) {
    unsigned long pos = len / numThreads * i;
    ifs.clear();
    ifs.seekg(pos, std::ios_base::beg);
    getline(ifs, unused);
    partitions[i] = ifs.tellg();
  }
  ifs.close();

  // todo: read from memory
  // std::vector<Sample> partitionData[numThreads];
  std::vector<std::vector<Sample>> partitionData(numThreads);
  std::vector<std::thread> dataThreads;
  for (size_t i = 0; i < numThreads; i++) {
    dataThreads.emplace_back([i, &fileName, &partitions, &partitionData] {
      std::ifstream ifs_t(fileName);
      ifs_t.seekg(partitions[i]);
      std::string line;
      while (ifs_t.tellg() < partitions[i+1] && getline(ifs_t, line)) {
        Sample sample;
        Parser::parse(line, sample);
        partitionData[i].emplace_back(sample);
      }
      ifs_t.close();
    });
  }
  for (auto &t : dataThreads) {
    t.join();
  }

  size_t totalSize = std::accumulate(partitionData.begin(), partitionData.end(),
      size_t(0), [](size_t l, std::vector<Sample> &r) { return l + r.size(); });
  data.resize(totalSize);
  dataSize = totalSize;
  size_t cur = 0;
  for (const auto &pd : partitionData) {
    std::move(pd.begin(), pd.end(), data.begin() + cur);
    cur += pd.size();
  }
  printf("Total number of samples loaded: %zu\n", totalSize);
  auto end = std::chrono::steady_clock::now();
  auto dur = std::chrono::duration<double>(end - start).count();
  printf("time elapsed: %.4lfs\n", dur);
}

}

#endif //FTRL_FFM_READER_H