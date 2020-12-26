#ifndef FTRL_FFM_PC_TASK_H
#define FTRL_FFM_PC_TASK_H

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <mutex>
#include <thread>
#include <queue>
#include <vector>
#include <unordered_map>
#include <semaphore.h>
#include <condition_variable>

class pc_task {
public:
  pc_task() = default;
  pc_task(int n_threads, bool cmd = false): nthreads(n_threads), cmd(cmd) { }
  void openFile(const std::string &file_path);
  void rewindFile();
  void run();
  virtual void run_task(std::vector<std::string> &dataBuffer, int t = 0) = 0;
  virtual ~pc_task() = default;

private:
  std::ifstream ifs;
  std::mutex bufMutex;
  std::condition_variable pro_cv, con_cv;
  std::queue<std::string> buffer;
  std::vector<std::thread> threadVec;
  int nthreads;
  int bufSize = 20000;
  int logNum = 1000000;
  bool input_end = false;
  bool cmd;
  void producerThread();
  void consumerThread(int t);
};

void pc_task::openFile(const std::string &file_path) {
  if (!cmd) {
    ifs.open(file_path, std::istream::in | std::istream::binary);
    if (!ifs) {
      fprintf(stderr, "open file <%s> error. \n", file_path.c_str());
      EXIT_FAILURE;
    }
  }
}

void pc_task::rewindFile() {
  if (ifs.eof()) {
    ifs.clear();
    ifs.seekg(0L, std::ifstream::beg);
  }
}

void pc_task::run() {
  input_end = false;
  threadVec.clear();
  threadVec.emplace_back(std::thread(&pc_task::producerThread, this));
  for (int i = 0; i < nthreads; i++) {
    threadVec.emplace_back(std::thread(&pc_task::consumerThread, this, i));
  }
  for (std::thread &t : threadVec)
    t.join();
}

void pc_task::producerThread() {
  std::string line;
  int line_num = 0;
  while (true) {
    std::unique_lock<std::mutex> lck(bufMutex);
    pro_cv.wait(lck, [&]() { return buffer.empty(); });
    for (int i = 0; i < bufSize; i++) {
      if (cmd ? (!getline(std::cin, line)) : (!getline(ifs, line))) {
        input_end = true;
        break;
      }
      buffer.push(line);
      line_num++;
      if (line_num % logNum == 0) {
        std::cout << line_num << " lines finished..." << std::endl;
      }
    }
    lck.unlock();
    con_cv.notify_all();
    if (input_end) break;
  }
}

void pc_task::consumerThread(int t) {
  bool thread_end = false;
  std::vector<std::string> input_vec;
  input_vec.reserve(bufSize);
  while (true) {
    input_vec.clear();
    std::unique_lock<std::mutex> lck(bufMutex);
    con_cv.wait(lck, [&]() {
      thread_end = input_end;
      return !buffer.empty() || input_end;
    });
    if (buffer.empty() && input_end) break;

    while (!buffer.empty()) {
      input_vec.emplace_back(buffer.front());
      buffer.pop();
    }
    lck.unlock();
    pro_cv.notify_one();
    run_task(input_vec, t);
    if (thread_end) break;
  }
  con_cv.notify_one();
}

#endif //FTRL_FFM_PC_TASK_H