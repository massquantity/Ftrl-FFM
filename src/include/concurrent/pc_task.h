#ifndef FTRL_FFM_PC_TASK_H
#define FTRL_FFM_PC_TASK_H

#include <semaphore.h>

#include <atomic>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

class PcTask {
 public:
  PcTask() = default;
  PcTask(int n_threads, bool cmd) : n_threads(n_threads), cmd(cmd) {}
  void open_file(const std::string &file_path);
  void rewind_file();
  void run();
  virtual void run_task(std::vector<std::string> &data_buffer, int t) = 0;
  virtual ~PcTask() = default;

 protected:
  std::ifstream ifs;
  std::mutex buf_mutex;
  std::condition_variable pro_cv, con_cv;
  std::queue<std::string> buffer;
  std::vector<std::thread> thread_vec;
  int n_threads = 1;
  int buf_size = 20000;
  int log_num = 1000000;
  std::atomic<bool> input_end{false};
  bool cmd{false};
  void producer_thread();
  void consumer_thread(int t);
};

#endif  // FTRL_FFM_PC_TASK_H
