#include "threading/pc_task.h"

void PcTask::open_file(const std::string &file_path) {
  if (!cmd) {
    ifs.open(file_path, std::istream::in | std::istream::binary);
    if (!ifs.good()) {
      fprintf(stderr, "open file <%s> error. \n", file_path.c_str());  // NOLINT
      exit(EXIT_FAILURE);  // NOLINT
    }
  }
}

void PcTask::rewind_file() {
  if (ifs.eof()) {
    ifs.clear();
    ifs.seekg(0L, std::ifstream::beg);
  }
}

void PcTask::run() {
  input_end = false;
  thread_vec.clear();
  thread_vec.emplace_back(std::thread(&PcTask::producer_thread, this));  // NOLINT
  for (int i = 0; i < n_threads; i++) {
    thread_vec.emplace_back(std::thread(&PcTask::consumer_thread, this, i));  // NOLINT
  }
  for (std::thread &t : thread_vec)
    t.join();
}

void PcTask::producer_thread() {
  std::string line;
  int line_num = 0;
  while (true) {
    std::unique_lock<std::mutex> lck(buf_mutex);
    pro_cv.wait(lck, [&]() { return buffer.empty(); });
    for (int i = 0; i < buf_size; i++) {
      if (cmd ? (!getline(std::cin, line)) : (!getline(ifs, line))) {
        input_end = true;
        break;
      }
      buffer.push(line);
      line_num++;
      if (line_num % log_num == 0) {
        std::cout << line_num << " lines finished..." << std::endl;
      }
    }
    lck.unlock();
    con_cv.notify_all();
    if (input_end) break;
  }
}

void PcTask::consumer_thread(int t) {
  thread_local bool thread_end = false;
  std::vector<std::string> input_vec;
  input_vec.reserve(buf_size);
  while (true) {
    input_vec.clear();
    std::unique_lock<std::mutex> lck(buf_mutex);
    con_cv.wait(lck, [&]() {
      thread_end = input_end.load();
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
