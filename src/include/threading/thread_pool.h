#ifndef FTRL_FFM_THREAD_POOL_H
#define FTRL_FFM_THREAD_POOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <atomic>

class ThreadPool {
public:
  explicit ThreadPool(size_t n_threads);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;

  ThreadPool &operator=(const ThreadPool &) = delete;
  ThreadPool &operator=(ThreadPool &&) = delete;

  template <typename F, typename... Args>
  auto enqueue(F &&f, Args &&...args)
      -> std::future<typename std::invoke_result<F, Args...>::type>;

  void synchronize(int wait_count);
  size_t get_num_threads();

private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> thread_queue;
  std::mutex queue_mutex;
  std::condition_variable thread_cv;
  std::mutex sync_mutex;
  std::condition_variable sync_cv;
  std::atomic<bool> stop { false };
  std::atomic<int> sync_var { 0 };
};

inline ThreadPool::ThreadPool(size_t n_threads) {
  for (size_t i = 0; i < n_threads; i++) {
    workers.emplace_back([&] {
      while (true) {
        std::function<void()> task;
        std::unique_lock<std::mutex> lock(queue_mutex);
        thread_cv.wait(lock, [&] {
          return stop || !thread_queue.empty();
        });
        if (stop && thread_queue.empty()) {
          return;
        }
        task = std::move(thread_queue.front());
        thread_queue.pop();
        lock.unlock();
        // perform function
        task();
        sync_var.fetch_add(1, std::memory_order_relaxed);
        sync_cv.notify_one();
      }
    });
  }
}

template <typename F, typename... Args>
auto ThreadPool::enqueue(F &&f, Args &&...args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
  if (stop) {
    throw std::runtime_error("can't enqueue on stopped ThreadPool...");
  }
  using return_type = typename std::invoke_result<F, Args...>::type;
  std::function<decltype(f(args...))()> func =
      std::bind(std::forward<F>(f), std::forward<Args>(args)...);
  auto task_ptr = std::make_shared<std::packaged_task<return_type()>>(func);
  // std::future<return_type> res = task->get_future();
  std::unique_lock<std::mutex> lock(queue_mutex);
  thread_queue.emplace([task_ptr] { (*task_ptr)(); });
  lock.unlock();
  thread_cv.notify_one();
  return task_ptr->get_future();
}

inline void ThreadPool::synchronize(int wait_count) {
  {
    std::unique_lock<std::mutex> lock(sync_mutex);
    sync_cv.wait(lock, [&] { return sync_var == wait_count;} );
  }
  sync_var = 0;
}

inline size_t ThreadPool::get_num_threads() {
  return workers.size();
}

inline ThreadPool::~ThreadPool() {
  stop = true;
  thread_cv.notify_all();
  for (auto &worker : workers) {
    worker.join();
  }
}

#endif //FTRL_FFM_THREAD_POOL_H