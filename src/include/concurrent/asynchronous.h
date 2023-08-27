#ifndef FTRL_FFM_ASYNCHRONOUS_H
#define FTRL_FFM_ASYNCHRONOUS_H

#include <future>
#include <utility>

template <typename F, typename... Args>
inline auto execute_async(F &&f, Args &&...params)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
  return std::async(std::launch::async, std::forward<F>(f), std::forward<Args>(params)...);
}

#endif  // FTRL_FFM_ASYNCHRONOUS_H
