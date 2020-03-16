#ifndef FIRST_PC_THREADING_H
#define FIRST_PC_THREADING_H

#include <string>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <queue>
#include <semaphore.h>
#include <iostream>
#include <fstream>
#include "pc_task.h"
using namespace std;


class pc_threading {     // producer_consumer multi_threading
public:
    pc_threading() = default;
    void init(pc_task &task, int n_threads, bool cmd = false,
              int buf_size = 20000, int log_num = 1000000);
    void openFile(const string &file_path ="");
    void rewindFile();
    void run();
private:
    ifstream ifs;
    int nthreads;
    pc_task *pTask;
    mutex bufMutex;
    sem_t semPro, semCon;
    queue<string> buffer;
    vector<thread> threadVec;
    int bufSize;
    int logNum;
//    void producerFileThread();
    void producerThread(bool cmd);
    void consumerThread(int t);
};


#endif //FIRST_PC_THREADING_H
