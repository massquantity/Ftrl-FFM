#include "pc_threading.h"
#include <chrono>
#include <unistd.h>

void pc_threading::init(pc_task &task, int n_threads, bool cmd,
                        int buf_size, int log_num)
{
    pTask = &task;
    nthreads = n_threads;
    bufSize = buf_size;
    logNum = log_num;
    sem_init(&semPro, 0, 1);
    sem_init(&semCon, 0, 0);
    threadVec.clear();
    threadVec.emplace_back(thread(&pc_threading::producerThread, this, cmd));
    for (int i = 0; i < nthreads; ++i) {
        threadVec.emplace_back(thread(&pc_threading::consumerThread, this, i));
    }
}

void pc_threading::openFile(const string &file_path) {
    ifs.open(file_path, istream::in | istream::binary);
    if (!ifs) {
        fprintf(stderr, "open file <%s> error.\n", file_path.c_str());
        EXIT_FAILURE;
    }
}

void pc_threading::rewindFile() {
    if (ifs.eof()) {
        ifs.clear();
        ifs.seekg(0L, ifstream::beg);
    }
}

void pc_threading::run()
{
    for (int i = 0; i < threadVec.size(); ++i) {
        threadVec[i].join();
    }
}

void pc_threading::producerThread(bool cmd)
{
    string line;
    int line_num = 0;
    bool finished_flag = false;
    while (true)
    {
        sem_wait(&semPro);
        bufMutex.lock();
        for (int i = 0; i < bufSize; ++i)
        {
            if (cmd ? (!getline(cin, line)) : (!getline(ifs, line)))
            {
                finished_flag = true;
                break;
            }
            line_num++;
            buffer.push(line);
            if (line_num % logNum == 0)
                cout << line_num << " lines finished..." << endl;
        }
        bufMutex.unlock();
        sem_post(&semCon);
        if (finished_flag)
            break;
    }
}

void pc_threading::consumerThread(int t)
{
    int con_lin_num = 0;
    bool finished_flag = false;
    vector<string> input_vec;
    input_vec.reserve(bufSize);
    while (true)
    {
        input_vec.clear();
        sem_wait(&semCon);
        bufMutex.lock();
        for (int i = 0; i < bufSize; ++i) {
            if (buffer.empty()) {
                finished_flag = true;
                break;
            }
            con_lin_num++;
            input_vec.emplace_back(buffer.front());
            buffer.pop();
        }
        bufMutex.unlock();
        sem_post(&semPro);
        pTask->run_task(input_vec, t);
        if (finished_flag)
            break;
    }
    sem_post(&semCon);
}




