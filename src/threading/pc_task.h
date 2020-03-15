#ifndef FIRST_PC_TASK_H
#define FIRST_PC_TASK_H

#include <string>
#include <vector>
using namespace std;

class pc_task  // producer_consumer task
{
public:
    pc_task() = default;
    virtual void run_task(vector<string> &dataBuffer, int t = 0) = 0;
};



#endif //FIRST_PC_TASK_H
