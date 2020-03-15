//
// Created by massquantity on 2/23/20.
//

#ifndef FIRST_TEST_TASK_H
#define FIRST_TEST_TASK_H

#include <iostream>
#include "pc_task.h"
#include "../sample/lr_sample.h"
using namespace std;

class test_task : public pc_task
{
public:
    test_task(){}
    virtual void run_task(vector<string>& dataBuffer)
    {
        cout << "==========\n";
        for(int i = 0; i < dataBuffer.size(); ++i)
        {
            lr_sample sample(dataBuffer[i]);
            cout << sample.y << " || ";
            for (auto &j : sample.x)
                cout << j.first << " " << j.second << " ";
            cout << endl;
            cout << dataBuffer[i] << endl;
        }
        cout << "**********\n";
    }
};


#endif //FIRST_TEST_TASK_H
