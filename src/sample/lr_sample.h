#ifndef FIRST_LR_SAMPLE222_H
#define FIRST_LR_SAMPLE222_H

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
using namespace std;

const string splitter = " ";
const string innerSplitter = ":";

class lr_sample
{
public:
    int y;
    vector<pair<string, double> > x;
    lr_sample(const string &line)
    {
        this->x.clear();
        string::size_type begin = line.find_first_not_of(splitter, 0);
        string::size_type end = line.find_first_of(splitter, begin);
        int label = atoi(line.substr(begin, end - begin).c_str());
        y = label > 0 ? 1 : -1;
        string key;
        double value;
        string::size_type lineLength = line.size();
        while (end < lineLength) {
            begin = line.find_first_not_of(splitter, end);
            if (begin == string::npos)
                break;
            end = line.find_first_of(innerSplitter, begin);
            if (end == string::npos) {
                cout << "wrong input: " << line << endl;
                throw out_of_range(line);
            }
            key = line.substr(begin, end - begin);

            begin = end + 1;
            if(begin >= lineLength) {
                cout << "wrong input: " << line << endl;
                throw out_of_range(line);
            }
            end = line.find_first_of(splitter, begin);
            value = stod(line.substr(begin, end - begin));
            if (value != 0)
                x.push_back(make_pair(key, value));
        }
    }
};



#endif //FIRST_LR_SAMPLE222_H
