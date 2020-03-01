#include "utils.h"

const double kPrecision = 1e-11;

void utils::splitString(const string &line, const string &delimiter, vector<string> *r)
{
    string::size_type begin = line.find_first_not_of(delimiter, 0);
    string::size_type end = line.find_first_of(delimiter, begin);
    while (begin != string::npos || end != string::npos) {
        r->push_back(line.substr(begin, end - begin));
        begin = line.find_first_not_of(delimiter, end);
        end = line.find_first_of(delimiter, begin);
    }
}

int utils::sgn(double x)
{
    if (x > kPrecision)
        return 1;
    else
        return -1;
}

double utils::uniform()
{
    return rand() / ((double)RAND_MAX + 1.0);
}

double utils::gaussian(double mean = 0.0, double stddev = 0.01)
{
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist{mean, stddev};
    return dist(gen);
}


