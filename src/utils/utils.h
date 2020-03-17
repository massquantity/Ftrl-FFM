#ifndef FIRST_UTILS_H
#define FIRST_UTILS_H

#include <string>
#include <vector>
#include <random>
using namespace std;

class utils
{
public:
    static void splitString(const string &line, const string &delimiter, vector<string> *r);
    static int sgn(double x);
    static double uniform();
    static double gaussian(double mean, double stddev);
};


#endif //FIRST_UTILS_H
