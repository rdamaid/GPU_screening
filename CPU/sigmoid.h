#ifndef SIGMOID_CPU_H
#define SIGMOID_CPU_H


#include "../utils/module.h"


class Sigmoid_CPU: public Module{
    public:
        Sigmoid_CPU(int _sz_out);
        void forward(float *_inp, float *_out);
        void backward();
};


#endif
