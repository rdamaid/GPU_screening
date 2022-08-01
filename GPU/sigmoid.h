#ifndef SIGMOID_GPU_H
#define SIGMOID_GPU_H


#include "../utils/module.h"


class Sigmoid_GPU: public Module{
    public:
        int n_blocks;
        
        Sigmoid_GPU(int _sz_out);
        void forward(float *_inp, float *_out);
        void backward();
};


#endif
