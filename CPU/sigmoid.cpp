#include "sigmoid.h"
#include <cmath>


void sigmoid_forward_cpu(float *inp, float *out, int sz_out){
    for (int i=0; i<sz_out; i++){
        out[i] = 1 / (1 + exp(-inp[i]));
    }
}


void sigmoid_backward_cpu(float *inp, float *out, int sz_out){
    for (int i=0; i<sz_out; i++){
        inp[i] = (1 - out[i]) * out[i];
    }
}


Sigmoid_CPU::Sigmoid_CPU(int _sz_out){
    sz_out = _sz_out;
}


void Sigmoid_CPU::forward(float *_inp, float *_out){
    inp = _inp;
    out = _out;

    sigmoid_forward_cpu(inp, out, sz_out);
}


void Sigmoid_CPU::backward(){
    sigmoid_backward_cpu(inp, out, sz_out);
}
