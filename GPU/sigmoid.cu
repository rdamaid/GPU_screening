#include "sigmoid.h"


__global__
void sigmoid_forward_gpu(float *inp, float *out, int sz_out){
    int ind = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (ind < sz_out){
        out[ind] = 1 / (1 + expf(-inp[ind]));
    }
}


__global__
void sigmoid_backward_gpu(float *inp, float *out, int sz_out){
    int ind = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (ind < sz_out){
        inp[ind] = (1 - out[ind]) * out[ind];
    }
}


Sigmoid_GPU::Sigmoid_GPU(int _sz_out){
    sz_out = _sz_out;
    
    n_blocks = (sz_out + block_size - 1) / block_size;
}


void Sigmoid_GPU::forward(float *_inp, float *_out){
    inp = _inp;
    out = _out;

    sigmoid_forward_gpu<<<n_blocks, block_size>>>(inp, out, sz_out);
    cudaDeviceSynchronize();
}


void Sigmoid_GPU::backward(){    
    sigmoid_backward_gpu<<<n_blocks, block_size>>>(inp, out, sz_out);
    cudaDeviceSynchronize();
}
