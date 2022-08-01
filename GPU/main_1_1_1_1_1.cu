#include <chrono>

#include "linear.h"
#include "relu.h"
#include "sigmoid.h"
#include "train.h"
#include "../data/read_csv.h"


int main(){
    std::chrono::steady_clock::time_point begin, end;

    int bs = 405, n_in = 1281, n_epochs = 100;
    int bs_test = 135;
    int n_hidden1 = 10;
    int n_hidden2 = 3;
    float learning_rate = 0.01f;
    // ReLU

    float *inp, *targ, *inp_test, *targ_test;  
    cudaMallocManaged(&inp, bs*n_in*sizeof(float));
    cudaMallocManaged(&targ, (bs+1)*sizeof(float));
    cudaMallocManaged(&inp_test, bs*n_in*sizeof(float));
    cudaMallocManaged(&targ_test, (bs+1)*sizeof(float));
    
    begin = std::chrono::steady_clock::now();
    read_csv(inp,  "/content/drive/MyDrive/Tugas Akhir/kode1/data/nr/X_train.csv");
    read_csv(targ, "/content/drive/MyDrive/Tugas Akhir/kode1/data/nr/y_train.csv");
    read_csv(inp_test,  "/content/drive/MyDrive/Tugas Akhir/kode1/data/nr/X_test.csv");
    read_csv(targ_test, "/content/drive/MyDrive/Tugas Akhir/kode1/data/nr/y_test.csv");
    end = std::chrono::steady_clock::now();
    std::cout << "Data reading time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;
    
    Linear_GPU* lin1 = new Linear_GPU(bs, n_in, n_hidden1, learning_rate);
    ReLU_GPU* relu1 = new ReLU_GPU(bs*n_hidden1);
    Linear_GPU* lin2 = new Linear_GPU(bs, n_hidden1, 1, learning_rate);

    std::cout << "2" << std::endl;

    std::vector<Module*> layers = {lin1, relu1, lin2};
    Sequential_GPU seq(layers);

    std::cout << "3" << std::endl;

    begin = std::chrono::steady_clock::now();
    train_gpu(seq, inp, targ, bs, n_in, n_epochs, bs_test, inp_test, targ_test);
    end = std::chrono::steady_clock::now();
    std::cout << "Training and Testing time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;

    return 0;
}
