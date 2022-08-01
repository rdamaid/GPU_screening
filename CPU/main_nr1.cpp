#include <chrono>

#include "linear.h"
#include "relu.h"
#include "sigmoid.h"
#include "train.h"
#include "../data/read_csv.h"


int main(){
    std::chrono::steady_clock::time_point begin, end;

    int bs = 405, n_in = 1281, n_epochs = 100;
    int n_hidden1 = 100;
    int n_hidden2 =  20;
    int bs_test = 135;
    float learning_rate = 0.01f;

    float *inp = new float[bs*n_in], *targ = new float[bs+1];
    float *inp_test = new float[bs*n_in], *targ_test = new float[bs+1];
    
    begin = std::chrono::steady_clock::now();
    read_csv(inp,  "/content/drive/MyDrive/Tugas Akhir/kode1/data/nr/X_train.csv");
    read_csv(targ, "/content/drive/MyDrive/Tugas Akhir/kode1/data/nr/y_train.csv");
    read_csv(inp_test,  "/content/drive/MyDrive/Tugas Akhir/kode1/data/nr/X_test.csv");
    read_csv(targ_test, "/content/drive/MyDrive/Tugas Akhir/kode1/data/nr/y_test.csv");
    end = std::chrono::steady_clock::now();
    std::cout << "Data reading time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;
    
    Linear_CPU* lin1 = new Linear_CPU(bs, n_in, n_hidden1, learning_rate);
    Sigmoid_CPU* func1 = new Sigmoid_CPU(bs*n_hidden1);
    Linear_CPU* lin2 = new Linear_CPU(bs, n_hidden1, 1, learning_rate);
    Sigmoid_CPU* func2 = new Sigmoid_CPU(bs*n_hidden2);
    Linear_CPU* lin3 = new Linear_CPU(bs, n_hidden2, 1, learning_rate);

    std::vector<Module*> layers = {lin1, func1, lin2};
    Sequential_CPU seq(layers);

    begin = std::chrono::steady_clock::now();
    train_cpu(seq, inp, targ, bs, n_in, n_epochs, bs_test, inp_test, targ_test);
    end = std::chrono::steady_clock::now();
    std::cout << "Training and Testing time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;

    return 0;
}
