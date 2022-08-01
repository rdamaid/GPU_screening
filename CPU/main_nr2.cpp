#include <chrono>

#include "linear.h"
#include "relu.h"
#include "sigmoid.h"
#include "train.h"
#include "../data/read_csv.h"


int main(int argc, char **argv){

    std::chrono::steady_clock::time_point begin, end;

    std::string dataset = "null";
    int n_in = 1281, n_epochs = 100;
    int bs = 0, bs_test = 0;

    std::cout << "You have entered " << argc
         << " arguments:" << "\n";
    
    // dataset
    if (argv[1] == std::string("E")) {
        dataset = "e";
        bs = 13167;
        bs_test = 4389;
    } 
    else if (argv[1] == std::string("IC")) {
        dataset = "ic";
        bs = 6642;
        bs_test = 2214;
    }
    else if (argv[1] == std::string("GPCR")) {
        dataset = "gpcr";
        bs = 2857;
        bs_test = 953;
    }
    else if (argv[1] == std::string("NR")) {
        dataset = "nr";
        bs = 405;
        bs_test = 135;
    }

    float *inp = new float[bs*n_in], *targ = new float[bs+1];
    float *inp_test = new float[bs*n_in], *targ_test = new float[bs+1];

    std::string data_path = "/content/drive/MyDrive/Tugas Akhir/kode1/data/";
    std::string inp_path = data_path + dataset + "/X_train.csv";
    std::string targ_path = data_path + dataset + "/y_train.csv";
    std::string inp_test_path = data_path + dataset + "/X_test.csv";
    std::string targ_test_path = data_path + dataset + "/y_test.csv";

    begin = std::chrono::steady_clock::now();
    read_csv(inp,  inp_path);
    read_csv(targ, targ_path);
    read_csv(inp_test,  inp_test_path);
    read_csv(targ_test, targ_test_path);
    end = std::chrono::steady_clock::now();
    std::cout << "Data reading time NR: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;

    int n_hidden1 = 0, n_hidden2 = 0, n_hidden3 = 0;
    float learning_rate = 0.0f;

    // hidden layer ke-0
    if (argv[2] == std::string("1")) {
        n_hidden1 =  300;
    }
    else if (argv[2] == std::string("2")) {
        n_hidden1 =  500;
    }
    std::cout << "n_hidden1 = " << n_hidden1 << std::endl;

    // hidden layer ke-(n-1)
    if (argv[3] == std::string("1")) {
        n_hidden2 =  n_hidden1 / 2;
        n_hidden3 =  n_hidden2 / 2;
    }
    else if (argv[3] == std::string("2")) {
        n_hidden2 =  n_hidden1 * 2 / 3;
        n_hidden3 =  n_hidden2 * 2 / 3;
    }
    std::cout << "n_hidden2 = " << n_hidden2 << std::endl;
    std::cout << "n_hidden3 = " << n_hidden3 << std::endl;

    // jumlah hidden layer
    std::cout << "jumlah hidden layer = " << std::string(argv[4]) << std::endl;

    // learning rate
    if (argv[5] == std::string("1")) {
        learning_rate = 0.01f;
    }
    else if (argv[5] == std::string("2")) {
        learning_rate = 0.02f;
    }
    std::cout << "learning_rate = " << learning_rate << std::endl;

    // membuat model berdasarkan jumlah HL dan Activation function
    if (argv[4] == std::string("1") && argv[6] == std::string("1")) {
        Linear_CPU* lin1 = new Linear_CPU(bs, n_in, n_hidden1, learning_rate);
        ReLU_CPU* func1 = new ReLU_CPU(bs*n_hidden1);
        Linear_CPU* lin2 = new Linear_CPU(bs, n_hidden1, n_hidden2, learning_rate);
        ReLU_CPU* func2 = new ReLU_CPU(bs*n_hidden2);
        Linear_CPU* lin3 = new Linear_CPU(bs, n_hidden2, 1, learning_rate);

        std::vector<Module*> layers = {lin1, func1, lin2, func2, lin3};
        Sequential_CPU seq(layers);
        
        begin = std::chrono::steady_clock::now();
        train_cpu(seq, inp, targ, bs, n_in, n_epochs, bs_test, inp_test, targ_test);
        end = std::chrono::steady_clock::now();
        std::cout << "Training and Testing time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;
    }
    else if (argv[4] == std::string("1") && argv[6] == std::string("2")) {
        Linear_CPU* lin1 = new Linear_CPU(bs, n_in, n_hidden1, learning_rate);
        Sigmoid_CPU* func1 = new Sigmoid_CPU(bs*n_hidden1);
        Linear_CPU* lin2 = new Linear_CPU(bs, n_hidden1, n_hidden2, learning_rate);
        Sigmoid_CPU* func2 = new Sigmoid_CPU(bs*n_hidden2);
        Linear_CPU* lin3 = new Linear_CPU(bs, n_hidden2, 1, learning_rate);

        std::vector<Module*> layers = {lin1, func1, lin2, func2, lin3};
        Sequential_CPU seq(layers);
        
        begin = std::chrono::steady_clock::now();
        train_cpu(seq, inp, targ, bs, n_in, n_epochs, bs_test, inp_test, targ_test);
        end = std::chrono::steady_clock::now();
        std::cout << "Training and Testing time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;
    }
    else if (argv[4] == std::string("2") && argv[6] == std::string("1")) {
        Linear_CPU* lin1 = new Linear_CPU(bs, n_in, n_hidden1, learning_rate);
        ReLU_CPU* func1 = new ReLU_CPU(bs*n_hidden1);
        Linear_CPU* lin2 = new Linear_CPU(bs, n_hidden1, n_hidden2, learning_rate);
        ReLU_CPU* func2 = new ReLU_CPU(bs*n_hidden2);
        Linear_CPU* lin3 = new Linear_CPU(bs, n_hidden2, n_hidden3, learning_rate);
        ReLU_CPU* func3 = new ReLU_CPU(bs*n_hidden3);
        Linear_CPU* lin4 = new Linear_CPU(bs, n_hidden3, 1, learning_rate);

        std::vector<Module*> layers = {lin1, func1, lin2, func2, lin3, func3, lin4};
        Sequential_CPU seq(layers);
        
        begin = std::chrono::steady_clock::now();
        train_cpu(seq, inp, targ, bs, n_in, n_epochs, bs_test, inp_test, targ_test);
        end = std::chrono::steady_clock::now();
        std::cout << "Training and Testing time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;
    }
    else if (argv[4] == std::string("2") && argv[6] == std::string("2")) {
        Linear_CPU* lin1 = new Linear_CPU(bs, n_in, n_hidden1, learning_rate);
        Sigmoid_CPU* func1 = new Sigmoid_CPU(bs*n_hidden1);
        Linear_CPU* lin2 = new Linear_CPU(bs, n_hidden1, n_hidden2, learning_rate);
        Sigmoid_CPU* func2 = new Sigmoid_CPU(bs*n_hidden2);
        Linear_CPU* lin3 = new Linear_CPU(bs, n_hidden2, n_hidden3, learning_rate);
        Sigmoid_CPU* func3 = new Sigmoid_CPU(bs*n_hidden3);
        Linear_CPU* lin4 = new Linear_CPU(bs, n_hidden3, 1, learning_rate);

        std::vector<Module*> layers = {lin1, func1, lin2, func2, lin3, func3, lin4};
        Sequential_CPU seq(layers);
        
        begin = std::chrono::steady_clock::now();
        train_cpu(seq, inp, targ, bs, n_in, n_epochs, bs_test, inp_test, targ_test);
        end = std::chrono::steady_clock::now();
        std::cout << "Training and Testing time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;
    }

    return 0;
    // float learning_rate = 0.01f;

    // float *inp = new float[bs*n_in], *targ = new float[bs+1];
    // float *inp_test = new float[bs*n_in], *targ_test = new float[bs+1];
    
    // begin = std::chrono::steady_clock::now();
    // read_csv(inp,  "/content/drive/MyDrive/Tugas Akhir/kode1/data/nr/X_train.csv");
    // read_csv(targ, "/content/drive/MyDrive/Tugas Akhir/kode1/data/nr/y_train.csv");
    // read_csv(inp_test,  "/content/drive/MyDrive/Tugas Akhir/kode1/data/nr/X_test.csv");
    // read_csv(targ_test, "/content/drive/MyDrive/Tugas Akhir/kode1/data/nr/y_test.csv");
    // end = std::chrono::steady_clock::now();
    // std::cout << "Data reading time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;
    
    // Linear_CPU* lin1 = new Linear_CPU(bs, n_in, n_hidden1, learning_rate);
    // Sigmoid_CPU* func1 = new Sigmoid_CPU(bs*n_hidden1);
    // Linear_CPU* lin2 = new Linear_CPU(bs, n_hidden1, 1, learning_rate);
    // Sigmoid_CPU* func2 = new Sigmoid_CPU(bs*n_hidden2);
    // Linear_CPU* lin3 = new Linear_CPU(bs, n_hidden2, 1, learning_rate);

    // std::vector<Module*> layers = {lin1, func1, lin2};
    
    // Sequential_CPU seq(layers);
    // begin = std::chrono::steady_clock::now();
    // train_cpu(seq, inp, targ, bs, n_in, n_epochs, bs_test, inp_test, targ_test);
    // end = std::chrono::steady_clock::now();
    // std::cout << "Training and Testing time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;

    // return 0;
}
