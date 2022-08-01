#include <iostream>

#include "mse.h"
#include "train.h"
#include "../utils/utils.h"

void train_gpu(Sequential_GPU seq, float *inp, float *targ, int bs, int n_in, int n_epochs, int bs_test, float *inp_test, float *targ_test){
    MSE_GPU mse(bs);
    
    int sz_inp = bs*n_in;
    float *cp_inp, *out;
    cudaMallocManaged(&cp_inp, sz_inp*sizeof(float));

    for (int i=0; i<n_epochs; i++){
        set_eq(cp_inp, inp, sz_inp);

        seq.forward(cp_inp, out);
        mse.forward(seq.layers.back()->out, targ);
        
        mse.backward();
        seq.update();
    }

    seq.forward(inp, out);
    seq.forward(inp_test, out);

    int tp = 0, tn = 0, fp = 0, fn = 0;

    for (int i=0; i<bs_test; i++){
        float y_hat = seq.layers.back()->out[i];
        if ((y_hat > 0.5) && (targ_test[i] > 0.5)) tp++;        // true positive
        else if ((y_hat < 0.5) && (targ_test[i] < 0.5)) tn++;   // true negative
        else if ((y_hat > 0.5) && (targ_test[i] < 0.5)) fp++;   // false positive
        else if ((y_hat < 0.5) && (targ_test[i] > 0.5)) fn++;   // false negative
        // std::cout << "out " << i << ": "<< seq.layers.back()->out[i] << std::endl;
        // std::cout << "targ " << i << ": "<< targ_test[i] << std::endl;
    }

    std::cout << "TP: "<< tp << std::endl;
    std::cout << "TN: "<< tn << std::endl;
    std::cout << "FP: "<< fp << std::endl;
    std::cout << "FN: "<< fn << std::endl;
    std::cout << "akurasi: "<< (float(tp) + float(tn))/(bs_test) << " %" << std::endl;

    float rec = float(tp)/(float(tp) + float(fn)), prec = float(tp)/(float(tp) + float(fp));
    std::cout << "recall: "<< rec << " %" << std::endl;
    std::cout << "precision: "<< prec << " %" << std::endl;
    std::cout << "f-measure: "<< (2 * prec * rec)/(prec + rec) << " %" << std::endl;

    // mse._forward(seq.layers.back()->out, targ_test);
    // std::cout << "The final loss is: " << targ_test[bs_test] << std::endl;

}
