#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <thread>
#include <random>
#include <cmath>
#include <vector>
#include <iostream>



double
r_convolution(double *v1s, double *v2s, int vlen1, int vlen2, int *item1s, int *item2s, double *vertex_kernel, int m,
              double gamma, int equal_size_only) {
    if (equal_size_only > 0 && vlen1 != vlen2) {
        return 0.0;
    }
    int i, j;
    double v = 0.0;
    for (i = 0; i < vlen1; i++) {
        for (j = 0; j < vlen2; j++) {
            v += vertex_kernel[item1s[i] * m + item2s[j]] * exp(-gamma * pow(v1s[i] - v2s[j], 2.0));
        }
    }
    return v;
}



void sub_rpci(double *values, int *items, int *lengths, double *vertex_kernel,
              double *output,
              int n,
              int m, int divide, int remnant, double gamma, int algorithm, int equal_size_only) {
//    cout << "";
    double *value_i = values;
    int *items_i = items;

    for (int i = 0; i < n; i++) {
        double *value_j = value_i;
        int *items_j = items_i;
        for (int j = i; j < n; j++) {
            if ((i + j) % divide == remnant) { // thread-based computation
                if (algorithm == 0) {
                    output[j * n + i] = output[i * n + j] = r_convolution(value_i, value_j,
                                                                          lengths[i], lengths[j],
                                                                          items_i, items_j,
                                                                          vertex_kernel, m, gamma, equal_size_only);
                } else if (algorithm == 1) {
                    output[j * n + i] = output[i * n + j] = 0.0;    // algorithm 1 is removed
                } else {
                    output[j * n + i] = output[i * n + j] = 0.0;
                }

            }

            value_j += lengths[j];
            items_j += lengths[j];
        }
        value_i += lengths[i];
        items_i += lengths[i];
    }
}

void rpci(double *values, int *items, int *lengths, double *vertex_kernel,
          double *output,
          int n,
          int m, int n_threads, double gamma, int algorithm, int equal_size_only) {
    //
    // values: unrolled values, [pair[1] for tuple in data for pair in tuple]
    // items: unrolled items    [pair[0] for tuple in data for pair in tuple]
    // lengths:                 [len(tuple) for tuple in data]
    // vertex_kernel: unrolled vertex kernel matrix, of the shape (m,m)
    // output: output kernel matrix, must be of the shape, (n, n)
    // n: number of instances of data,
    // m,
    if (n_threads == 1) {
        sub_rpci(values, items, lengths, vertex_kernel, output, n, m, n_threads, 0, gamma, algorithm, equal_size_only);
        std::cout << ""; // TODO strange behavior? output unflushed?
    } else {
        std::vector<std::thread> threads;
        for (int i = 0; i < n_threads; i++) {
            threads.push_back(std::thread(sub_rpci, values, items, lengths, vertex_kernel,
                                          output, n, m, n_threads, i, gamma, algorithm, equal_size_only));
        }
        // joint threads,
        for (int i = 0; i < n_threads; i++) {
            threads[i].join();
        }
        std::cout << ""; // TODO strange behavior?
    }
}
