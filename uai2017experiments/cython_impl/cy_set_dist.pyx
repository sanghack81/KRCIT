cimport numpy as np
import cython
import numpy as np

cdef extern from "c_cy_set_dist.h":
    void rpci(double *values, int *items, int *lengths, double *vertex_kernel, double *output, int n, int m,
              int n_threads, double gamma, int algorithm, int equal_size_only)



@cython.boundscheck(False)
@cython.wraparound(False)
def cy_rpci(np.ndarray[double, ndim=1, mode="c"] values not None,
            np.ndarray[int, ndim=1, mode="c"]  items not None,
            np.ndarray[int, ndim=1, mode="c"]  lengths not None,
            np.ndarray[double, ndim=2, mode="c"]  vertex_kernel not None,
            np.ndarray[double, ndim=2, mode="c"]  output not None,
            int n_jobs, double gamma, int algorithm, int equal_size_only):
    cdef int n, m
    n = len(lengths)
    m = vertex_kernel.shape[0]
    rpci(&values[0], &items[0], &lengths[0], &vertex_kernel[0, 0],
         &output[0, 0], n, m, n_jobs, gamma, algorithm, equal_size_only)
