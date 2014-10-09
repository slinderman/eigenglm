# distutils: language = c++
# distutils: sources = npglm2.cpp
# distutils: libraries = stdc++
# distutils: extra_compile_args = -std=c++11

from libcpp.vector cimport vector
from cython cimport floating

cdef extern from "npglm2.h":
    cdef cppclass SpikeTrain[Type]:
        SpikeTrain()
        SpikeTrain(int, int, int, double, Type*, vector[Type*]) except +

cdef class PySpikeTrain:
    cdef SpikeTrain[double] *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, int T, int N, int D_imp, double dt, double[::1] S, list filtered_S):

        cdef vector[double*] filtered_S_vector
        cdef double[:,::1] temp
        for n in range(N):
            temp = filtered_S[n]
            filtered_S_vector.push_back(&temp[0,0])

        self.thisptr = new SpikeTrain[double](T, N, D_imp, dt, &S[0], filtered_S_vector)

    def __dealloc__(self):
        del self.thisptr
