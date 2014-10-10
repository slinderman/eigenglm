# distutils: language = c++
# distutils: sources = glm.cpp
# distutils: libraries = stdc++
# distutils: extra_compile_args = -std=c++11

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector

cdef extern from "glm.h":
    cdef cppclass SpikeTrain:
        SpikeTrain(int, int, double, double*, int, vector[double*]) except +

    cdef cppclass Glm:
        Glm(int, int) except +
        void add_spike_train(SpikeTrain *s)
        double log_likelihood()
        double log_probability()
        void coord_descent_step(double momentum)

        void get_firing_rate(SpikeTrain *s, double* fr)

        
cdef class PySpikeTrain:
    cdef SpikeTrain *thisptr      # hold a C++ instance which we're wrapping
    cdef int T
    cdef int N
    cdef double dt
    cdef double[::1] S
    cdef int D_imp
    cdef list filtered_S

    def __cinit__(self, int N, int T, double dt, double[::1] S, int D_imp, list filtered_S):
        # Store the values locally
        self.T = T
        self.N = N
        self.dt = dt
        self.S = S
        self.D_imp = D_imp
        self.filtered_S = filtered_S

        # Cast the list to a cpp vector
        cdef vector[double*] filtered_S_vect
        cdef double[:,::1] temp
        cdef int n
        for n in range(N):
            temp = filtered_S[n]
            filtered_S_vect.push_back(&temp[0,0])

        self.thisptr = new SpikeTrain( N, T, dt, &S[0], D_imp, filtered_S_vect)

    def __dealloc__(self):
        del self.thisptr

cdef class PyGlm:
    cdef Glm *thisptr

    def __cinit__(self, int N, int D_imp):
        self.thisptr = new Glm(N, D_imp)

    def __dealloc__(self):
        del self.thisptr

    def add_spike_train(self, PySpikeTrain st):
        self.thisptr.add_spike_train(st.thisptr)

    def log_likelihood(self):
        return self.thisptr.log_likelihood()

    def log_probability(self):
        return self.thisptr.log_probability()

    def coord_descent_step(self, double momentum):
        self.thisptr.coord_descent_step(momentum)

    def get_firing_rate(self, PySpikeTrain st):
        cdef double[::1] fr = np.zeros(st.T, dtype=np.double)
        self.thisptr.get_firing_rate(st.thisptr, &fr[0])
        return fr