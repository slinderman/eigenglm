# distutils: language = c++
# distutils: sources = glm.cpp
# distutils: libraries = stdc++
# distutils: extra_compile_args = -std=c++11

from libcpp.vector cimport vector

cdef extern from "npglm.h":
    cdef cppclass NpSpikeTrain:
        NpSpikeTrain(int, int, double, double*, int, vector[double*]) except +

    cdef cppclass NpBiasCurrent:
        NpBiasCurrent(double) except +
        double log_probability()
        void resample()

    cdef cppclass NpGlm:
        NpGlm() except +
        void add_spike_train(NpSpikeTrain *s)

        
cdef class PyNpSpikeTrain:
    cdef NpSpikeTrain *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, int N, int T, double dt, double[::1] S, int D_imp, list filtered_S):

        # Cast the list to a cpp vector
        cdef vector[double*] filtered_S_vect
        cdef double[:,::1] temp
        cdef int n
        for n in range(N):
            temp = filtered_S[n]
            filtered_S_vect.push_back(&temp[0,0])

        self.thisptr = new NpSpikeTrain( N, T, dt, &S[0], D_imp, filtered_S_vect)

    def __dealloc__(self):
        del self.thisptr

cdef class PyNpBiasCurrent:
    cdef NpBiasCurrent *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, double bias):
        self.thisptr = new NpBiasCurrent(bias)
    def __dealloc__(self):
        del self.thisptr

    def log_probability(self):
        return self.thisptr.log_probability()

    def resample(self):
        self.thisptr.resample()


cdef class PyNpGlm:
    cdef NpGlm *thisptr

    def __cinit__(self):
        self.thisptr = new NpGlm()

    def __dealloc__(self):
        del self.thisptr

    def add_spike_train(self, PyNpSpikeTrain st):
        self.thisptr.add_spike_train(st.thisptr)