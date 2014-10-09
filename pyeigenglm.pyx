# distutils: language = c++
# distutils: sources = glm.cpp
# distutils: libraries = stdc++
# distutils: extra_compile_args = -std=c++11

cdef extern from "npglm.h":
    cdef cppclass NpSpikeTrain:
        NpSpikeTrain(int, int, double, double*) except +

    cdef cppclass NpBiasCurrent:
        NpBiasCurrent(double) except +
        double log_probability()
        void resample()

        
cdef class PyNpSpikeTrain:
    cdef NpSpikeTrain *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, int T, int N, double dt, double[::1] S):
        self.thisptr = new NpSpikeTrain(T, N, dt, &S[0])
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
