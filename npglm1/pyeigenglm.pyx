# distutils: language = c++
# distutils: sources = glm.cpp
# distutils: libraries = stdc++
# distutils: extra_compile_args = -std=c++11

from libcpp.vector cimport vector

cdef extern from "npglm.h":
    cdef cppclass NpSpikeTrain:
        NpSpikeTrain(int, int, double, double*, int, vector[double*]) except +

#    cdef cppclass NpBiasCurrent:
#        NpBiasCurrent(double) except +
#        double log_probability()
#        void resample()
#
    cdef cppclass NpGlm:
        NpGlm(int, int) except +
        void add_spike_train(NpSpikeTrain *s)
        void firing_rate(NpSpikeTrain *s, double* fr)
        double log_likelihood()
        double log_probability()
        void coord_descent_step(double momentum)

        double get_bias()

        
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

#cdef class PyNpBiasCurrent:
#    cdef NpBiasCurrent *thisptr      # hold a C++ instance which we're wrapping
#    def __cinit__(self, double bias):
#        self.thisptr = new NpBiasCurrent(bias)
#    def __dealloc__(self):
#        del self.thisptr
#
#    def log_probability(self):
#        return self.thisptr.log_probability()
#
#    def resample(self):
#        self.thisptr.resample()
#

cdef class PyNpGlm:
    cdef NpGlm *thisptr

    def __cinit__(self, int N, int D_imp):
        self.thisptr = new NpGlm(N, D_imp)

    def __dealloc__(self):
        del self.thisptr

    def add_spike_train(self, PyNpSpikeTrain st):
        self.thisptr.add_spike_train(st.thisptr)

    def firing_rate(self, PyNpSpikeTrain st, double[::1] fr):
        self.thisptr.firing_rate(st.thisptr, &fr[0])

    def log_likelihood(self):
        return self.thisptr.log_likelihood()

    def log_probability(self):
        return self.thisptr.log_probability()

    def coord_descent_step(self, double momentum):
        self.thisptr.coord_descent_step(momentum)

    property bias:
        def __get__(self): return self.thisptr.get_bias()