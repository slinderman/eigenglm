# distutils: language = c++
# distutils: sources = eigenglm/cpp/eigenglm.cpp eigenglm/cpp/impulse.cpp
# distutils: libraries = stdc++
# distutils: extra_compile_args = -std=c++11 -O3

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector

# Import C++ distributions
from pydistributions cimport DiagonalGaussian

# Import C++ classes from eigenglm.h
cdef extern from "eigenglm.h":
    # Spike train class encapsulates the observed datasets
    cdef cppclass SpikeTrain:
        SpikeTrain(int, int, double, double*, int, vector[double*]) except +

    # Bias class for the constant activation bias
    cdef cppclass BiasCurrent:
        double get_bias()

    # Linear impulse response class
    cdef cppclass LinearImpulseCurrent:
        void d_ll_d_w(SpikeTrain* st, int n, double* dw_buffer)
        void get_w(double* w_buffer)
        void set_w(double* w_buffer)

    # Dirichlet impulse response class
    cdef cppclass DirichletImpulseCurrent:
        void d_ll_d_g(SpikeTrain* st, int n, double* dg_buffer)
        void get_w(double* w_buffer)
        void get_g(double* g_buffer)
        void set_g(double* g_buffer)

    # Gaussian network class
    cdef cppclass GaussianNetworkColumn:
        void get_A(double* A_buffer)
        void get_W(double* W_buffer)
        void set_A(int n_pre, double a)
        void set_W(int n_pre, double w)


    # Main GLM class
    cdef cppclass StandardGlm:
        StandardGlm(int, int, int) except +
        void add_spike_train(SpikeTrain *s)

        # Getters
        BiasCurrent* get_bias_component()
        LinearImpulseCurrent* get_impulse_component()
        double log_prior()
        double log_likelihood()
        double log_probability()
        void get_firing_rate(SpikeTrain *s, double* fr)

        # Inference
        void coord_descent_step(double momentum)
        void resample()

    # Normalized GLM class
    cdef cppclass NormalizedGlm:
        NormalizedGlm(int, int, int) except +
        void add_spike_train(SpikeTrain *s)

        # Getters
        BiasCurrent* get_bias_component()
        DirichletImpulseCurrent* get_impulse_component()
        GaussianNetworkColumn* get_network_component()
        double log_prior()
        double log_likelihood()
        double log_probability()
        void get_firing_rate(SpikeTrain *s, double* fr)

        # Inference
        void coord_descent_step(double momentum)
        void resample()


# Expose the SpikeTrain class to Python
cdef class PySpikeTrain:
    cdef SpikeTrain *thisptr

    # Also save the parameters for easy access from Python
    cdef public int T
    cdef public int N
    cdef public double dt
    cdef public double[::1] S
    cdef public int D_imp
    cdef public list filtered_S

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

# Expose the GLM class to Python
cdef class PyStandardGlm:
    cdef StandardGlm *thisptr
    cdef public int n
    cdef public int N
    cdef public int D_imp

    def __cinit__(self, int n, int N, int D_imp):
        self.thisptr = new StandardGlm(n, N, D_imp)
        self.n = n
        self.N = N
        self.D_imp = D_imp

    def __dealloc__(self):
        del self.thisptr

    def add_spike_train(self, PySpikeTrain st):
        self.thisptr.add_spike_train(st.thisptr)

    def get_bias(self):
        return self.thisptr.get_bias_component().get_bias()

    def get_w_ir(self):
        cdef double[:,::1] w = np.zeros((self.N, self.D_imp))
        self.thisptr.get_impulse_component().get_w(&w[0,0])
        return np.asarray(w).reshape((self.N, self.D_imp))

    def set_w_ir(self, double[:,::1] w):
        assert w.shape[0] == self.N and w.shape[1] == self.D_imp, "w is not the correct shape!"
        self.thisptr.get_impulse_component().set_w(&w[0,0])

    def get_dll_dw(self, PySpikeTrain st, int n):
        cdef double[::1] dw = np.zeros(self.D_imp)
        self.thisptr.get_impulse_component().d_ll_d_w(st.thisptr, n, &dw[0])
        return np.asarray(dw)

    def log_prior(self):
        return self.thisptr.log_prior()

    def log_likelihood(self):
        return self.thisptr.log_likelihood()

    def log_probability(self):
        return self.thisptr.log_probability()

    def get_firing_rate(self, PySpikeTrain st):
        cdef double[::1] fr = np.zeros(st.T, dtype=np.double)
        self.thisptr.get_firing_rate(st.thisptr, &fr[0])
        return fr

    def coord_descent_step(self, double momentum):
        self.thisptr.coord_descent_step(momentum)

    def resample(self):
        self.thisptr.resample()


# Expose the GLM class to Python
cdef class PyNormalizedGlm:
    cdef NormalizedGlm *thisptr
    cdef public int n
    cdef public int N
    cdef public int D_imp

    def __cinit__(self, int n, int N, int D_imp):
        self.thisptr = new NormalizedGlm(n, N, D_imp)
        self.n = n
        self.N = N
        self.D_imp = D_imp

    def __dealloc__(self):
        del self.thisptr

    def add_spike_train(self, PySpikeTrain st):
        self.thisptr.add_spike_train(st.thisptr)

    def get_bias(self):
        return self.thisptr.get_bias_component().get_bias()

    def get_w_ir(self):
        cdef double[:,::1] w = np.zeros((self.N, self.D_imp))
        self.thisptr.get_impulse_component().get_w(&w[0,0])
        return np.asarray(w).reshape((self.N, self.D_imp))

    def get_g_ir(self):
        cdef double[:,::1] g = np.zeros((self.N, self.D_imp))
        self.thisptr.get_impulse_component().get_g(&g[0,0])
        return np.asarray(g).reshape((self.N, self.D_imp))

    def set_g_ir(self, double[:,::1] g):
        assert g.shape[0] == self.N and g.shape[1] == self.D_imp, "w is not the correct shape!"
        self.thisptr.get_impulse_component().set_g(&g[0,0])

    def get_A(self):
        cdef double[::1] A = np.zeros(self.N)
        self.thisptr.get_network_component().get_A(&A[0])
        return np.asarray(A)

    def get_W(self):
        cdef double[::1] W = np.zeros(self.N)
        self.thisptr.get_network_component().get_W(&W[0])
        return np.asarray(W)

    def set_A(self, int n_pre, double a):
        self.thisptr.get_network_component().set_A(n_pre, a)

    def set_W(self, int n_pre, double w):
        self.thisptr.get_network_component().set_W(n_pre, w)

    def get_dll_dg(self, PySpikeTrain st, int n):
        cdef double[::1] dg = np.zeros(self.D_imp)
        self.thisptr.get_impulse_component().d_ll_d_g(st.thisptr, n, &dg[0])
        return np.asarray(dg)

    def log_prior(self):
        return self.thisptr.log_prior()

    def log_likelihood(self):
        return self.thisptr.log_likelihood()

    def log_probability(self):
        return self.thisptr.log_probability()

    def get_firing_rate(self, PySpikeTrain st):
        cdef double[::1] fr = np.zeros(st.T, dtype=np.double)
        self.thisptr.get_firing_rate(st.thisptr, &fr[0])
        return fr

    def coord_descent_step(self, double momentum):
        self.thisptr.coord_descent_step(momentum)

    def resample(self):
        self.thisptr.resample()
