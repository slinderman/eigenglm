# distutils: language = c++
# distutils: libraries = stdc++
# distutils: extra_compile_args = -std=c++11

from libcpp.vector cimport vector
from cython cimport floating

cdef extern from "npglm3.h":
    cdef cppclass NPGlm[Type]:
        NPGlm()
        Type log_likelihood(int T, Type dt, Type* S, Type I_bias, Type* I_stim, Type* I_net)

        void compute_I_net(int T, int D_imp, Type* ir, Type* w_ir, Type* I_net)

        void compute_all_I_net(int T, int N, int D_imp, vector[Type*] irs, vector[Type*] w_irs, vector[Type*] I_nets)

# Expose the NPGlm methods
def log_likelihood(int T,
                 floating dt,
                 floating[::1] S,
                 floating I_bias,
                 floating[::1] I_stim,
                 floating[::1] I_net):

    # Create a reference to the dummy class.
    cdef NPGlm[floating] ref
    return ref.log_likelihood(T, dt, &S[0], I_bias, &I_stim[0], &I_net[0]);

def compute_I_net(int T,
                  int D_imp,
                  floating[:,::1] ir,
                  floating[::1] w_ir,
                  floating[::1] I_net):

    # Create a reference to the dummy class.
    cdef NPGlm[floating] ref
    ref.compute_I_net(T, D_imp, &ir[0,0], &w_ir[0], &I_net[0])

def compute_all_I_net(int T,
                      int N,
                      int D_imp,
                      list irs not None,
                      list w_irs not None,
                      list I_nets not None):

    # Create a reference to the dummy class.
    cdef NPGlm[double] ref

    # Create vectors of inputs
    cdef vector[double*] irs_vect
    cdef vector[double*] w_irs_vect
    cdef vector[double*] I_nets_vect

    cdef double[:,:] temp_ir
    cdef double[:] temp_w_ir
    cdef double[:] temp_I_net


    cdef int n
    for n in range(N):
        temp_ir = irs[n]
        irs_vect.push_back(&temp_ir[0,0])
        temp_w_ir = w_irs[n]
        w_irs_vect.push_back(&temp_w_ir[0])
        temp_I_net = I_nets[n]
        I_nets_vect.push_back(&temp_I_net[0])


    ref.compute_all_I_net(T, N, D_imp, irs_vect, w_irs_vect, I_nets_vect)

