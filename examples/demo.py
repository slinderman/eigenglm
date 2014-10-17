import numpy as np
import matplotlib.pyplot as plt

from eigenglm import StandardGLM, StandardGLMParameters

# Make fake data
def create_test_data(N, T, dt=1.0):
    # Create a fake spike trains
    S = np.random.randint(0,3,(T,N)).astype(np.double)

    return {'N' : N,
            'T' : T,
            'dt' : dt,
            'S' : S}

def run():
    # Specify the number of neurons in the population
    N = 1

    # Make a parameters object that we can modify
    prms = StandardGLMParameters()
    # E.g. change the number of basis elements
    prms.impulse.basis.n_bas = 5

    # Make the GLM object
    glm = StandardGLM(0, N, prms)

    # Make some fake data
    T = 600
    M = 1
    for m in range(M):
        glm.add_data(create_test_data(N, T))

    # Run some MCMC
    N_iters = 1000
    for i in range(N_iters):
        glm.resample()
        ll = glm.log_likelihood()

        if i % 25 == 0:
            print "Iteration ", i, ":\tLL: ", ll

run()