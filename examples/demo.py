import time
import numpy as np

# Set the random seed for reproducability
np.random.seed(0)

import matplotlib.pyplot as plt

from eigenglm import StandardGLMPopulation, StandardGLM, StandardGLMParameters, StandardGLMPopulationParameters
from eigenglm import NormalizedGLMPopulation, NormalizedGLMParameters, NormalizedGLMPopulationParameters

# Make fake data
def create_test_data(N, T, dt=0.001):
    # Create a fake spike trains
    S = np.random.randint(0,3,(T,N)).astype(np.double)
    return {'N' : N,
            'T' : T,
            'dt' : dt,
            'S' : S}

def run():
    # Specify the number of neurons in the population
    N = 2

    # Make a parameters object that we can modify
    # prms = StandardGLMParameters()
    # prms = NormalizedGLMParameters()
    # prms = StandardGLMPopulationParameters(N)
    prms = NormalizedGLMPopulationParameters(N)
    # E.g. change the number of basis elements
    prms.glms[0].impulse.basis.n_bas = 5
    prms.glms[1].impulse.basis.n_bas = 5

    # Make the GLM object
    # glm = StandardGLM(0, N, prms)
    # glm = NormalizedGLM(0, N, prms)
    # population = StandardGLMPopulation(N, prms)
    population = NormalizedGLMPopulation(N, prms)

    print "A: ", population.A
    print "W: ", population.W

    # Simulate some data
    T = 60
    dt = 0.001
    data = population.simulate(T, dt)
    population.add_data(data)
    # Make some fake data
    # T = 60000
    # M = 1
    # for m in range(M):
    #     glm.add_data(create_test_data(N, T))

    # Run some MCMC
    N_iters = 1000
    print "Running ", N_iters, " iterations of MCMC."
    intvl = 25
    start = time.time()
    for i in range(N_iters):
        population.resample()
        ll = population.log_likelihood()
        lp = population.log_probability()

        if i % intvl == 0:
            stop = time.time()
            print "Iteration ", i, ":\tLP: %.3f" % lp, "\tIters/sec: %.3f" % (intvl/(stop-start))
            start = stop

run()