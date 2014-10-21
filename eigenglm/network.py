import numpy as np

import eigenglm.deps.pybasicbayes.distributions as pbd
from eigenglm.parameters import GaussianErdosRenyiNetworkHyperParameters

class Network(object):

    def __init__(self, N, population, prms):
        self.N = N
        self.population = population
        self.prms = prms

    def resample(self):
        raise NotImplementedError()

class ConstantNetwork(Network):
    def resample(self):
        pass

class GaussianErdosRenyiNetwork(Network):
    def __init__(self, N, population, prms):
        assert isinstance(prms, GaussianErdosRenyiNetworkHyperParameters)
        super(GaussianErdosRenyiNetwork, self).__init__(N, population, prms)

        # TODO: Create a beta prior over the sparsity
        self.rho_synaptic = prms.A_synaptic_rho
        self.rho_refractory = prms.A_refractory_rho

        # Create the shared priors for the "synaptic" and "refractory" weights
        self.W_synaptic_hyperprior = prms.cls(**prms.W_synaptic_prms)
        self.W_refractory_hyperprior = prms.cls(**prms.W_refractory_prms)

        # Create masks to extract "synaptic" and "refractory" weights
        self.syn_mask = np.ones((N,N), dtype=np.bool)
        self.syn_mask[np.diag_indices(N)] = False
        self.refractory_mask = np.bitwise_not(self.syn_mask)

    def W_mu(self, n):
        # Return the mean of the n-th column of W
        mu = self.W_synaptic_hyperprior.mu * np.ones(self.N)
        # Overwrite the refractory prior
        mu[n] = self.W_refractory_hyperprior.mu
        return mu

    def W_sigma(self, n):
        # Return the std dev of the n-th column of W
        sigma = np.sqrt(self.W_synaptic_hyperprior.sigmasq) * np.ones(self.N)
        # Overwrite the refractory prior
        sigma[n] = np.sqrt(self.W_refractory_hyperprior.sigmasq)
        return sigma

    def A_rho(self, n):
        rho = self.rho_synaptic * np.ones(self.N)
        rho[n] = self.rho_refractory
        return rho

    def resample(self):
        """
        Resample the parameters of the prior
        """
        # Get the current network
        A = self.population.A
        W = self.population.W

        # Resample the synaptic weight prior
        W_synaptic = W[A & self.syn_mask]
        self.W_synaptic_hyperprior.resample(W_synaptic)

        # Resample the refractory weight prior
        W_refractory = W[A & self.refractory_mask]
        self.W_refractory_hyperprior.resample(W_refractory)

        # Update the priors for each GLM
        for n, glm in enumerate(self.population.glms):
            mu_n = self.W_mu(n)
            sigma_n = self.W_sigma(n)

            # TODO: Should we assume the GLM has a "W_prior" attribute?
            glm.W_prior.set_mu(mu_n)
            glm.W_prior.set_sigma(sigma_n)