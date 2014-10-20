import numpy as np

from eigenglm.glm import StandardGLM, NormalizedGLM

class Population(object):
    """
    Population of connected GLMs.
    """
    def __init__(self, N, prms):
        """
        Initialize the population of GLMs connected by a network.
        """
        self.N = N
        self.prms = prms

        # Initialize a list of data sequences
        self.data_sequences = []

        # TODO: Initialize latent variables of the population
        self._latent = None

        # TODO: Create a network model to connect the GLMs
        self._network = None

        # Create a list of glms
        self._glms = []

    @property
    def latent(self):
        return self._latent

    @property
    def network(self):
        return self._network

    @property
    def glms(self):
        return self._glms

    def add_data(self, data):
        """
        Add another data sequence to the population. Recursively call components
        to prepare the new data sequence. E.g. the background model may preprocess
        the stimulus with a set of basis filters.
        """
        for glm in self.glms:
            glm.add_data(data)

    def log_prior(self):
        """ Compute the log joint probability under a given set of variables
        """
        lp = 0.0
        # lp += self.latent.log_probability()
        # lp += self.network.log_probability()
        for glm in self.glms:
            lp += glm.log_prior()

        return lp

    def log_likelihood(self):
        """ Compute the log likelihood under a given set of variables
        """
        ll = 0.0
        # Add the likelihood from each GLM
        for glm in self.glms:
            ll += glm.log_likelihood()

        return ll

    def log_probability(self):
        """ Compute the log joint probability under a given set of variables
        """
        lp = 0.0
        lp += self.log_prior()

        # Add the likelihood of each data sequence
        lp += self.log_likelihood()

        return lp

    def simulate(self, T_stop, dt, T_start=0, stim=None, dt_stim=None,
                 nlin=lambda x: np.log(1+np.exp(x))):
        """ Simulate spikes from a network of coupled GLMs
        :param vars - the variables corresponding to each GLM
        :type vars    list of N variable vectors
        :param dt    - time steps to simulate

        :rtype TxN matrix of spike counts in each bin
        """
        # Initialize the background rates
        N = self.N
        t = np.arange(T_start, T_stop, dt)
        t_ind = np.arange(int(T_start/dt), int(T_stop/dt))
        assert len(t) == len(t_ind)
        T = len(t)

        # Initialize the background rate
        X = np.zeros((T,N))
        for n in np.arange(N):
            X[:,n] = self.glms[n].bias

        # Add stimulus induced currents if given
        # temp_data = {'S' : np.zeros((nT, N)),
        #              'stim' : stim,
        #              'dt_stim': dt_stim}
        # self.add_data(temp_data)
        # for n in np.arange(N):
        #     X[:,n] += seval(self.glm.bkgd_model.I_stim,
        #                     syms,
        #                     nvars)
        # print "Max background rate: %s" % str(self.glm.nlin_model.f_nlin(np.amax(X)))

        # Get the impulse response functions
        imps = []
        for n_post in np.arange(N):
            imps.append(self.glms[n_post].impulse_response(dt, self.prms.glms[n_post].impulse.dt_max))
        imps = np.transpose(np.array(imps), axes=[1,0,2])
        T_imp = imps.shape[2]

        # Debug: compute effective weights
        # tt_imp = dt*np.arange(T_imp)
        # Weff = np.trapz(imps, tt_imp, axis=2)
        # print Weff

        # Iterate over each time step and generate spikes
        S = np.zeros((T,N))
        acc = np.zeros(N)
        thr = -np.log(np.random.rand(N))

        At = np.tile(self.A[:,:,None], [1,1,T_imp])
        Wt = np.tile(self.W[:,:,None], [1,1,T_imp])

        # Count the number of exceptions arising from more spikes per bin than allowable
        n_exceptions = 0
        for t in np.arange(T):
            # Update accumulator
            if np.mod(t,10000)==0:
                print "Iteration %d" % t

            lam = nlin(X[t,:])
            acc = acc + lam*dt

            # Spike if accumulator exceeds threshold
            i_spk = acc > thr
            S[t,i_spk] += 1
            n_spk = np.sum(i_spk)

            # Compute the length of the impulse response
            t_imp = np.minimum(T-t-1,T_imp)

            # Iterate until no more spikes
            # Cap the number of spikes in a time bin
            max_spks_per_bin = 10
            while n_spk > 0:
                if np.any(S[t,:] >= max_spks_per_bin):
                    n_exceptions += 1
                    break
                # Add weighted impulse response to activation of other neurons)
                X[t+1:t+t_imp+1,:] += np.sum(At[i_spk,:,:t_imp] *
                                             Wt[i_spk,:,:t_imp] *
                                             imps[i_spk,:,:t_imp],0).T

                # Subtract threshold from the accumulator
                acc -= thr*i_spk
                acc[acc<0] = 0

                # Set new threshold after spike
                thr[i_spk] = -np.log(np.random.rand(n_spk))

                i_spk = acc > thr
                S[t,i_spk] += 1
                n_spk = np.sum(i_spk)

                #if np.any(S[t,:]>10):
                #    import pdb
                #    pdb.set_trace()
                #    raise Exception("More than 10 spikes in a bin! Decrease variance on impulse weights or decrease simulation bin width.")
                
        # DEBUG:
        tt = dt * np.arange(T)
        lam = np.zeros_like(X)
        for n in np.arange(N):
            lam[:,n] = nlin(X[:,n])
            
        print "Max firing rate (post sim): %f" % np.max(lam)
        E_nS = np.trapz(lam,tt,axis=0)
        nS = np.sum(S,0)

        print "Sampled %s spikes." % str(nS)
        print "Expected %s spikes." % str(E_nS)

        if np.any(np.abs(nS-E_nS) > 3*np.sqrt(E_nS)):
            print "ERROR: Actual num spikes (%s) differs from expected (%s) by >3 std." % (str(nS),str(E_nS))

        print "Number of exceptions arising from multiple spikes per bin: %d" % n_exceptions

        # Package the data
        data = { 'S' : S,
                 'T' : T,
                 'N' : N,
                 'dt' : dt,
               }

        return data

    def resample(self):
        """
        Resample the latent variables, the network, and the GLMs
        :return:
        """
        for glm in self.glms:
            glm.resample()

    # Properties
    @property
    def A(self):
        A = np.zeros((self.N, self.N))
        for n,glm in enumerate(self.glms):
            A[:,n] = glm.An
        return A

    @property
    def W(self):
        W = np.zeros((self.N, self.N))
        for n,glm in enumerate(self.glms):
            W[:,n] = glm.Wn
        return W

class StandardGLMPopulation(Population):
    def __init__(self, N, prms):
        super(StandardGLMPopulation, self).__init__(N, prms)

        # Initialize the GLMs
        for n in range(N):
            self.glms.append(StandardGLM(n, N, prms.glms[n]))

class NormalizedGLMPopulation(Population):
    def __init__(self, N, prms):
        super(NormalizedGLMPopulation, self).__init__(N, prms)

        # Initialize the GLMs
        for n in range(N):
            self.glms.append(NormalizedGLM(n, N, prms.glms[n]))
