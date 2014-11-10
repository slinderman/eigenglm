"""
Implement a distribution class for negative binomial regression
"""
import numpy as np
from numpy import newaxis as na
from scipy.special import gammaln

from deps.pybasicbayes.abstractions import Collapsed
from deps.pybasicbayes.distributions import Regression, GibbsSampling, GaussianFixed
from deps.pybasicbayes.util.general import blockarray

from hips.distributions.polya_gamma import polya_gamma

class AugmentedNegativeBinomialCounts(GibbsSampling):
    """
    Class to keep track of a set of counts and the corresponding Polya-gamma
    auxiliary variables associated with them.
    """
    def __init__(self, X, counts, nbmodel):
        # Data must be a T x (D_in + 1) matrix where the last column contains the counts
        assert counts.ndim == 1
        self.counts = counts
        self.T = counts.shape[0]

        # assert X.ndim == 2 and X.shape[0] == self.T
        self.X = X

        # Keep this pointer to the model
        self.model = nbmodel

        # Initialize auxiliary variables
        self.omegas = np.ones(self.T)
        self.psi = np.zeros(self.T)

    def log_likelihood(self, x):
        return 0

    def rvs(self,size=[]):
        return None

    def resample(self, data=None, stats=None):
        """
        Resample omega given xi and psi, then resample psi given omega, X, w, and sigma
        """
        xi = self.model.xi
        mu = self.model.mu_psi(self.X)
        sigma = self.model.sigma
        trunc = self.model.trunc

        # import pdb; pdb.set_trace()
        sigma = np.asscalar(sigma)
        # Resample the auxiliary variables, omega
        self.omega = polya_gamma(self.counts+xi, self.psi, trunc)

        # Resample the rates, psi given omega and the regression parameters
        sig_post = 1.0 / (1.0/sigma + self.omega)
        mu_post = sig_post * ((self.counts-xi)/2.0 + mu / sigma)
        self.psi = mu_post + np.sqrt(sig_post) * np.random.normal(size=(self.T,))

class RegressionFixedCov(GibbsSampling, Collapsed):
    def __init__(self,
                 sigma,
                 mu_A=None, Sigma_A=None,
                 affine=False,
                 A=None):

        self.affine = affine
        self.sigma = sigma
        self.A = A

        self.mu_A = mu_A
        self.Sigma_A = Sigma_A

        if Sigma_A is not None:
            self.Sigma_A_inv = np.linalg.inv(Sigma_A)
        else:
            self.Sigma_A_inv = None

        if A is None and not any(_ is None for _ in (mu_A, Sigma_A)):
            assert mu_A.ndim == 2
            self.resample() # initialize from prior

    @property
    def D_in(self):
        # NOTE: D_in includes the extra affine coordinate
        mat = self.A if self.A is not None else self.mu_A
        return mat.shape[1]

    @property
    def D_out(self):
        # For now, assume we are doing scalar regression
        return 1

    ### getting statistics

    def _get_statistics(self,data):
        if isinstance(data,list):
            return sum((self._get_statistics(d) for d in data),self._empty_statistics())
        else:
            data = data[~np.isnan(data).any(1)]
            n, D = data.shape[0], self.D_out

            statmat = data.T.dot(data)
            xxT, yxT, yyT = statmat[:-D,:-D], statmat[-D:,:-D], statmat[-D:,-D:]

            if self.affine:
                xy = data.sum(0)
                x, y = xy[:-D], xy[-D:]
                xxT = blockarray([[xxT,x[:,na]],[x[na,:],np.atleast_2d(n)]])
                yxT = np.hstack((yxT,y[:,na]))

            return np.array([yyT, yxT, xxT, n])

    def _empty_statistics(self):
        return np.array([np.zeros((self.D_out, self.D_out)),
                         np.zeros((1,self.D_in)),
                         np.zeros((self.D_in, self.D_in)),
                         0])

    ### distribution

    def log_likelihood(self,xy):
        A, sigma, D = self.A, self.sigma, self.D_out
        x, y = xy[:,:-D], xy[:,-D:]

        if self.affine:
            A, b = A[:,:-1], A[:,-1]
            mu_y = x.dot(A.T) + b
        else:
            mu_y = x.dot(A.T)

        ll = (-0.5/sigma * (y-mu_y)**2).sum()
        ll -= -0.5 * np.log(2*np.pi * sigma**2)

        return ll

    def rvs(self,x=None,size=1,return_xy=True):
        A, sigma = self.A, self.sigma

        if self.affine:
            A, b = A[:,:-1], A[:,-1]

        x = np.random.normal(size=(size,A.shape[1])) if x is None else x
        y = x.dot(A.T) + np.sqrt(sigma) * np.random.normal(size=(x.shape[0],self.D_out))

        if self.affine:
            y += b.T

        return np.hstack((x,y)) if return_xy else y

    ### Gibbs sampling

    def resample(self,data=[],stats=None):
        ss = self._get_statistics(data) if stats is None else stats
        yxT = ss[1]
        xxT = ss[2]

        # Posterior mean of a Gaussian
        Sigma_A_post = np.linalg.inv(xxT + self.Sigma_A_inv)
        mu_A_post = (yxT + self.mu_A.dot(self.Sigma_A_inv)).dot(Sigma_A_post)

        # self.A = np.random.multivariate_normal(mu_A_post, Sigma_A_post)
        self.A = mu_A_post + np.random.normal(size=(1,self.D_in)).dot(np.linalg.cholesky(Sigma_A_post).T)

    ### Prediction
    def predict(self, X):
        A, sigma = self.A, self.sigma
        if self.affine:
            A, b = A[:,:-1], A[:,-1]

        y = X.dot(A.T)
        if self.affine:
            y += b.T
        return y

    ### Collapsed
    def log_marginal_likelihood(self,data):
        pass
        # The marginal distribution for multivariate Gaussian mean and
        #  fixed covariance is another multivariate Gaussian.
        if isinstance(data, list):
            return [self.log_marginal_likelihood(d) for d in data]
        elif isinstance(data, np.ndarray):
            X,y = data[:,:-1], data[:,-1]
            N = data.shape[0]

            # TODO: Implement this with matrix inversion lemma
            # Compute the marginal distribution parameters
            mu_marg = X.dot(self.mu_A.T)
            # Covariances add
            Sig_marg = self.sigma * np.eye(N) + X.dot(self.Sigma_A.dot(X.T))

            # Compute the marginal log likelihood
            return GaussianFixed(mu_marg, Sig_marg).log_likelihood(y)
        else:
            raise Exception("Data must be list of numpy arrays or numpy array")


class NegativeBinomialRegression(Regression):
    """
    Psi = log(p/(1-p)) is a simple Gaussian linear regression in this model
    The tricky part is sampling Psi given the observed counts. To do so, we
    use the Polya-gamma auxiliary variable trick
    """
    def __init__(self,
            nu_0=None,S_0=None,M_0=None,K_0=None,
            A=None,sigma=None,
            xi=1.0,
            pg_truncation=500):

        affine = False
        self.data_list = []

        super(NegativeBinomialRegression, self).__init__(nu_0, S_0, M_0, K_0, affine, A, sigma)
        self.xi = xi
        self.trunc = pg_truncation


    @property
    def D_in(self):
        # NOTE: D_in includes the extra affine coordinate
        mat = self.A if self.A is not None else self.natural_hypparam[1]
        return mat.shape[1]

    @property
    def D_out(self):
        mat = self.A if self.A is not None else self.natural_hypparam[1]
        return mat.shape[0]

    def mu_psi(self, X):
        T = X.shape[0]
        return np.dot(X, self.A.T).reshape((T,))

    def add_data(self, X, counts):
        """
        Unlike the typical regression class, here we have auxiliary variables that
        need to persist from one MCMC iteration to the next. To implement this, we
        keep the data around as a class variable. This is pretty much the same as
        what is done with the state labels in PyHSMM.

        :param data:
        :return:
        """
        # Data must be a T x (D_in + 1) matrix where the last column contains the counts
        assert X.ndim == 2 and X.shape[1] == self.D_in
        T = X.shape[0]
        assert counts.ndim == 1 and counts.size == T

        assert np.all(counts >= 0)
        counts = counts.astype(np.int)

        # Create an augmented counts object
        augmented_data = AugmentedNegativeBinomialCounts(X, counts, self)
        self.data_list.append(augmented_data)

    ### distribution

    def log_likelihood(self, X, y):
        # X, y = xy[:,:-1], xy[:,-1]
        T = X.shape[0]
        A, sigma = self.A, self.sigma

        psi = X.dot(A.T) + np.random.normal(size=(T,self.D_out))\
                .dot(np.linalg.cholesky(sigma).T)

        # Convert the psi's into the negative binomial rate parameter, p
        p = np.exp(psi) / (1.0 + np.exp(psi))
        p = np.clip(p, 1e-16, 1-1e-16)

        ll = gammaln(self.xi + y).sum() - T * gammaln(self.xi) - gammaln(y+1).sum() + \
             self.xi * np.log(1.0-p).sum() - (y*np.log(p)).sum()
        return ll

    def rvs(self,x=None,size=1,return_xy=True):
        A, sigma = self.A, self.sigma

        x = np.random.normal(size=(size,A.shape[1])) if x is None else x
        psi = x.dot(A.T) + np.random.normal(size=(x.shape[0],self.D_out))\
                .dot(np.linalg.cholesky(sigma).T)

        # Convert the psi's into the negative binomial rate parameter, p
        p = np.exp(psi) / (1.0 + np.exp(psi))

        # Sample the negative binomial. Note that the definition of p is
        # backward in the Numpy implementation
        y = np.random.negative_binomial(self.xi, 1.0-p)

        return np.hstack((x,y)) if return_xy else y

    ### Gibbs sampling

    def resample(self, data=None, stats=None):
        assert data is None
        assert stats is None

        for augmented_data in self.data_list:
            # Sample omega given the data and the psi's derived from A, sigma, and X
            augmented_data.resample()

        if len(self.data_list) > 0:
            # Concatenate the X's and psi's
            X = np.vstack([data.X for data in self.data_list])
            psi = np.vstack([data.psi[:,None] for data in self.data_list])

            # Resample the Gaussian linear regression given the psi's and the X's as data
            super(NegativeBinomialRegression, self).resample(np.hstack([X, psi]))
        else:
            super(NegativeBinomialRegression, self).resample()

        # TODO: Resample the xi's under a gamma prior


class SpikeAndSlabNegativeBinomialRegression(GibbsSampling):
    """
    Psi = log(p/(1-p)) is a spike-and-slab Gaussian linear regression in this model
    The tricky part is sampling Psi given the observed counts. To do so, we
    use the Polya-gamma auxiliary variable trick.

    y_n ~ NB(xi, p_n)
    p_n = exp(psi_n) / (1+ exp(psi_n))
    psi_n = bias + sum_{m} A_m * X_m * w_m + phi * N(0,1)

    where:

    bias ~ Normal(mu_bias, sigma_bias)
    A_m ~ Bernoulli(rho_m)
    w_m ~ Normal(mu_m, sigma_m)
    phi ~ InvGamma(a_phi, b_phi)

    Note the change of notation from the regular Regression model:
    A -> w
    sigma -> phi

    A is now an indicator
    """
    def __init__(self,
            bias_model, regression_models,
            rho_s=None, a_phi=None, b_phi=None,
            As=None, phi=None,
            xi=10,
            pg_truncation=500):

        self.xi = xi
        self.trunc = pg_truncation
        self.data_list = []

        self.rho_s = rho_s
        self.a_phi = a_phi
        self.b_phi = b_phi

        self.bias_model = bias_model
        self.regression_models = regression_models
        self.As = As
        self.phi = phi

        # For each parameter, make sure it is either specified or given a prior

        if As is None:
            assert rho_s is not None
            self.resample_As_and_regression_models()

        if phi is None:
            assert not any(_ is None for _ in (a_phi, b_phi))
            self.resample_phi()

        # Set the number of inputs
        self.M = len(self.As)
        assert len(self.regression_models) == self.M

        self.Ds = np.array([rm.D_in for rm in self.regression_models])

    def add_data(self, Xs, counts):
        """
        Unlike the typical regression class, here we have auxiliary variables that
        need to persist from one MCMC iteration to the next. To implement this, we
        keep the data around as a class variable. This is pretty much the same as
        what is done with the state labels in PyHSMM.

        :param data:
        :return:
        """
        assert counts.ndim == 1 and np.all(counts >= 0)
        T = counts.shape[0]
        counts = counts.astype(np.int)

        assert len(Xs) == self.M
        for D,X in zip(self.Ds, Xs):
            # Data must be a T x (D_in + 1) matrix where the last column contains the counts
            assert X.ndim == 2 and X.shape == (T,D)

        # Create an augmented counts object
        augmented_data = AugmentedNegativeBinomialCounts(Xs, counts)
        self.data_list.append(augmented_data)

    def mu_psi(self, Xs):
        mu = self.bias_model
        for X,A,rm in zip(Xs, self.As, self.regression_models):
            mu += A * rm.predict(X)

        return mu

    ### distribution

    def log_likelihood(self, Xs, y):
        psi = self.mu_psi(Xs)

        # Convert the psi's into the negative binomial rate parameter, p
        p = np.exp(psi) / (1.0 + np.exp(psi))
        p = np.clip(p, 1e-16, 1-1e-16)

        ll = gammaln(self.xi + y).sum() - y.size * gammaln(self.xi) - gammaln(y+1).sum() + \
             self.xi * np.log(1.0-p).sum() - (y*np.log(p)).sum()
        return ll

    def rvs(self, Xs=None, size=1, return_xy=True):

        if Xs is None:
            T = size
            Xs = []
            for D in self.Ds:
                Xs.append(np.random.normal(size=(T,D)))
        else:
            T = Xs[0].shape[0]
            Ts = np.array([X.shape[0] for X in Xs])
            assert np.all(Ts == T)

        psi = self.mu_psi(Xs) + np.sqrt(self.phi) * np.random.normal(size=(T,))

        # Convert the psi's into the negative binomial rate parameter, p
        p = np.exp(psi) / (1.0 + np.exp(psi))

        # Sample the negative binomial. Note that the definition of p is
        # backward in the Numpy implementation
        y = np.random.negative_binomial(self.xi, 1.0-p)

        return Xs,y if return_xy else y

    ### Gibbs sampling

    def resample(self, data=None, stats=None):
        assert data is None
        assert stats is None

        for augmented_data in self.data_list:
            # Sample omega given the data and the psi's derived from A, sigma, and X
            augmented_data.resample(self.xi, self.A, self.sigma)

        if len(self.data_list) > 0:
            # Concatenate the X's and psi's
            X = np.vstack([data.X for data in self.data_list])
            psi = np.vstack([data.psi[:,None] for data in self.data_list])

            # Resample the Gaussian linear regression given the psi's and the X's as data
            super(NegativeBinomialRegression, self).resample(np.hstack([X, psi]))
        else:
            super(NegativeBinomialRegression, self).resample()

        # TODO: Resample the xi's under a gamma prior

    def resample_phi(self):
        """
        Resample the noise variance phi.

        :return:
        """

        # Update the regression model covariances
        for rm in self.regression_models:
            rm.sigma = self.phi

    def resample_bias(self):
        """
        Resample the bias given the weights and psi
        :return:
        """

        psi_resids = []
        for data in self.data_list:
            psi_resids.append(data.psi - (self.mu_psi(data.X) - self.bias))

        self.bias_model.resample(psi_resids)

    def resample_As_and_regression_models(self):
        """
        Jointly resample the spike and slab indicator variables and regression models
        :return:
        """
        # For each regression model, sample the spike and slab variable
        for m in range(self.M):
            D = self.Ds[m]
            rho = self.rho_s[m]
            rm = self.regression_models[m]

            # Compute residual
            self.As[m] = 0  # Make sure mu is computed without the current regression model
            residuals = [d.psi - self.mu_psi(d) for d in self.data_list]

            lp_A = np.zeros(2)

            # Compute log Pr(A=0|...)
            lp_A[0] = np.log(1.0-rho) + GaussianFixed(0, self.phi).log_likelihood(residuals)

            # Compute log Pr(A=1|...)
            lp_A[1] = np.log(rho) + rm.log_marginal_likelihood(residuals)

            # Sample the spike variable
            self.As[m] = log_sum_exp_sample(lp_A)

            # Sample the slab variable
            if self.As[m]:
                rm.resample(residuals)

def test_nbregression():
    # Make a model
    D = 2
    xi = 10
    b = -1.0
    A = np.ones((1,D+1))
    A[0,-1] = b

    sigma =  0.001 * np.ones((1,1))
    true_model = NegativeBinomialRegression(A=A, sigma=sigma, xi=xi)

    # Make synthetic data
    T = 100
    X = np.random.normal(size=(T,D))
    X = np.hstack((X, b*np.ones((T,1))))
    y = true_model.rvs(X, return_xy=False).reshape((T,))

    psi = X.dot(A.T)
    print "Max Psi:\t", np.amax(psi)
    print "Max p:\t", np.amax(np.exp(psi)/(1.0 + np.exp(psi)))
    print "Max y:\t", np.amax(y)

    # Scatter the data
    import matplotlib.pyplot as plt
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.scatter(X[:,0], X[:,1], c=y, cmap='hot')
    plt.colorbar()
    plt.show()

    # Fit the model with a matrix normal prior on A and sigma
    nu = D+1
    M = np.zeros((1,D+1))
    S = np.eye(1)
    K = np.eye(D+1)
    inf_model = NegativeBinomialRegression(nu_0=nu, S_0=S, M_0=M, K_0=K, xi=xi)

    # Add data
    inf_model.add_data(X, y)

    # MCMC
    for i in range(100):
        inf_model.resample()
        print "ll:\t", inf_model.log_likelihood(X, y)
        print "A:\t", inf_model.A
        print "sig:\t", inf_model.sigma



test_nbregression()


