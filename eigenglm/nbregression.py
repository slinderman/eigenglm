"""
Implement a distribution class for negative binomial regression
"""
import numpy as np
from scipy.special import gammaln

from deps.pybasicbayes.distributions import Regression
from hips.distributions.polya_gamma import polya_gamma

class AugmentedNegativeBinomialCounts:
    """
    Class to keep track of a set of counts and the corresponding Polya-gamma
    auxiliary variables associated with them.
    """
    def __init__(self, X, counts):
        # Data must be a T x (D_in + 1) matrix where the last column contains the counts
        assert counts.ndim == 1
        self.counts = counts
        self.T = counts.shape[0]

        assert X.ndim == 2 and X.shape[0] == self.T
        self.X = X

        # Initialize auxiliary variables
        self.omegas = np.ones(self.T)
        self.psi = np.zeros(self.T)

    def resample(self, xi, w, sigma, trunc=500):
        """
        Resample omega given xi and psi, then resample psi given omega, X, w, and sigma
        """
        # import pdb; pdb.set_trace()
        sigma = np.asscalar(sigma)
        # Resample the auxiliary variables, omega
        self.omega = polya_gamma(self.counts+xi, self.psi, trunc)

        # Resample the rates, psi given omega and the regression parameters
        sig_post = 1.0 / (1.0/sigma + self.omega)
        mu_post = sig_post * ((self.counts-xi)/2.0 + np.dot(self.X, w.T).reshape((self.T,)) / sigma)
        self.psi = mu_post + np.sqrt(sig_post) * np.random.normal(size=(self.T,))


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
        augmented_data = AugmentedNegativeBinomialCounts(X, counts)
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


