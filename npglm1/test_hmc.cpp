#include "inference.h"
#include <Eigen/Core>
#include <iostream>

using namespace Eigen;

class GaussianHmcSampler : public AdaptiveHmcSampler
{
    MatrixXd mu;
    MatrixXd sigma;

public:
    GaussianHmcSampler(MatrixXd mu, MatrixXd sigma, std::default_random_engine rng) : AdaptiveHmcSampler(rng)
    {
        this->mu = mu;
        this->sigma = sigma;
    }

    double logp(MatrixXd x)
    {
        return (-0.5/sigma.array().pow(2) * (x.array() - mu.array()).pow(2)).sum();
    }

    MatrixXd grad(MatrixXd x)
    {
        return -(x.array()-mu.array())/sigma.array();
    }

};

int main()
{
    // Create a Gaussian sampler
    int D = 2;
    RowVectorXd mu = VectorXd::Constant(D, 0);
    RowVectorXd sigma = VectorXd::Constant(D, 1);

    int seed = time(NULL);
    std::cout << "Random seed: " << seed << std::endl;
    std::default_random_engine rng(seed);
    GaussianHmcSampler* sampler = new GaussianHmcSampler(mu, sigma, rng);

    int N_samples = 10000;
    MatrixXd samples = MatrixXd::Zero(N_samples, D);
    MatrixXd next_sample(1,D);

    // Iteratively call HMC, starting from previous sample.
    for (int s=1; s<N_samples; s++)
    {
        if (s % 250 == 0)
        {
            std::cout << "Iteration:\t" << s << std::endl;
//            std::cout << "Step sz: " << sampler->step_sz << std::endl;
//            std::cout << "Accept rate: " << sampler->avg_accept_rate << std::endl;
        }
        sampler->sample(samples.row(s-1), &next_sample);
        samples.row(s) = next_sample;
    }

    // Compute sample statistics
    VectorXd sample_mean = samples.colwise().mean();
    VectorXd sample_var = (samples.rowwise() - mu).array().pow(2).colwise().mean();

    std::cout << "Sample mean: " << sample_mean << std::endl;
    std::cout << "Sample var: " << sample_var << std::endl;

}

/*
def test_gamma_linear_regression_hmc():
    """
    Test ARS on a gamma distributed coefficient for a gaussian noise model
    y = c*x + N(0,1)
    c ~ gamma(2,2)
    """
    a = 6.
    b = 1.
    x = 1
    sig = 1.0
    avg_accept_rate = 0.9
    stepsz = 0.01
    nsteps = 10
    N_samples = 10000

    from scipy.stats import gamma, norm
    g = gamma(a, scale=1./b)
    prior = lambda logc: a * logc -b*np.exp(logc)
    dprior = lambda logc: a -b*np.exp(logc)
    lkhd = lambda logc,y: -0.5/sig**2 * (y-np.exp(logc)*x)**2
    dlkhd = lambda logc,y: 1.0/sig**2 * (y-np.exp(logc)*x) * np.exp(logc)*x
    posterior = lambda logc,y: prior(logc) + lkhd(logc,y)
    dposterior = lambda logc,y: dprior(logc) + dlkhd(logc,y)

    logc_smpls = np.zeros(N_samples)
    y_smpls = np.zeros(N_samples)
    logc_smpls[0] = np.log(g.rvs(1))
    y_smpls[0] = np.exp(logc_smpls[0]*x) + sig*np.random.randn()

    for s in np.arange(1,N_samples):
        if np.mod(s, 100) == 0:
            print "Sample ", s
        # Sample y given c
        y_smpls[s] = np.exp(logc_smpls[s-1])*x + sig*np.random.randn()

        # Sample c given y
        logc_smpls[s], stepsz, avg_accept_rate =  \
            hmc(lambda logc: -1.0*posterior(logc, y_smpls[s]),
                lambda logc: -1.0*dposterior(logc, y_smpls[s]),
                stepsz, nsteps,
                logc_smpls[s-1].reshape((1,)),
                avg_accept_rate=avg_accept_rate,
                adaptive_step_sz=True)

    import matplotlib.pyplot as plt
    f = plt.figure()
    _, bins, _ = plt.hist(np.exp(logc_smpls), 20, normed=True, alpha=0.2)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, g.pdf(bincenters), 'r--', linewidth=1)
    plt.show()
*/
