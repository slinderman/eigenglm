import numpy as np
from eigenglm.nbregression import RegressionFixedCov, SpikeAndSlabNegativeBinomialRegression
from eigenglm.deps.pybasicbayes.distributions import GaussianFixedCov

def test_ss_nbregression():
    # Make a model
    D = 2
    xi = 10
    b = np.array([-1.0])
    w = np.ones((D,))
    # A = np.ones(D)
    A = np.array([1,1])
    sigma =  0.1 * np.ones((1,1))


    # Make regression models for each dimension
    true_bias_model = GaussianFixedCov(mu=b, sigma=sigma)
    true_regression_models = [RegressionFixedCov(A=w[d].reshape((1,)), sigma=sigma) for d in range(D)]
    true_model = SpikeAndSlabNegativeBinomialRegression(true_bias_model, true_regression_models, As=A, sigma=sigma, xi=xi)

    # Make synthetic data
    datasets = 10
    Xss = []
    ys = []
    T = 100
    for i in range(datasets):
        X = np.random.normal(size=(T,D))
        Xs = [X[:,d].reshape((T,1)) for d in range(D)]
        y = true_model.rvs(Xs, return_xy=False)
        print "Max y:\t", np.amax(y)

        Xss.append(Xs)
        ys.append(y)

    # Scatter the data
    import matplotlib.pyplot as plt
    plt.figure()
    plt.gca().set_aspect('equal')
    if T > 100:
        inds = (T * np.random.random(100)).astype(np.int)
    else:
        inds = np.arange(T)
    plt.scatter(Xss[0][0][inds], Xss[0][1][inds], c=ys[0][inds], cmap='hot')
    plt.title('%d / %d Datapoints' % (len(np.unique(inds)), T))
    plt.colorbar(label='Count')

    # Plot A
    l_true = plt.plot([0, A[0] * w[0]], [0, A[1] * w[1]], ':k')


    # Fit with the same model
    inf_bias_model = GaussianFixedCov(mu_0=np.zeros((1,)), lmbda_0=np.ones((1,1)), sigma=sigma)
    inf_regression_models = [RegressionFixedCov(mu_A=np.zeros((1,)), Sigma_A=np.ones((1,1)), sigma=sigma) for d in range(D)]
    inf_model = SpikeAndSlabNegativeBinomialRegression(inf_bias_model, inf_regression_models,
                                                       rho_s=0.5*np.ones(D), sigma=sigma, xi=xi)

    # Add data
    for Xs,y in zip(Xss, ys):
        inf_model.add_data(Xs, y)

    # Plot the initial sample
    l_inf = plt.plot([0, inf_model.As[0] * inf_regression_models[0].A[0]],
                     [0, inf_model.As[1] * inf_regression_models[1].A[0]], '-k')

    plt.ion()
    plt.show()

    # MCMC
    raw_input("pause\n")
    for i in range(100):
        inf_model.resample()
        print "ll:\t", inf_model.log_likelihood(Xs, y)
        print "A:\t", inf_model.As
        print "bias:\t", inf_bias_model.mu
        # print "sig:\t", inf_model.sigma

        l_inf[0].set_data([0, inf_model.As[0] * inf_regression_models[0].A[0]],
                          [0, inf_model.As[1] * inf_regression_models[1].A[0]])
        plt.pause(0.1)

test_ss_nbregression()
