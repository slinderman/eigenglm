import numpy as np

from eigenglm.nbregression import RegressionFixedCov

# Make a model
D = 2
xi = 10
b = -1.0
A = np.ones((1,D+1))
A[0,-1] = b

A_true = np.ones((1,D))
mu_A = np.zeros((1,D))
eta = 1.0
Sigma_A = eta * np.eye(D)
sigma =  0.1
true_model = RegressionFixedCov(A=A_true, sigma=sigma)

# Make synthetic data
T = 100
X = np.random.normal(size=(T,D))
# X = np.hstack((X, b*np.ones((T,1))))
xy = true_model.rvs(X)
y = xy[:,-1]

print "Max y:\t", np.amax(y)

# Scatter the data
import matplotlib.pyplot as plt
plt.figure()
plt.gca().set_aspect('equal')
plt.scatter(X[:,0], X[:,1], c=y, cmap='hot')
plt.colorbar()

# Plot A
l_true = plt.plot([0, A_true[0,0]], [0, A_true[0,1]], ':k')

# Fit the model with a matrix normal prior on A and sigma
inf_model = RegressionFixedCov(mu_A=mu_A, Sigma_A=Sigma_A, sigma=sigma)

# Plot the initial sample
l_inf = plt.plot([0, inf_model.A[0,0]], [0, inf_model.A[0,1]], '-k')

# Begin interactive plotting
plt.ion()
plt.show()

raw_input("Press any key to begin sampling...\n")

# MCMC
for i in range(100):
    print "ll:\t", inf_model.log_likelihood(xy)
    print "A:\t", inf_model.A
    inf_model.resample(xy)
    l_inf[0].set_data([0, inf_model.A[0,0]], [0, inf_model.A[0,1]])
    plt.pause(0.1)
