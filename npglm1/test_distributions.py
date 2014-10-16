import pydistributions as pd
import numpy as np

dg = pd.PyDiagonalGaussian()
print dg.logp(np.array([1.0]))

print "dg[-1]", dg.grad(np.array([-1.0]))
print "dg[0]", dg.grad(np.array([0.0]))
print "dg[1]", dg.grad(np.array([1.0]))

print "Dirichlet"
dd = pd.PyDirichlet(2)
x = np.array([0.5, 0.5])
alpha = np.array([0.1, 0.1])
print "lp ", x, ": \t", dd.logp(x)
print "np: ", np.sum((0.1 - 1.) * np.log(np.abs(x))) - np.sum(np.abs(x))

print "dlp ", x, ": \t", dd.grad(x)
