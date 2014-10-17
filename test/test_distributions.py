import eigenglm.pydistributions as pd
import numpy as np

dg = pd.PyDiagonalGaussian()
print dg.logp(np.array([1.0]))

print "dg[-1]", dg.grad(np.array([-1.0]))
print "dg[0]", dg.grad(np.array([0.0]))
print "dg[1]", dg.grad(np.array([1.0]))

print "Dirichlet"
dd = pd.PyDirichlet(2)
alpha = np.array([0.1, 0.1])
g = np.random.gamma(alpha)

print "lp ", g, ": \t", dd.logp(g)
print "np: ", np.sum((0.1 - 1.) * np.log(np.abs(g))) - np.sum(np.abs(g))
print "dlp ", g, ": \t", dd.grad(g)

dwdg = dd.dw_dg(g)
w = dd.as_dirichlet(g)
# dg = np.array([1e-3, -1e-3])
dg = 1e-4 * np.random.randn(*g.shape)
dw_pred = np.dot(dwdg,  dg)
wf_pred = w+dw_pred
dw_true = dd.as_dirichlet(g+dg) - w
# print "w(", g, "): \t", w
# print "wf(", g+dg, "): \t",
print "wf pred \t", wf_pred, "\t (sum) ", wf_pred.sum()


print ""
print "Model dw\t", dw_pred
print "Acutal dw\t", dw_true
