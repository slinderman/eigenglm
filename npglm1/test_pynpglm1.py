import numpy as np
import pyeigenglm as pe

# Create a spike train
T = 100
N = 2
dt = 0.001
S = np.random.randint(0,10,T).astype(np.double)

# Filter the spike train
filtered_S = []
D_imp = 1
for n in range(N):
    filtered_S.append(np.random.randn(T, D_imp).astype(np.double))

st = pe.PyNpSpikeTrain(N, T, dt,  S, D_imp, filtered_S)

print st

