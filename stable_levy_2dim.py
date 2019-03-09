import numpy as np
from math import exp
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necessary for 3D plotting
import matplotlib.cm as cm
from pylab import figure, plot, show, grid, axis, xlabel, ylabel, title, hold, draw
import scipy.stats as stats
from scipy.stats import levy_stable
import seaborn as sns
from sympy import plot_implicit
plt.style.use('seaborn-whitegrid')



# simulation of one dimensional symmetric stable process
a1 = 1.9
N=1000
T=1
dt = T/N

#generation of N independent r.v. W_i as Unif([-0.5*pi, 0.5*pi])
Gamma = np.random.uniform(-0.5*np.pi, 0.5*np.pi, N)

#generation of N independent r.v. W as standard exponential
W = np.random.exponential(1, N)

#compute increments of a symetric a-stable
Delta_X = pow(dt, 1/a1)*(np.sin(a1*Gamma))/ pow(np.cos(Gamma), 1/a1)* \
                                   pow((np.cos((1-a1)*Gamma))/W, (1-a1)/a1)

#simulation of the discretized trajectory for the symmetric a-stable X~S_a(sigma, b=0, mu)
stable1 = np.cumsum(Delta_X)

# simulation of one dimensional symmetric stable process
index = [0.5, 0.9]
def two_dim_stable_increments():
    Delta_X = np.zeros(len(index), np.size(N))
    stable = np.zeros(len(index), np.size(N))
    for i in range(len(index)):
        for a in index:
            Delta_X[i] = pow(dt, 1/a)*(np.sin(a*Gamma))/ pow(np.cos(Gamma), 1/a)* \
                                           pow((np.cos((1-a)*Gamma)), (1-a)/a)
            stable[i]= np.cumsum(Delta_X[i])
            return stable
stable = two_dim_stable_increments()
print(stable)



plt.show()
