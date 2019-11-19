import pandas as pd
import numpy as np
from olsgmm import olsgmm


"""
Choose data-generating process for simulations
mc = 
    1  : VAR(12) for yields       
    2  : Expectations hypothesis, AR(12) for short rate
    3  : Cointegrated VAR(12)

"""

mc = 1

"""
More choices : 
S     = number of simulations (will be displayed in output)
nlags = lags for cov matrix
nw    = 1 for newey west
L     = number of lags in estimated VAR for simulations. Should be 12; too few and you miss lag structure and don't see tent. 
"""

S = 50000
nlags = 18
nw = 1
L = 12

"""
load bond price data
this is the file from CRSP completely unmodified
sample: 1952 - 2003
"""

bondprice = np.array(pd.read_table("bondprice.dat", skiprows=[0, 1], sep="     "))
T = len(bondprice)

y = np.multiply(-np.log(bondprice[:, 1:]/100), np.ones((T, 1))*np.array([1/1, 1/2, 1/3, 1/4, 1/5]))
famablisyld = np.column_stack((bondprice[:, 0], y))


"""
collect the annual yields, form prices, forwards, hpr
yields(t), forwards(t) are yields, forwards at time t
"""

yields = famablisyld[:, 1:]
mats = np.array([[1, 2, 3, 4, 5]]).T
prices = -np.multiply(np.dot(np.ones((T, 1)), mats.T), yields)
forwards = prices[:, 0:4]-prices[:, 1:5]
fs = forwards-np.matmul(yields[:, 0].reshape(T, 1), np.ones((1, 4)))

hpr = prices[12:T, 0:4]-prices[0:T-12, 1:5]
hprx = hpr - np.matmul(yields[0:T-12, 0].reshape(T-12, 1), np.ones((1, 4)))
hpr = np.vstack((np.zeros((12, 4)), hpr))   # pads out the initial values with zeros so same length as other series
hprx = np.vstack((np.zeros((12, 4)), hprx))


# set beginning date 140 = 1964. Same as FB sample, and previous data unreliable per Fama.

yields = yields[139:, ]
forwards = forwards[139:, ]
fs = fs[139:, ]
hprx = hprx[139:, ]
T = len(yields)


HPRX = 100*hprx[12:T, :]
AHPRX = np.mean(HPRX, axis=1).reshape(-1, 1)

FS = np.hstack((np.ones((T-12, 1)), 100*fs[:T-12, :]))     # forward-spot spread
FT = np.hstack((np.ones((T-12, 1)), 100*yields[:T-12, 0].reshape(-1, 1), 100*forwards[:T-12, :]))  # yeilds and forwards
YT = np.hstack((np.ones((T-12, 1)), 100*yields[:T-12, :]))  # all yields


"""
Estimate data-generating process for yields
"""
y = yields * 100

yL = np.ones((T-L, 1))
dyL = np.ones((T-L, 1))
yc = np.ones((T-L, 1))

if mc == 1:
    # right - hand side for VAR(12)
    for i in range(L, 0, -1):
        yL = np.hstack((yL, y[i-1:T-L+i-1, :]))

    """
    arcoef is constant, 
    then 2-6 is AR(1), 7-11 is AR(2), ... 
    """
    arcoef = np.linalg.lstsq(yL, y[L:T, :], rcond=None)[0]
    err = y[L:T, :] - np.matmul(yL, arcoef)   # iid shocks
    yL = yL[:, 1:]                            # rhs without constant

if mc == 3:
    for i in range(L, 0, -1):
        yL = np.hstack((yL, y[i-1:T-L+i-1, :]))

    for i in range(L, 1, -1):
        dyL = np.hstack((dyL, (y[i-1:T-L+i-1, :]-y[i-2:T-L+i-2, :]))) # lagged yield differences

    spr = np.zeros((468, 4))
    spr[:, 0] = yL[:, 1] - yL[:, 5]
    spr[:, 1] = yL[:, 2] - yL[:, 5]
    spr[:, 2] = yL[:, 3] - yL[:, 5]
    spr[:, 3] = yL[:, 4] - yL[:, 5]

    dyL = dyL[:, 1:]

    # left-hand side for co-integrated VAR(12)
    dy = y[L:T, :] - y[L-1:T-1, :]
    RHS = np.hstack((np.ones((T-12, 1)), spr, dyL))
    arcoef = np.linalg.lstsq(RHS, dy, rcond=None)[0]

    err = dy - np.matmul(RHS, arcoef)  # iid shocks

"""
Run Monte Carlo
"""
for s in range(S):
    if (s+1)/1000 == np.floor((s+1)/1000):
        print("Process: N = " + str(s+1))

    if mc == 1:
        yt = y[L, :].reshape(5, 1)  # start in t=L
        ytL = np.append(yL[0, 5:], np.zeros(5)).reshape(1, 61)
        yh = np.zeros((T, 5))
        yh[0:L, :] = y[0:L, :]  #presample values

        for t in range(t, T):
            ytL = np.hstack((yt.T, ytL[:, 0:-5]))
            ut = err[int(np.floor((T-L)*np.random.rand())), :]


