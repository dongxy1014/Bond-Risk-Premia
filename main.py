import pandas as pd
import numpy as np
from olsgmm import olsgmm
from prettytable import PrettyTable
from numpy.linalg import inv


"""
 load bond price data
 this is the file from CRSP completely unmodified
 sample: 1952 - 2003
"""

bondprice = np.array(pd.read_table("bondprice.dat", skiprows=[0, 1], sep="     "))

"""
As we're replicating Table1, no need to plot graphs, do nothing with date
"""

bondprice[:, 1:] = np.round(bondprice[:, 1:], 4)
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



"""
 hprx(t) is the holding period return over last year
"""


hpr = prices[12:T, 0:4]-prices[0:T-12, 1:5]
hprx = hpr - np.matmul(yields[0:T-12, 0].reshape(T-12, 1), np.ones((1, 4)))
hpr = np.vstack((np.zeros((12, 4)), hpr))   #pads out the initial values with zeros so same length as other series
hprx = np.vstack((np.zeros((12, 4)), hprx))


beg = 140     # set beginning date 140 = 1964. Same as FB sample, and previous data unreliable per Fama.

"""
 capitalized variables do not follow the convention of being padded out with initial zeros
 instead, HPRX starts 12 months later, so the first HPRX is in 65 while the first FS, FT, YT is 1964. 
 These are set up so you can regress HPRX, AHPRX on YT, FT, etc. directly
 with no subscripting. They also include a column of ones for use in
 regressions. 
"""

HPRX = 100*hprx[beg+12-1:T, :]
AHPRX = np.mean(HPRX, axis=1).reshape(-1, 1)

Ts = T-beg-12+1;
FS = np.hstack((np.ones((Ts, 1)), 100*fs[beg-1:T-12, :]))     # forward-spot spread
FT = np.hstack((np.ones((Ts, 1)), 100*yields[beg-1:T-12, 0].reshape(-1, 1), 100*forwards[beg-1:T-12, :])) # yeilds and forwards
YT = np.hstack((np.ones((Ts, 1)), 100*yields[beg-1:T-12, :]))  # all yields

print('-----------------------------------------------------')
print('TABLE 1 Panel B: Individual-bond regression (Unrestricted Model)')
print('-----------------------------------------------------')


betas, tbetas, R2, R2adj, v, F_trash = olsgmm(HPRX, FT, 12, 0)  # std errors: HH with 12 lags
betas, stbetas_trash, R2_trash, R2adj_trash, v, F = olsgmm(HPRX, FT, 18, 1)  # tests: NW with 18 lags
errall = HPRX - np.matmul(FT, betas)


# regressions of actual (not log) returns to check Gallant suspicions

lhv = 100*(np.exp(hprx[beg+12-1:T,:] + np.dot(yields[beg-1: T-12,0].reshape(T-11-beg,1), np.ones((1,4)))) - np.dot\
    (yields[beg-1: T-12,0].reshape(T-11-beg,1), np.ones((1,4))))


blevel, stlevel, R2level, R2adjlevel, vlevel, Flevel = olsgmm(lhv, FT, 12, 0)


# round to 2 decimal

R2 = np.around(R2, 2)
R2level = np.around(R2level, 2)
F = np.around(F, 2)

PanelB_un = PrettyTable()
PanelB_un.field_names = ["n", "R2", "EH", "Level R2", "Chi2(5)"]

PanelB_un.add_row([2, R2[0], "NA", R2level[0], F[0,0]])
PanelB_un.add_row([3, R2[1], "NA", R2level[1], F[1,0]])
PanelB_un.add_row([4, R2[2], "NA", R2level[2], F[2,0]])
PanelB_un.add_row([5, R2[3], "NA", R2level[3], F[3,0]])

print(PanelB_un)

# Save as text file for review

PanelB_un = PanelB_un.get_string()
with open('PanelB_un.txt', 'w') as file:
    file.write(PanelB_un)


print('Column EH unfinished')


print('-----------------------------------------------------')
print('TABLE 1 Panel B: Individual-bond regression (Restricted Model)')
print('-----------------------------------------------------')


olsse = np.zeros((10, 1))
F = olsgmm(AHPRX, FT, 18, 1)[5]  # joint tests use NW 18 lags
gammas, olsse[4:, :], R2hump, R2humpadj, v = olsgmm(AHPRX, FT, 12, 0)[:5]  # std errors using HH
hump = np.matmul(FT, gammas)  # in sample fit.


# Estimation of b's (without constant)
bets, temp, R2hprx, R2hprxadj, v = olsgmm(HPRX, hump, 12, 0)[:5]


"""
-----------------------------------------------------------------
No need for TABLE 1 Panel B (restricted), just in case if needed
----------------------------------------------------------------
erravg = AHPRX-hump
humpall = np.matmul(np.hstack((np.ones((T, 1)), 100*yields[:, 0].reshape(T, 1), 100*forwards)), gammas)

olsse[:4, :] = temp.T
err = HPRX - np.matmul(hump, bets)

# Calculate standard errors of two step by using two step OLS moments

u = np.hstack((np.multiply(np.matmul(erravg, np.ones((1, 6))), FT), np.multiply(err, np.matmul(hump, np.ones((1, 4))))))
gt = np.mean(u, axis=0).reshape(10, 1)
Eff = np.matmul(FT.T, FT)/FT.shape[0]
Eef = np.matmul(err.T, FT)/FT.shape[0]
Erf = np.matmul(HPRX.T, FT)/FT.shape[0]
d = - np.vstack((np.hstack((Eff,np.zeros((6,4)))),
               np.hstack((-Erf+2*np.matmul(np.matmul(bets.T,gammas.T), Eff),
                          np.matmul(np.matmul(gammas.T, Eff), gammas)[0, 0]*np.identity(4)))))
S = spectralmatrix(u, 12, 0)
gmmsex = np.power(np.diag(np.matmul(np.matmul(inv(d), S), inv(d).T))/FT.shape[0], 0.5)
gmmse = np.zeros(olsse.shape[0])
gmmse[4:] = gmmsex[:6]  # formula above was derived with gamma first
gmmse[:4] = gmmsex[6:]  # convention below is bs first
"""

# round to 2 decimal

bets = np.around(bets.flatten(), 2)
R2hprx = np.around(R2hprx, 2)

PanelB_res = PrettyTable()
PanelB_res.field_names = ["n", "bn", "Large T", "Small T", "R2", "Small T (range)"]

PanelB_res.add_row([2, bets[0], "NA", "NA", R2hprx[0], "NA"])
PanelB_res.add_row([3, bets[1], "NA", "NA", R2hprx[1], "NA"])
PanelB_res.add_row([4, bets[2], "NA", "NA", R2hprx[2], "NA"])
PanelB_res.add_row([5, bets[3], "NA", "NA", R2hprx[3], "NA"])

print(PanelB_res)

# Save as text file for review

PanelB_res = PanelB_res.get_string()
with open('PanelB_res.txt', 'w') as file:
    file.write(PanelB_res)


print('-----------------------------------------------------')
print('TABLE 1 Panel A: Estimate of the return-forecasting factor ')
print('-----------------------------------------------------')

# OLS & HH, 12 lags

gammas, olsse[4:, :], R2hump, R2humpadj, vHH, FHH = olsgmm(AHPRX, FT, 12, 0)  # std errors using HH
teststat = np.around(np.matmul(gammas[1:, ].T, np.matmul(inv(vHH[1:, 1:]), gammas[1:, ]))[0, 0], 2)

OLSgammas = np.around(gammas.flatten(), 2)
HH12lags = np.around(olsse[4:, :].flatten(), 2)
HH12lags = ['(' + str(i) + ')' for i in HH12lags]
OLSR2 = np.around(R2hump[0], 2)

PanelA = PrettyTable()
PanelA.field_names = [" ", "gamma0", "gamma1", "gamma2", "gamma3", "gamma4", "gamma5", "R2", "chi2(5)"]

PanelA.add_row(["OLS estimates", OLSgammas[0], OLSgammas[1], OLSgammas[2], OLSgammas[3], OLSgammas[4], OLSgammas[5], OLSR2, " "])
PanelA.add_row(["HH, 12 lags", HH12lags[0], HH12lags[1], HH12lags[2], HH12lags[3], HH12lags[4], HH12lags[5], " ", teststat])

# NW 18 lags

gammas_NW, se_NW, R2_trash, R2adj_trash, vNW, FNW = olsgmm(AHPRX, FT, 18, 1)
teststat = np.around(np.matmul(gammas_NW[1:, ].T, np.matmul(inv(vNW[1:, 1:]), gammas_NW[1:, ]))[0, 0], 2)
NW18lags = np.around(se_NW.flatten(), 2)
NW18lags = ['(' + str(i) + ')' for i in NW18lags]
PanelA.add_row(["NW, 18 lags", NW18lags[0], NW18lags[1], NW18lags[2], NW18lags[3], NW18lags[4], NW18lags[5], " ", teststat])

# Simplified HH

err = AHPRX - np.matmul(FT, gammas)
sigerr = np.matmul(err.T, err)[0,0]/Ts
Exx = np.matmul(FT.T, FT)/Ts
S = np.matmul(FT.T, FT)/Ts * sigerr

for i in range(1, 12):
    Sadd = np.matmul(FT[i:, :].T, FT[:-i, ])/Ts * sigerr *(12-i)/12 + np.matmul(FT[:-i, :].T, FT[i:, :])/Ts * sigerr * (12-i)/12
    S = S + Sadd

v_st = np.matmul(np.matmul(inv(Exx),S), inv(Exx))/Ts
se_st = np.diag(v_st)
se_st = np.around(np.sign(se_st)*np.power(np.abs(se_st), 0.5), 2)
se_st = ['(' + str(i) + ')' for i in se_st]
teststat = np.around(np.matmul(gammas[1:, ].T, np.matmul(inv(v_st[1:, 1:]), gammas[1:, ]))[0, 0], 2)
PanelA.add_row(["Simplified HH", se_st[0], se_st[1], se_st[2], se_st[3], se_st[4], se_st[5], " ", teststat])

# Non-Overlapping

ann = np.arange(1, Ts, 12)
v2avg = np.zeros((6, 6))

se_mo = np.zeros((6, 12))
for smonth in range(1,13):
    gammas_mo, temp, R2_mo, R2adj_mo, v2_mo = olsgmm(AHPRX[ann + smonth - 2], FT[ann + smonth - 2, :], 0, 0)[:5]
    se_mo[:, smonth - 1] = temp.flatten()
    v2avg = v2avg + v2_mo

v2avg = v2avg/12
teststat = np.around(np.matmul(gammas[1:, ].T, np.matmul(inv(v2avg[1:, 1:]), gammas[1:, ]))[0, 0], 2)
NoOverlap = np.around(np.mean(se_mo, axis=1), 2)
NoOverlap = ['(' + str(i) + ')' for i in NoOverlap]
PanelA.add_row(["No overlap", NoOverlap[0], NoOverlap[1], NoOverlap[2], NoOverlap[3], NoOverlap[4], NoOverlap[5], " ", teststat])
print(PanelA)

# Save as text file for review

PanelA = PanelA.get_string()
with open('PanelA.txt', 'w') as file:
    file.write(PanelA)


