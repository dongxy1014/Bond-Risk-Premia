import numpy as np
from numpy.linalg import inv
from scipy.stats import chi2


def olsgmm(lhv, rhv, lags, weight):


    """

    :param lhv: T x N vector, left hand variable data
    :param rhv: T x K matrix, right hand variable data
    If N > 1, this runs N regressions of the left hand columns on all the (same) right hand variables
    :param lags:  number of lags to include in GMM corrected standard errors
    :param weight: 1 for newey-west weighting
                   0 for even weighting
                  -1 skips standard error computations. This speeds the program up a lot; used inside monte carlos where only estimates are needed

    :return: b: regression coefficients K x 1 vector of coefficients
             seb: K x N matrix standard errors of parameters.
             (Note this will be negative if variance comes out negative)
             v: variance covariance matrix of estimated parameters. If there are many y variables, the vcv are stacked vertically
             R2v:    unadjusted
             R2vadj: adjusted R2
             F: [Chi squared statistic    degrees of freedom    pvalue] for all coeffs jointly zero.
    Note: program checks whether first is a constant and ignores that one for test
    """

    global Exxprim
    global inner

    if rhv.shape[0] != lhv.shape[0]:
        print('olsgmm: left and right sides must have same number of rows. Current rows are %d %d'
              % (rhv.shape[0], lhv.shape[0]))

    T = lhv.shape[0]
    N = lhv.shape[1]
    K = rhv.shape[1]
    sebv = np.zeros((K, N))
    Exxprim = inv((np.matmul(rhv.T, rhv))/T)
    bv = np.linalg.lstsq(rhv, lhv,  rcond=None)[0] # This equals to bv = rhv\lhv in Matlab

    if weight == -1:
        sebv = np.nan
        R2v = np.nan
        R2vadj = np.nan
        v = np.nan
        F = np.nan
    else:
        errv = lhv - np.matmul(rhv, bv)
        s2 = np.mean(np.power(errv, 2), axis=0)
        vary = lhv - np.matmul(np.ones((T,1)), np.mean(lhv, axis=0).reshape(1,lhv.shape[1]))
        vary = np.mean(np.power(vary,2),axis=0)

        R2v = 1 - s2/vary
        R2vadj = 1 - (s2/vary)*(T-1)/(T-K)

        # compute GMM standard errors
        F = np.zeros((N, 3))
        for indx in range(N):
            err = errv[:, indx].reshape(errv.shape[0], 1)
            inner = np.matmul(np.multiply(rhv, np.matmul(err, np.ones((1, K)))).T,
                              np.multiply(rhv, np.matmul(err, np.ones((1, K)))))/T

            for jindx in range(lags):
                inneradd = np.matmul(np.multiply(rhv[0:T-jindx-1,:], np.matmul(err[0:T-jindx-1,:], np.ones((1,K)))).T,
                                     np.multiply(rhv[(1+jindx):T,:], np.matmul(err[(1+jindx):T],np.ones((1, K)))))/T
                inner = inner + (1 - weight*(1+jindx)/(lags+1))*(inneradd + inneradd.T)

            varb = 1 / T * np.matmul(np.matmul(Exxprim,inner),Exxprim)

            # F test for all coeffs(except constant) zero - - actually chi2 test

            if (rhv[:, 0].reshape(rhv.shape[0],1) == np.ones((rhv.shape[0],1))).all():
                chi2val = np.matmul(np.matmul(bv[1:,indx].T, inv(varb[1:,1:])), bv[1:,indx])
                dof = bv[1:, 0].shape[0]
                pval = 1-chi2.cdf(chi2val, df=dof)
                F[indx, 0:3] = [chi2val, dof, pval]
            else:
                chi2val = np.matmul(np.matmul(bv[:, indx].T, inv(varb)), bv[:, indx])
                dof = bv[:, 0].shape[0]
                pval = 1 - chi2.cdf(chi2val, df=dof)
                F[indx, 0:3] = [chi2val, dof, pval]

            if indx == 0:
                v = varb
            else:
                v = np.vstack((v, varb))

            seb = np.diag(varb)
            seb = np.sign(seb)*np.power(np.abs(seb), 0.5)
            sebv[:, indx] = seb

    return (bv, sebv, R2v, R2vadj, v, F)


