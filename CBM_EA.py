# Copyright (c) 2020 Cognitive & Perceptual Developmental Lab
#                    Washington University School of Medicine

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Primary Contact: Muhamed Talovic
# Email: muhamed.talovic@wustl.edu

# Secondary Contact: Alexandre Todorov
# Email: todorov@wustl.edu

# The following functions have been vetted:
# percentThresh: (AT checked 05/11/2020)
# multipleRegressionShuffle: (MT checked)
# univariatePoissonScreen (MT checked 10/09/2020) - Max error = 9.96416e-07
# univariatePoissonScreenShuffle (MT checked 10/09/2020) - Max error = 9.96416e-07


# --------------------------------- #
# Import necessary built-in functions

import math
import numpy as np
from copy import copy

# --------------------------------- #
# Linear Regression (from Linear Models package - entire code not yet available to the public)

def multipleRegressionShuffle(nshuffles, Y, X, Fc):
    if nshuffles > 0:
        YS = np.array([Y, ]* (nshuffles + 1) ).transpose()
        for i in range(1, nshuffles + 1):
            np.random.shuffle(YS[:, i])
    else:
        YS = Y

    npairs, nobs = Fc.shape
    if X is not None:
        df = nobs - 2 - X.shape[1]
        X = np.hstack((np.ones((nobs, 1)),X))
    else:
        df = nobs - 2
        X = np.ones((nobs, 1))

    XpX = X.T.dot(X)
    A = np.linalg.inv(XpX)
    M = A.dot(X.T)
    Q = np.sum(Fc**2, axis=1)
    Z = (YS - X.dot(M.dot(YS)))
    sse0 = np.sum(Z ** 2, axis=0)
    FX = Fc.dot(X)
    FM = Fc.dot(M.T)
    K = (Q - np.sum(FX*FM, axis=1)).reshape(-1,1)
    G = Fc.dot(Z)
    B = G/K
    S = np.sqrt((sse0 - (G*B))/(df*K))
    B /= S
    return B.T

# --------------------------------- #
# Poisson (from GLM package - entire code not yet available to the public)

def univariatePoissonScreen(Y, Fc):
    convtol = 1e-6
    npairs, nobs = Fc.shape
    sumY = np.sum(Y)
    mu0 = sumY / nobs
    b0 = np.log(mu0)
    LL0 = (b0 - 1.) * sumY
    D0 = Y - mu0

    chi2 = np.zeros(npairs)
    beta = np.zeros(npairs)
    std = np.zeros(npairs)

    B = np.zeros(2)
    mu = np.zeros(nobs, dtype=float)
    Z1 = np.hstack((np.ones((nobs, 1)), np.zeros((nobs, 1))))
    Z2 = np.hstack((np.ones((nobs, 1)), np.zeros((nobs, 1)), np.zeros((nobs, 1))))
    for pair in range(0, npairs):
        Z1[:, 1] = Fc[pair, :]
        Z2[:, 1] = Fc[pair, :]
        Z2[:, 2] = Fc[pair, :]**2

        B[0] = b0
        B[1] = 0

        dLdb0, dLdb1 = D0.dot(Z1)
        d2Ld00, d2Ld10, d2Ld11 = mu0 * np.sum(Z2, axis=0)
        for iter in range(1, 11):
            det = d2Ld00 * d2Ld11 - d2Ld10 ** 2
            H00 = d2Ld11 / det
            H10 = -d2Ld10 / det
            H11 = d2Ld00 / det

            delta0 = H00 * dLdb0 + H10 * dLdb1
            delta1 = H10 * dLdb0 + H11 * dLdb1
            if math.sqrt(delta0 ** 2 + delta1 ** 2) < convtol:
                break

            B[0] += delta0
            B[1] += delta1
            xb = Z1.dot(B)
            np.exp(xb, out=mu)
            D = Y - mu
            dLdb0, dLdb1 = D.dot(Z1)
            d2Ld00, d2Ld10, d2Ld11 = mu.dot(Z2)
        chi2[pair] = Y.dot(xb) - np.sum(mu)
        beta[pair] = B[1]
        std[pair] = math.sqrt(H11)
    chi2 = 2*(chi2 - LL0)
    return beta, beta/std, chi2


def univariatePoissonScreenShuffle(nshuffles, Y, Fc):
    npairs, nobs = Fc.shape
    beta = np.zeros((nshuffles+1, npairs), dtype=float)
    betastd = np.zeros((nshuffles+1, npairs), dtype=float)
    chi2 = np.zeros((nshuffles + 1, npairs), dtype=float)
    YS = copy(Y)
    for i in range(0, nshuffles + 1):
        beta[i,:], betastd[i, :], chi2[i,:]  = univariatePoissonScreen(YS, Fc)
        np.random.shuffle(YS)
    return beta, betastd, chi2


# --------------------------------- #
# Percent Threshold (from utilities package - entire code not yet available to the public)

def percentThresh(rho, p, sides, absval=False):
    """
    percentThresh(rho, p, sides, absval=False)
    Binarize a matrix using percentiles calculated for each row.
    Note: "U, 5" = upper 5% (i.e., the 95th percentile)
    L: B[i, j] = R[i, j] < t[i] where t[i] = percentile p, row i,
    U: B[i, j] = R[i, j] > t[i] where t[i] = percentile p, row i
    T: B[i, j] = R[i, j] > t1[i] OR R[i, j] < t2[i] where t1 and t2 are the p/2 and 100-p/2 percentiles

    :param rho: Input matrix
    :param p: percentile (1 < p <= 50)
    :param sides: L (lower), U (upper), T (2-sided)
    :param absval: Absolute value before calculating percentiles (default=False)
    :return: The binarized matrix (integer, 0 or 1)
    """
    if absval:
        rho = np.abs(rho)

    sides = sides.upper()
    if sides not in ['T', 'L', 'U']:
        print("ERROR Percentile invalid parameter")
        return None

    if p >= 50:
        print("ERROR Percentile must be less than 50")
        return None

    if sides == 'T':
        p /= 2.

    if rho.ndim == 1:
        if sides == 'L' or sides == 'T':
            t1 = np.percentile(rho, p)
        if sides == 'U' or sides == 'T':
            t2 = np.percentile(rho, 100-p)
    else:
        if sides == 'L' or sides == 'T':
            t1 = np.percentile(rho, p, axis=1).reshape(-1, 1)
        if sides == 'U' or sides == 'T':
            t2 = np.percentile(rho, 100-p, axis=1).reshape(-1, 1)

    if sides == 'U':
        F = (rho > t2).astype(int)
    elif sides == 'L':
        F = (rho < t1).astype(int)
    else:
        F = (rho < t1).astype(int) + (rho > t2).astype(int)
    return F



# ---------------------------------------------------------------------------- #
# Run analysis -- Entire code is not yet available to the public.

# I. Import input data and assign to ROI map -- not shown
# (code not yet available for public use):

n_shuffles = 50000 # (50,000 replicates)
alpha_level = 0.05 # (5% of hits)

Y = 'bx scores'
fc = 'correlation z-scores'

# -------------------------------------------------------------------------#
# II. RUNNING ENRICHMENT- LINEAR REGRESSION SHUFFLE (EMPIRICAL):

# 1. run linear regression shuffle:
lin_reg_shuf = multipleRegressionShuffle(n_shuffles, Y, None, fc)
#   returns matrix of beta values for each roi-roi pair
#   first the actual, and then the permuted/shuffled results
#   (shuffling the pairings of bx scores (Y) w/ correlation z-scores (fc))

slin_reg_hits_shuf = percentThresh(lin_reg_shuf, alpha_level * 100, 'T')


# -------------------------------------------------------------------------#
# III. RUNNING ENRICHMENT- POISSON SHUFFLE (EMPIRICAL):

# 1. run linear regression shuffle:
beta, sdbeta_shuff, chi2_shuff = univariatePoissonScreenShuffle(n_shuffles, Y, fc)
#   returns matrix of sd beta values for each roi-roi pair
#   first the actual, and then the permuted/shuffled results
#   (shuffling the pairings of bx scores (Y) w/ correlation z-scores (fc))

poisson_hits_shuf = percentThresh(sdbeta_shuff, alpha_level * 100, 'T')

