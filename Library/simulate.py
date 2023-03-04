import pandas as pd
import numpy as np
from Library.variance import spear_cor

def chol_psd(mat):
    # cholesky for psd
    L = np.zeros(mat.shape)
    n = len(mat)
    for i in range(0, n):
        s = 0
        if i > 0:
            s = np.dot(L[i, :i].T, L[i, :i])

        temp = mat[i, i] - s
        if 0 >= temp >= -1e-8:
            temp = 0
        L[i, i] = np.sqrt(temp)

        if L[i, i] == 0:
            L[i, (i + 1):] = 0
        else:
            ir = 1 / L[i, i]
            for j in range((i + 1), n):

                s = np.dot(L[j, :i].T, L[i, :i])
                if i == 0:
                    s = 0

                L[j, i] = (mat[j, i] - s) * ir
    return L


def near_psd(a, epsilon=0):
    # nearest psd
    n = len(a)
    result = a
    # check corr:

    is_cov = np.any(np.diagonal(a) != 1)

    if is_cov:
        inv_std = np.diag(1 / np.sqrt(np.diagonal(a)))
        result = np.dot(np.dot(inv_std, a), inv_std)
    vals, vecs = np.linalg.eigh(result)
    vals[np.where(vals < epsilon)] = epsilon
    T = 1 / np.dot(np.square(vecs), vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = np.dot(np.dot(T, vecs), l)
    result = np.dot(B, B.T)
    if is_cov:
        std = np.sqrt(np.diagonal(a))
        result = (std.reshape(n, 1)) * result * (std.reshape(1, n))

    return result


def fro_norm(a):
    return np.sum(np.square(a))


## higham
def P_s(a, wh):
    # helper function for higham, the second projection
    a = wh.dot(a).dot(wh)
    vals, vecs = np.linalg.eigh(a)
    vals[np.where(vals < 0)] = 0
    a = vecs.dot(np.diag(vals)).dot(vecs.T)

    wh_inv = np.diag(1 / np.diagonal(wh))

    a = wh_inv.dot(a).dot(wh_inv)
    return a


def P_u(a):
    # helper function for higham, the first projection
    np.fill_diagonal(a, 1)
    return a


def higham_psd(a, tol=None, max_iter=100, weights=None):
    # higham nearest psd
    if tol is None:
        tol = np.spacing(1) * len(a)
    if weights is None:
        weights = np.ones(len(a))
    w_h = np.diag(np.sqrt(weights))
    Y = np.copy(a)
    ds = np.zeros(np.shape(a))
    for i in range(0, max_iter):
        norm_Y_pre = fro_norm(Y)
        R = Y - ds
        X = P_s(R, w_h)

        ds = X - R
        Y = P_u(X)
        norm_Y = fro_norm(Y)

        if -1 * tol < norm_Y - norm_Y_pre < tol:
            break
    return Y


def PCA_sim(mat, exp=1, nsim=25000):
    # PCA simulation
    vals, vecs = np.linalg.eigh(mat)

    vecs = vecs[:, np.argsort(vals)]

    vals.sort()

    vals = vals[::-1]

    vecs = vecs[:, ::-1]
    vecs = vecs[:, np.where(vals > 0)[0]]

    vals = vals[np.where(vals > 0)]

    k = len(vals)
    explain = np.zeros(k)
    for i in range(0, k):
        explain[i] = vals[i] / vals.sum()

    explain_cum = explain.cumsum()
    if exp < 1:
        k = np.where(explain_cum >= exp)[0][0] + 1
        vals = vals[:k]
        vecs = vecs[:, :k]

    B = np.dot(vecs, np.diag(np.sqrt(vals)))

    r = np.random.normal(0, 1, size=(k, nsim))

    return B @ r


def multi_norm_sim(mat,mu=0, nsim=25000):
    # PCA simulation

    L = chol_psd(mat)
    r = np.random.normal(0, 1, size=(len(mat), nsim))

    return L @ r + mu

def copula_sim(df, method="T", simN=500):
    df = np.transpose(df)
    Z = []
    n = len(df)
    simX = []
    params = []
    if method.upper() =='T':

        for r in df:
            df, mu_t, sigma_t = t.fit(r, method='mle')
            u = t.cdf(r, df, mu_t,sigma_t)
            params.append([df, mu_t, sigma_t])
            Z.append(norm.ppf(u))
        Z = np.array(Z)
        cor = spear_cor(Z)
        sim = np.random.multivariate_normal(mean= np.zeros(n), cov=cor,size=simN)
        sim = sim.T
        for i in range(0,n):
            u = norm.cdf(sim[i])
            param = params[i]
            x = t.ppf(u,df=param[0],loc=param[1],scale=param[2])
            simX.append(x)