import numpy as np
from scipy.stats import spearmanr

# calculate the weight terms
def cum_weight(n, lambda_):
    w = np.zeros(n)
    cum_w = np.zeros(n)
    for i in range(0, n):
        w[n - i - 1] = (1 - lambda_) * (np.power(lambda_, i + 1))

    tw = w.sum()
    for i in range(0, n):
        w[n - i - 1] = w[n - i - 1] / tw

    return w


# calculate cumulative variance for each (x,y) pair
def ew_var(x, y, lambda_):
    """
    Calculate exponetially weighted covariance of x and y
    ndarray:param x:
    ndarray:param y:
    float:param lambda_:
    float:return:
    """
    n = len(x)
    w = cum_weight(n, lambda_=lambda_)
    x_bar = np.mean(x)
    y_bar = np.mean(y)

    cov = w.T @ ((x - x_bar) * (y - y_bar))

    return cov


# calculate the covariance matrix
def ew_cov(df, lambda_):
    """
    Calculate exponantially weighted covariance matrix
    ndarray:param df:
    float:param lambda_:
    ndarray:return: exponentially weighted covariance matrix
    """
    n = df.shape[1]
    df = np.transpose(df)
    cov_m = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            cov = ew_var(df[i], df[j], lambda_=lambda_)
            cov_m[i, j] = cov
    return cov_m


def spear_cor(df):
    n = df.shape[0]
    cor_spear = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
           res= spearmanr(df[i], df[j])
           cor_spear[i, j] = res.correlation
    return cor_spear





def get_cor(cov, var):
    inv_std = np.diag(np.sqrt(1/var))
    cor = inv_std @ cov @ inv_std
    return cor

def get_cov(cor, var):
    std = np.diag(np.sqrt(var))
    cov =std @ cor @ std
    return cov

