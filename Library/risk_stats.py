import pandas as pd
import numpy as np
from scipy.stats import norm, t
from statsmodels.tsa.arima.model import ARIMA


from Library.variance import ew_cov, cum_var

def return_calculate(df, method="DISCRETE", dateColumn="Date"):
    """
    calculate return functions
    Dataframe:param df: daily price with dates
    str:param method: (Discrete, Classic, Log) return
    str:param dateColumn: the column name of the dates column
    Dataframe:return: Dataframe of calculated return
    """

    vars = list(df.columns.values)
    var_num = len(vars)
    if dateColumn not in vars:
        raise ValueError("dateColumn: " + str(dateColumn) + " not in DataFrame")
    vars.remove(dateColumn)

    var_num -= 1
    price = df[vars].values
    price_2 = price[1:] / price[:-1]

    if method.upper() == "DISCRETE":
        price_2 = price_2 - 1
    elif method.upper() == "LOG":
        price_2 = np.log(price_2)
    elif method.upper() == "CLASSIC":
        price_2 = price[1:] - price[:-1]
    else:
        raise ValueError("method: ", method, " must be in (\"LOG\",\"DISCRETE\")")

    dates = df[dateColumn].values[1:]
    result = pd.concat([pd.DataFrame({dateColumn: dates}), pd.DataFrame(columns=vars, data=price_2)], axis=1)
    return result


def var_calculate(x, alpha=0.05, method="NORMAL", lambda_=0.94):
    """
    Calculate VaR
    ndarray(n,1):param x data to calculate
    str:param method: (Normal, Normal_EW, T, AR_1, Historic)
    float:param alpha: alpha of VaR
    float:return: VaR value
    """

    if method.upper() == "NORMAL":
        mu = np.mean(x)
        sigma = np.std(x)
        return -norm.ppf(alpha, loc=mu, scale=sigma)
    elif method.upper() == "NORMAL_EW":
        sigma_exp = np.sqrt(cum_var(x,x,lambda_))
        mu = np.mean(x)
        return -norm.ppf(alpha, loc=mu, scale=sigma_exp)
    elif method.upper() == "T":
        df, mu, sigma = t.fit(x, method='mle')
        return -t.ppf(0.05, df, mu, sigma)
    elif method.upper() == "AR_1":
        AR_1 = ARIMA(x, order=(1, 0, 0))
        AR_1_fit = AR_1.fit()
        sigma = np.sqrt(AR_1_fit.params[2])
        return -norm.ppf(alpha, loc=0, scale=sigma)
    elif method.upper() == "HISTORIC":
        N = np.random.randint(len(x), size=int(len(x)*1.5))
        his_sim = x[N]

        return -np.quantile(his_sim, 0.05)
    else:
        raise ValueError("method: ", method, " must be in (\"NORMAL\",\"NORMAL_EW\", \"T\", \"AR_1\", \"Historic\")")


def es_calculate(x, alpha=0.05, method="NORMAL", lambda_=0.94, simN=100):
    """
    Calculate Expected Shortfall
    ndarray:param x:
    float:param alpha:
    str:param method:
    float:param lambda_: lambda for exponentially weights
    int:param simN:
    float:return: VaR, Expected Shortfall
    """
    if method.upper() == "NORMAL":
        mu = np.mean(x)
        sigma = np.std(x)
        sim = norm.rvs(loc=mu, scale=sigma, size=simN)
        var = np.quantile(sim,alpha)
        es = np.mean(sim[np.where(sim < var)])
        return -var,  -es
    elif method.upper() == "NORMAL_EW":
        sigma_exp = np.sqrt(cum_var(x, x, lambda_))
        mu = np.mean(x)
        sim = norm.rvs(loc=mu, scale=sigma_exp, size=simN)
        var = np.quantile(sim, alpha)
        return -var, -np.mean(sim[np.where(sim < var)])
    elif method.upper() == "T":
        df, mu, sigma = t.fit(x, method='mle')
        sim = t.rvs(df,loc=mu, scale=sigma, size=simN)
        var = np.quantile(sim, alpha)
        es = -np.mean(sim[np.where(sim < var)])
        return -var, es

    else:
        raise ValueError("method: ", method, " must be in (\"NORMAL\",\"NORMAL_EW\", \"T\")")





