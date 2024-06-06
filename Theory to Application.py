import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import random
import pandas as pd
import scipy.special as sc
import scipy.integrate as integrate
from scipy.stats import uniform
from scipy.stats import norm
from statsmodels.stats.power import TTestIndPower
from scipy.stats import gaussian_kde
from arch import arch_model
from random import gauss
from scipy.special import gamma
from scipy.stats import binom

def VaR(alpha = 0.975, sigma2 = 1, mu = 0, shift = 0, scale=1, df=None, type='Normal' , Norm = False):
    if Norm == False:
        if type == 'Normal':
            VaR = mu+np.sqrt(sigma2)*norm.ppf(1-alpha)

        elif type == 't':
            VaR = t.ppf(1-alpha, df, loc=shift, scale = scale)
    elif Norm == True:
        if type == 'Normal':
            VaR = norm.ppf(1-alpha)

        elif type == 't':
            VaR = np.sqrt((df-2)/df)*t.ppf(1-alpha, df, loc=shift, scale = scale)
    return -VaR

def ES(alpha = 0.975, sigma2 = 1, mu = 0, shift = 0, scale = 1, df=None, type='Normal' , Norm = False):
    if Norm == False:
        if type == 'Normal':
            ES = (mu-np.sqrt(sigma2)/(1-alpha)*norm.pdf(norm.ppf(1-alpha)))*scale+shift
        
        elif type == 't':
            x = t.ppf(1-alpha, df)
            ES = -(t.pdf(x, df)/(1-alpha)*(df+x**2)/(df-1))*scale+shift
    
    elif Norm == True:
        if type == 'Normal':
            ES = 1/(1-alpha)*norm.pdf(norm.ppf(1-alpha))*scale+shift
        
        elif type == 't':
            x = t.ppf(1-alpha, df)
            ES = np.sqrt((df-2)/df)*(t.pdf(x, df)/(1-alpha)*(df+x**2)/(df-1))*scale+shift
    
    return ES

def DataGen(returns_H_0, returns_H_1, n, alpha, H_0_model = 'garch', H_1_model = 'garch'):
    T = len(returns_H_1)
    R_t_list = []
    es_list = []
    var_list = []
    H_0_var_list = []
    if H_0_model == 'garch':
        H_0 = arch_model(returns_H_0.values, mean = 'zero', vol=H_0_model, p=1, q=1, dist = 't')
        H_0_fitted = H_0.fit(disp='off')
    elif H_0_model == 'egarch':
        H_0 = arch_model(returns_H_0.values, mean = 'zero', vol=H_0_model, p=1, q=1, o=1, dist = 't')
        H_0_fitted = H_0.fit(disp='off')

    
    if H_1_model == 'garch':
        H_1  = arch_model(returns_H_1.values, mean = 'zero', vol = H_1_model, p=1, q=1, dist = 't')
        H_1_fitted = H_1.fit(disp='off')
    elif H_1_model == 'egarch':
        H_1  = arch_model(returns_H_1.values, mean = 'zero', vol = H_1_model, p=1, q=1, o=1, dist = 't')
        H_1_fitted = H_1.fit(disp='off')
    elif H_1_model == 'normal':
        MA_H_1 = []
        for i in range(len(returns_H_1)):
            MA_H_1_i = sum(returns_H_1[0:i])/(len(returns_H_1[0:i])+1)
            MA_H_1.append(MA_H_1_i)
        
        MVAR_H_1 = [np.var(returns_H_1)]
        for i in range(len(returns_H_1)-1):
            MVAR_H_1_i = 0.95*MVAR_H_1[i]+0.05*returns_H_1[i]**2
            MVAR_H_1.append(MVAR_H_1_i)
            
    for _ in range(n):
        if H_0_model == 'garch' or H_0_model == 'egarch':
            H_1_forecast = H_1_fitted.forecast(horizon=T, reindex=False, start = T-1, method='simulation', simulations=1)
            R_t = H_1_forecast.simulations.values[0][0]
            dfnull = H_0_fitted.params['nu']
            H_0_var = [H_0_fitted.conditional_volatility[-1]**2]
            if H_0_model == 'garch':
                for i in range(len(R_t)-1):
                    variance = H_0_fitted.params['omega'] + H_0_fitted.params['alpha[1]'] * R_t[i]**2 + H_0_fitted.params['beta[1]'] * H_0_var[i]
                    H_0_var.append(variance)
            elif H_0_model == 'egarch':
                absEZ_t = (2 / np.sqrt(np.pi)) * (gamma((dfnull + 1) / 2) / gamma(dfnull / 2)) * np.sqrt((dfnull - 2)) / (dfnull - 1)
                for i in range(T-1):
                    variance = H_0_fitted.params['omega'] + H_0_fitted.params['alpha[1]']*(np.abs(R_t[i])/np.sqrt(H_0_var[i])-absEZ_t) + H_0_fitted.params['beta[1]']*np.log(H_0_var[i])+H_0_fitted.params['gamma[1]']*R_t[i]/np.sqrt(H_0_var[i])
                    H_0_var.append(np.exp(variance))
            
            var = VaR(alpha=0.975, df=dfnull, type='t', Norm=True) * np.array(np.sqrt(H_0_var))
            es = ES(alpha=0.975, df=dfnull, type='t', Norm=True) * np.array(np.sqrt(H_0_var))
        
        elif H_0_model == 'normal':
            if H_1_model == 'garch' or H_1_model == 'egarch':
                H_1_forecast = H_1_fitted.forecast(horizon=T, reindex=False, start = T-1, method='simulation', simulations=1)
                R_t = H_1_forecast.simulations.values[0][0]
            elif H_1_model == 'normal':
                R_t = np.array(MA_H_1)+norm.rvs(size=T)*np.array(np.sqrt(MVAR_H_1))
            
            H_0_MA = []
            for i in range(len(returns_H_1)):
                MA_H_0_i = sum(R_t[0:i])/(len(R_t[0:i])+1)
                H_0_MA.append(MA_H_0_i)
        
            H_0_var = [np.var(returns_H_0)]
            for i in range(len(returns_H_1)-1):
                MVAR_H_0_i = 0.95*H_0_var[i]+0.05*R_t[i]**2
                H_0_var.append(MVAR_H_0_i)
            var = np.array(H_0_MA) + VaR(alpha=alpha, type='Normal', Norm=True)*np.array(np.sqrt(H_0_var))
            es =  np.array(H_0_MA) + ES (alpha=alpha, type='Normal', Norm=True)*np.array(np.sqrt(H_0_var))

        
        R_t_list.append(R_t)
        es_list.append(es.tolist())
        var_list.append(var.tolist())
        H_0_var_list.append(H_0_var)
        
    return [R_t_list , es_list , var_list , H_0_var_list]


def VaR_backtest(R_t, var, n):
    var_back = []
    for j in range(n):
        exceedances = sum(R_t[j] + var[j]<0)
        var_back.append(exceedances)
    
    var_back = np.array(var_back)
    var_back = var_back[~np.isnan(var_back)]
    return var_back, np.mean(var_back)

def Z_1_ES(R_t, es, var, n):
    T = len(R_t[0])
    Z_1_list = []
    for j in range(n):
        q=0
        N_t = sum(R_t[j] + var[j]<0)
        conf_level = [binom.ppf(0.05,n = T, p = 0.025),binom.ppf(0.95,n = T, p = 0.025)]
        conf_level_l = conf_level[0]
        conf_level_u = conf_level[1]

        for i in range(T):
            if N_t>=conf_level_l and N_t<=conf_level_u:
                if R_t[j][i]+var[j][i]<0:
                    q+= R_t[j][i]/es[j][i]
                Z_1 = q/N_t+1
            else:
                Z_1 = np.nan
            
        Z_1_list.append(Z_1)
            
    Z_1_list = np.array(Z_1_list)
    Z_1_list = Z_1_list[~np.isnan(Z_1_list)]
    
    return Z_1_list, np.mean(Z_1_list)

def Z_2_ES(R_t, es, var, n):
    Z_2_list = []
    T = len(R_t[0])
    for j in range(n):
        q = 0
        Talpha = T * 0.025
            
        for i in range(T):
            if R_t[j][i] + var[j][i] < 0:
                q += (R_t[j][i] / es[j][i])
        Z_2 = q / Talpha + 1
        Z_2_list.append(Z_2)

    return Z_2_list, np.mean(Z_2_list)

def NZ_ES_one(R_t, es, var, H_0_var, alpha, n):
    T_2_list = []
    T = len(R_t[0])
    Index_list = []
    pvcond_list = []
    for j in range(n):
        xr_1 = np.zeros(T)
        for i in range(T):
            if R_t[j][i]>var[j][i]:
                xr_1[i]=1
        
        
        identi_t_1 = 1-alpha-xr_1
        identi_t_2 = np.array(var[j]) - np.array(es[j]) - 1/(1-alpha)*xr_1*(np.array(var[j])-np.array(R_t[j]))
        
        Omega = 0
        Z_t = 0
        for i in range(T):
            V_t = np.array([identi_t_1[i],identi_t_2[i]])
            h_t = np.array([[1,0],[np.abs(var[j][i]),0],[0,1],[0,np.sqrt(H_0_var[j][i])**(-1)]])
            omega_t = np.outer(h_t@V_t,h_t@V_t)
            Omega += omega_t
            Z_t += h_t@V_t
        
        Omega = Omega/T
        Z_t_bar = Z_t/T
        q = 4
        
        T_2 = np.sqrt(T)*np.sqrt(np.diag(Omega))**(-1)*Z_t_bar
        pi = np.sort(1 - norm.cdf(T_2))
        cq = np.sum(1 / np.arange(1, q + 1))
        pi_div = pi / np.arange(1, q + 1)
        min_index = np.argmin(pi_div)
        pvcond = min(q * cq * np.min(min_index), 1)
        
        T_2_list.append(T_2)
        pvcond_list.append(pvcond)
        Index_list.append(min_index)
        
    return T_2_list, np.mean(T_2_list) , pvcond_list, Index_list

class Power:
    def __init__(self, data_null, data_alternatives, significance_level, alternative_names=None):
       
        self.data_null = data_null
        self.data_alternatives = data_alternatives
        self.significance_level = significance_level
        self.alternative_names = alternative_names if alternative_names else [f'Alternative {i+1}' for i in range(len(data_alternatives))]
        self.sorted_null = np.sort(data_null)
        self.crit_value = self.compute_critical_value()

    def CDF(self, data):
        data_sorted = np.sort(data)
        return np.arange(1, len(data_sorted) + 1) / len(data_sorted)

    def compute_critical_value(self):
        cdf_0 = self.CDF(self.data_null)
        crit_value_index = np.argmax(cdf_0 >= self.significance_level)
        return self.sorted_null[crit_value_index]

    def compute_powers(self):
        powers = {'Significance Level': self.significance_level}
        for name, data in zip(self.alternative_names, self.data_alternatives):
            power = np.sum(data <= self.crit_value) / len(data)
            powers[name] = power
        return pd.DataFrame([powers])

    def plot_cdfs(self):
        plt.figure(figsize=(10, 6))
        cdf_0 = self.CDF(self.data_null)
        plt.plot(self.sorted_null, cdf_0, label='CDF of Null Data', color='blue')

        for name, data in zip(self.alternative_names, self.data_alternatives):
            sorted_data = np.sort(data)
            cdf_data = self.CDF(data)
            plt.plot(sorted_data, 1-cdf_data, label=f'1-CDF of {name}', color=np.random.rand(3,))
        
        plt.axvline(x=self.crit_value, color='black', linestyle='--', label=f'Critical Value = {self.crit_value:.2f}')
        plt.title('Comparison of CDFs for Null and Alternative Hypotheses')
        plt.xlabel('Value')
        plt.ylabel('Cumulative Probability')
        plt.legend()
        plt.show()
