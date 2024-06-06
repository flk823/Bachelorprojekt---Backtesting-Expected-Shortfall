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
            ES = (mu-np.sqrt(sigma2)/(1-alpha)*norm.pdf(norm.ppf(1-alpha)))*scale+shift
        
        elif type == 't':
            x = t.ppf(1-alpha, df)
            ES = -np.sqrt((df-2)/df)*(t.pdf(x, df)/(1-alpha)*(df+x**2)/(df-1))*scale+shift
    
    return -ES

def shift(alpha, df=None, dfnull=100): #insert significance level, sigma^2, mu and degrees of freedom
    VaR1 = VaR(1-alpha,df = dfnull, Norm = False, type = 't')
    VaR2 = VaR(1-alpha,df = df, Norm = False, type = 't')
    res = VaR1-VaR2
    return res

def scale(signi = 0.05, df = 100, type = 'Normal'):
    res = ES(alpha = 0.975 , sigma2 = 1,mu = 0, df = df, type = type)/ES(alpha = 1-signi , sigma2 = 1,mu = 0, df = df, type = type)
    return res

def Z_1(quan, sigma2, mu, T, n, shift = 0,  df=None, dfnull = 100, type='Normal'):
    Z_1_list = []
    var = VaR(alpha = 0.975, sigma2 = sigma2, mu=mu, df = dfnull, type = type)
    es = ES(alpha = 0.975, sigma2= sigma2, mu= mu, df = dfnull, type = type)
    
    for _ in range(n):
        X_t = t.rvs(df, loc = shift, size=T)
        q=0
        P = var + X_t
        N_t = np.sum(P < 0)
        

        for i in range(T):
            if X_t[i]+var<0:
                q+= X_t[i]/es
            Z_1 = q/N_t+1
        Z_1_list.append(Z_1)
    
    Z_1_list = np.array(Z_1_list)
    Z_1_list = Z_1_list[~np.isnan(Z_1_list)]
    quant = np.quantile(Z_1_list,quan)
    
    return quant, Z_1_list, np.mean(Z_1_list)

def Z_2(quan, sigma2, mu, T, n, shift = 0, scale=1, df=None, dfnull = 100, Norm = False, type='Normal'):
    Z_2_list = []
    var = VaR(alpha = 0.975, sigma2 = sigma2, mu = mu, df = dfnull, type = type, Norm = Norm)
    es =  ES( alpha = 0.975, sigma2 = sigma2, mu = mu, df = dfnull, type = type, Norm = Norm)
    if Norm == False:
        for _ in range(n):
            X_t = t.rvs(df = df, size=T, scale = scale, loc = shift)
            q=0
            Talpha = T*0.025
            
            
            for i in range(T):
                if X_t[i]+var<0:
                    q+= (X_t[i]/es)
                Z_2 = q/(Talpha)+1
            Z_2_list.append(Z_2)
    
    elif Norm == True:
        for _ in range(n):
            X_t = t.rvs(df = df, size=T, scale = np.sqrt((df-2)/df), loc = shift)
            q=0
            Talpha = T*0.025
            
            
            for i in range(T):
                if X_t[i]+var<0:
                    q+= (X_t[i]/es)
                Z_2 = q/(Talpha)+1
            Z_2_list.append(Z_2)
    Z_2_list = np.array(Z_2_list)
    Z_2_list = Z_2_list[~np.isnan(Z_2_list)]
    quant = np.quantile(Z_2_list,quan)
    return quant , Z_2_list, np.mean(Z_2_list)

def Z_3(quan, T, n, df=None, dfnull=100, shift = 0, scale = 1, Norm = False):
    Z_3_list = []
    TAlpha = round(T*0.025)
    if Norm == False:
        integral = -T/TAlpha * integrate.quad(lambda p: sc.betainc(T-TAlpha, TAlpha, 1-p)*t.ppf(p, df=dfnull), 0, 1)[0]
        for _ in range(n):
            U = uniform.rvs(size=T)
            q=0
            PU = t.ppf(U, df=df, scale=scale, loc = shift)
            PU = np.sort(PU)
            for i in range(TAlpha):
                q+= PU[i]
            
            ES_hat = -1/TAlpha*q
            Z_3 = -ES_hat/integral+1
            
            Z_3_list.append(Z_3)
    
    if Norm == True:
        integral = -T/TAlpha * integrate.quad(lambda p: sc.betainc(T-TAlpha, TAlpha, 1-p)*t.ppf(p, scale = np.sqrt((dfnull-2)/dfnull), df=dfnull), 0, 1)[0]
        for _ in range(n):
            U = uniform.rvs(size=T)
            q=0
            PU = t.ppf(U, df=df, scale=np.sqrt((df-2)/df), loc = shift)
            PU = np.sort(PU)
            for i in range(TAlpha):
                q+= PU[i]
            
            ES_hat = -1/TAlpha*q
            Z_3 = -ES_hat/integral+1
            
            Z_3_list.append(Z_3)
    Z_3_list = np.array(Z_3_list)
    Z_3_list = Z_3_list[~np.isnan(Z_3_list)]
    quant = np.quantile(Z_3_list,quan)
    return quant , Z_3_list, np.mean(Z_3_list)

def VaR_backtest(sigma2, mu, T, n, shift = 0, scale=1, df=None, dfnull = 100, type='Normal', Norm = False):
    var = VaR(alpha = 0.99, sigma2 = sigma2, mu = mu, df = dfnull, type = type, Norm = Norm)
    var_back = []
    if Norm == False:
        for _ in range(n):
            X_t = t.rvs(df, loc = shift, scale=scale, size=T)
            exceedances = 0
            for i in range (T):
                if X_t[i] + var<0:
                    exceedances += 1
            var_back.append(exceedances)
    if Norm == True:
        for _ in range(n):
            X_t = t.rvs(df, loc = shift, scale=np.sqrt((df-2)/df), size=T)
            exceedances = 0
            for i in range (T):
                if X_t[i] + var<0:
                    exceedances += 1
            var_back.append(exceedances)
    
    var_back = np.array(var_back)
    var_back = var_back[~np.isnan(var_back)]
    return var_back, np.mean(var_back)

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
            plt.plot(sorted_data, 1-cdf_data, label=f'{name}', color=np.random.rand(3,))
        
        plt.axvline(x=self.crit_value, color='black', linestyle='--', label=f'Critical Value = {self.crit_value:.2f}')
        plt.title('Comparison of CDFs for Null and Alternative Hypotheses')
        plt.xlabel('Value')
        plt.ylabel('Cumulative Probability')
        plt.legend()
        plt.show()
        
        
