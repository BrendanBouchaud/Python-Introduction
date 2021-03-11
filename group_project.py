# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

                 Group Project - Python
                     FEDEX Group (FDX)

            Created on Sat Dec 12 14:58:17 2020


                        @authors:
                            
                   - Brendan BOUCHAUD
                   - Thomas GRIMAULT
                   - Elias ESSAADOUNI
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#%% Modules

import numpy as np
import math as mt
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr
import scipy.stats as stats

#%% Functions

def location(tickers, event):
    loc = np.where(tickers["Date"] == event) #in the next 3 rows we gonna define the location of our event
    loc = loc[0] #switch from tupple to array 
    return loc[0]

def data_extraction(ticker,year1, year2, event,frame_s, frame_e):
    start = "01/01/" + str(year1)
    end = "31/12/" + str(year2) 
    index = pdr.get_data_yahoo(ticker,start,end)
    index.reset_index(inplace = True) #set up the index
    loc = location(index, event)
    start_frame = loc - frame_s - 1
    end_frame = loc + frame_e + 1
    return index.iloc[start_frame:end_frame]

def stock_return(tickers): #return function 
    return (tickers["Adj Close"]-tickers.shift(periods=1)["Adj Close"])/tickers.shift(periods=1)["Adj Close"]

def mean_return(tickers,start,end): #mean calculation for a sample
    return tickers["return"].iloc[start:end].mean()

def std_return (tickers,start,end):
    return tickers["return"].iloc[start:end].std()

def concat_window_return(stock, index,stock_ticker,index_ticker, start,end):
    window_index_stock = pd.concat([index["return"].iloc[start:end], stock["return"].iloc[start:end] ], axis=1, sort=False)#%% concatenate 
    window_index_stock.columns =[index_ticker, stock_ticker]    
    return window_index_stock

def coef_regression(ticker_name, index_name, window_index_stock):
    formule = ticker_name + " ~ 1 + " + index_name
    ols_reg = smf.ols(formula= formule, data = window_index_stock).fit() 
    return pd.read_html(ols_reg.summary().tables[1].as_html(),header=0,index_col=0)[0]

def hypothesis_test(stat,confidence_level):
    np.random.seed(100)
    # Gaussian Percent Point Function
    # define probability
    p = 1 - confidence_level
    # retrieve value <= probability
    value = stats.norm.ppf(p)
    if value < stat:       #we reject HO
          return value, "The CAR of the event window is not significantly different from 0"
    else:      #We fail to reject H0    
          return value, "The CAR of the event window is significanty different from 0"   

#%% 1 - 7 - Main Data

ticker = "FDX"
sp500 = "^GSPC"
event_nature = "Bank of America Merrill Lynch downgrade to neutral from buy (TP $220 from $304)"
ticker_name = "FEDEX"
index_name = "SP500"
event_date = "12/10/2018" 
max_frame = 5
min_frame = 120 
confidence_level = 0.95

index = data_extraction(sp500, 2018, 2019, event_date, min_frame, max_frame) #download SP500 data from yahoo
stock = data_extraction(ticker, 2018, 2019, event_date, min_frame, max_frame) #download FEDEX  tickers : FDX data from yahoo

t = location(stock, event_date) #find our t = event_date_index


#%% 8 - Plot stock evolution

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

ax1.plot(stock["Date"], stock['Adj Close'])
ax2.bar(stock["Date"], stock['Volume'], color = 'red')

plt.xlabel("Volume")
plt.suptitle(ticker_name + " stock prices")
plt.show()


#%% 9  - Return calculation for SP500 and FEDEX
"""
applied our mean function for the index and stock 
"""

index["return"] = stock_return(index) 
stock["return"] = stock_return(stock)


#%% 10 - Control window as [t-120, t-6]
"""
Where "t" is the event date. 
Compute the average returns and the volatility over the control window.
Control date is t = 121
"""

control_w_start = t - min_frame
control_w_end = t - 5

control_mean_stock = (mean_return(stock, control_w_start, control_w_end))
control_std_stock = (std_return(stock, control_w_start, control_w_end))
control_mean_index = (mean_return(index, control_w_start, control_w_end))
control_std_index = (std_return(index, control_w_start, control_w_end))


#%% 11 - Event window as [t, t+5]
"""
Where "t" is the event date. 
Compute the average returns and the volatility over the event window
"""

event_w_start = t
event_w_end = t + max_frame + 1

event_mean_stock = (mean_return(stock, event_w_start, event_w_end))
event_std_stock = (std_return(stock, event_w_start, event_w_end))
event_mean_index = (mean_return(index, event_w_start, event_w_end))
event_std_index = (std_return(index, event_w_start, event_w_end))


#%% 12 - Regression for control window as [t-120,t-6]

control_SP500_FDX = concat_window_return(stock, index, ticker_name, index_name, control_w_start, control_w_end)
coef = coef_regression(ticker_name, index_name, control_SP500_FDX)

beta = coef['coef'].values[1]
alpha = coef['coef'].values[0]


#%% 13 - Compute the abnormal returns of event window by substracting the expected returns using market model 

event_SP500_FDX = concat_window_return(stock, index, ticker_name, index_name, event_w_start, event_w_end)
event_SP500_FDX['abnormalreturns'] = event_SP500_FDX[ticker_name] - ( alpha + beta * event_SP500_FDX[index_name])


#%% 14 - Cumulative Abnormal returns

car = event_SP500_FDX['abnormalreturns'].sum()


#%% 15 - Stat statistics caculation

control_SP500_FDX['abnormalreturns'] = control_SP500_FDX[ticker_name]-( alpha + beta * control_SP500_FDX[index_name])# calculation of control window abdnormal returns
abnormalreturns_std = control_SP500_FDX['abnormalreturns'].std() # calculation of control window abnormal returns standard deviation
stat = car/(mt.sqrt(6)*abnormalreturns_std)


#%% 16
    
"""
H0 : CAR = 0, H1 CAR <> 0
Left tail test:
Test Statistic >= Critical Value: Fail to reject the null hypothesis of the statistical test.
Test Statistic < Critical Value: Reject the null hypothesis of the statistical test.
"""
value, test_result = hypothesis_test(stat, confidence_level)


#%%Presentation of the company

"""
Display result
"""


print("Corporation : ", ticker_name)
print("Ticker      : ", ticker)
print("Event nature: ", event_nature)
print("Event date  :", event_date)


print("\n===================")
print("\n======= \nSummary \n======= \n ")
print("Control Window \n--------------- \n")

#Display Control Window data

print("Average return tickers:FDX    :", round(control_mean_stock*100,5),"%")
print("Volatility tickers:FDX        :", round(control_std_stock*100,5),"%")
print("Average return tickers:S&P500 :", round(control_mean_index*100,5),"%")
print("Volatility tickers:S&P500     :", round(control_std_index*100,5),"%")
print("alpha 𝛼ොestimated             :", alpha)
print("beta 𝛽 estimated              :", beta)

#Display Event Window data

print("\nEvent Window \n--------------- \n")
print("Average return tickers:FDX    :", round(event_mean_stock*100,5),"%")
print("Volatility tickers:FDX        :", round(event_std_stock*100,5),"%")
print("Average return tickers:S&P500 :", round(event_mean_index*100,5),"%")
print("Volatility tickers:S&P500     :", round(event_std_index*100,5),"%")
print("cumulative abnormal returns   :", round(car*100,5),"%")
print("Volatility of Abnormal return :", round(abnormalreturns_std*100,5),"%")
print("T-Stat Value                  :", round(stat,5))
print("Critical Value (CI=95%)       :", round(value,5))
print("CARew significantly = 0 or <>0:", test_result)
