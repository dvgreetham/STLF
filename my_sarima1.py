#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:25:07 2018

@author: danica greetham

SARIMA forecast
"""
#import statsmodels.api as sm
import numpy as np
import statsmodels as sm
import math
import pandas as pd


def myload_data(data,C=1):
     X = data[['DateTime', 'WeekT','DayW','Season','F'+str(C)]]
     return X
def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()
def split_data(data):
#    train = data[:23856]
#    test = data[23856:]
    train = data[:3025]
    test = data[3025:]
    return train,test



def myARWD_forecast(X, p,d,q, P, D, Q):

    ii=str(X.columns.values[4])  
    train=X[ii][:-336]
    m1 = sm.tsa.statespace.sarimax.SARIMAX(train, order=(p, d, q),
                             seasonal_order=(P, D, Q,48))
  
    rm1=m1.fit(maxiter=200, method='powell')#disp=False)
    rm1.plot_diagnostics()
    print(rm1.summary())
    m1pred=np.array(rm1.predict(n_periods=336))
    return m1pred[-336:]

def NPMSE(pred, act, p):
    sm=0
    ss=0
    nn=len(pred)
    for i in range(nn):
        sm+=math.pow(pred[i]-act[i], p)
        ss+=math.pow(act[i],p)
    #pMSE=math.pow(sm/ss, 1./p)
    pMSE=math.pow(sm, 1./p)
    return pMSE

def NMAPE(pred, act):
    serr=0
    nn=len(pred)
    for i in range(nn):
        serr+=abs(pred[i]-act[i])
    MAPEc=100*(serr/sum(act))
    return MAPEc

def NMAE(pred,act):
    MAEc = abs(pred - act).mean()
    return MAEc

def NMAD(pred,act):
    MADc = np.median(abs(pred - act))
    return MADc



def main():
    data = pd.read_csv("my_epm1.csv", header=0)
    data.DateTime = pd.to_datetime(data.DateTime,format='%d/%m/%Y %H:%M')
    data = data[(data.DateTime >= pd.to_datetime('20/07/2015',format='%d/%m/%Y')) &(data.DateTime < pd.to_datetime('21/09/2015',format='%d/%m/%Y'))]


    data.index=pd.date_range(freq='30 min', start='19/07/2015', periods=len(data))
    forecast = np.zeros((336,226))
    MAPE = np.zeros((226))
    MSE = np.zeros((226))
    MAE = np.zeros((226))
    MAD = np.zeros((226))
 
    for C in range(1, 227):# range(1:202, 203:226)
        X = myload_data(data,C)
        if C==33:
            forecast[:,C-1]=myARWD_forecast(X, 2,1,3, 1, 0, 0)
       
        elif C==54:
            forecast[:,C-1]=myARWD_forecast(X, 5, 1, 1, 2, 0, 1)
        elif C==55:
            forecast[:,C-1]=myARWD_forecast(X, 2, 1, 0, 2, 0, 0)
        elif C==136:
           forecast[:,C-1]=myARWD_forecast(X, 6, 1, 1, 2, 0, 1)
        elif C==179:
           forecast[:,C-1]=myARWD_forecast(X, 3,1,3, 1, 0, 1)
        elif C==202:
            forecast[:,C-1]=myARWD_forecast(X, 3,1,3, 2, 0, 2)
        else:
    
            forecast[:,C-1]=myARWD_forecast(X, 2,1,0, 1, 0, 1)
        ii=str(X.columns.values[4])
        OBS = np.array(X[-336:][ii])
        MAPE[C-1] = NMAPE(forecast[:,C-1],OBS)
        MSE[C-1] = NPMSE(forecast[:,C-1],OBS,4)
        MAE[C-1] = NMAE(forecast[:,C-1],OBS)
        MAD[C-1] = NMAD(forecast[:,C-1],OBS)
    
    forecast = pd.DataFrame(forecast,columns = data.columns[7:],index=data[-336:].index)
    np.savetxt('SARIMAforecast.csv',forecast,delimiter=',')
    print('MAPE', np.mean(MAPE),'MAE',np.mean(MAE),'4th norm',np.mean(MSE),'MAD',np.mean(MAD))

import time
start = time.time()
main()
end = time.time()
print('Time', end - start)
