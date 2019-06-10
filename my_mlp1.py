#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:25:07 2018

@author: danica greetham

MLP forecast
"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


def transform_cat(x):
    if x == 'ww':
        return 0
    elif x == 'wh':
        return 1
    elif x == 'hw':
        return 2
    elif x == 'hh':
        return 3
    else:
        return 4

def mycustomer(data,C=1):
    "loads data for a specified customer, C"
    dat = data[['WeekT','DayN','DayW','HH','Season','DayC', 'F'+str(C)]]
    dat['DayC_new'] = data['DayC'].apply(lambda x: transform_cat(x))
    return dat

def scale(inp):
    values=pd.Series(inp).values.reshape(-1,1)
    values=values.astype('float32')
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled=scaler.fit_transform(values)
    return np.array(scaled)

    
def NPMSE(pred, act, p):
    sm=0
    nn=len(pred)
    for i in range(nn):
        sm+=math.pow(pred[i]-act[i], p)
       
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


def myMLP(data):
    
    rel = data[:-336]
    ii=str(rel.columns.values[6])
    y = np.array(rel[ii])
    hh=np.array(rel.HH)
    dayw = np.array(rel.DayW)
    ss=np.array(rel.Season)
    dayc=np.array(rel.DayC_new)

    X=np.column_stack((hh,dayw,ss,dayc))
# 
    mlp = MLPRegressor(hidden_layer_sizes=(9,9,9,9,9),
        activation='relu',random_state=1, solver='lbfgs',
        learning_rate='adaptive',max_iter=1000,learning_rate_init=0.01,alpha=0.1)
    mlp.fit(X, y)
    roll = data[-336:] 
    
  
    hh=np.array(roll.HH)
    dayw = np.array(roll.DayW)
    ss=np.array(roll.Season)
    dayc=np.array(roll.DayC_new)
    X_new=np.column_stack((hh,dayw,ss, dayc))

    pred_MLP = mlp.predict(X_new)
  
    return pred_MLP
  


 
def myfullMLP():
    "computes week long forecast for all customers"
    FULL = pd.read_csv("my_epm1.csv",header=0)
    FULL.DateTime = pd.to_datetime(FULL.DateTime,format='%d/%m/%Y %H:%M')
    FULL = FULL[(FULL.DateTime >= pd.to_datetime('19/07/2015',format='%d/%m/%Y')) &(FULL.DateTime < pd.to_datetime('21/09/2015',format='%d/%m/%Y'))]
  
    forecast = np.zeros((336,226))
    MAPE = np.zeros((226))
    MSE = np.zeros((226))
    MAE = np.zeros((226))
    MAD = np.zeros((226))
    for C in range(1,227):
        dat = mycustomer(FULL,C)
        pred_MLP = myMLP(dat)
        ii=str(dat.columns.values[6])
        OBS = np.array(dat[-336:][ii])
        MAPE[C-1] = NMAPE(pred_MLP,OBS)
        MSE[C-1] = NPMSE(pred_MLP,OBS,4)
        MAE[C-1] = NMAE(pred_MLP,OBS)
        MAD[C-1] = NMAD(pred_MLP,OBS)
        forecast[:,C-1] = pred_MLP
    np.savetxt('MLP_forecast.csv',forecast,delimiter=',')
    print('MAPE', np.mean(MAPE),'MAE',np.mean(MAE),'4th norm',np.mean(MSE),'MAD',np.mean(MAD))
    plt.hist(MSE)
    plt.show()
myfullMLP()
    
    


