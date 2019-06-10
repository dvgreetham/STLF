#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:25:07 2018

@author: danica greetham

LSTM forecast
"""
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM 

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
    #ss=0
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
    MAEc = np.mean(abs(pred - act))
    return MAEc

def NMAD(pred,act):
    MADc = np.median(abs(pred - act))
    return MADc


def myANN(data, OBS):
    
    rel = data[:-336]
    ii=str(rel.columns.values[6])
  
    y = np.array(rel[ii])

    hh=np.array(rel.HH)
    dayw = np.array(rel.DayW)
 
    dayc=np.array(rel.DayC_new)

    X=np.column_stack((hh,dayw, dayc))
    train_X = X.reshape((X.shape[0], 1, X.shape[1]))
   
  

   
    
    print(train_X.shape)
    
    roll = data[-336:] 
    
  
    hh=np.array(roll.HH)
    dayw = np.array(roll.DayW)

    dayc=np.array(roll.DayC_new)
    
    X_new=np.column_stack((hh,dayw, dayc))
  
    test_X = X_new.reshape((X_new.shape[0], 1, X_new.shape[1]))
    model = Sequential()
    model.add(LSTM(20, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(LSTM(20, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    history = model.fit(train_X, y, epochs=5, batch_size=48, validation_data=(test_X, OBS), verbose=2, shuffle=False)

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
   

    pred_ANN = model.predict(test_X)
    pred_ANN = pred_ANN[:,0]
    return pred_ANN
  


 
def myfullANN():
    "computes week long forecast for all customers"

    FULL = pd.read_csv("my_epm1.csv",header=0)
    FULL.DateTime = pd.to_datetime(FULL.DateTime,format='%d/%m/%Y %H:%M')
    FULL = FULL[(FULL.DateTime >= pd.to_datetime('19/07/2015',format='%d/%m/%Y')) &(FULL.DateTime < pd.to_datetime('21/09/2015',format='%d/%m/%Y'))]
   # FULL = FULL[(FULL.DateTime >= pd.to_datetime('19/07/2015',format='%d/%m/%Y')) &(FULL.DateTime < pd.to_datetime('07/09/2015',format='%d/%m/%Y'))]

    forecast = np.zeros((336,226))
    MAPE = np.zeros((226))
    MSE = np.zeros((226))
    MAE = np.zeros((226))
    MAD = np.zeros((226))
    for C in range(1,227):
        dat = mycustomer(FULL,C)
        ii=str(dat.columns.values[6])
        
        OBS = np.array(dat[-336:][ii])
        pred_ANN = myANN(dat, OBS)
      
        MAPE[C-1] = NMAPE(pred_ANN,OBS)
        MSE[C-1] = NPMSE(pred_ANN,OBS,4)
        MAE[C-1] = NMAE(pred_ANN,OBS)
        MAD[C-1] = NMAD(pred_ANN,OBS)
        forecast[:,C-1] = pred_ANN
        print(C)
    np.savetxt('ANN_20-20-forecast-new.csv',forecast,delimiter=',')
    print('MAPE', np.mean(MAPE),'MAE',np.mean(MAE),'4th norm',np.mean(MSE),'MAD',np.mean(MAD))
    plt.hist(MSE)
    plt.show()
myfullANN()
    
    



