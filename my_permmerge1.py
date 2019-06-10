#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:25:07 2018

@author: danica greetham

PM forecast
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx

import itertools
import math
def mycustomer(data,C=1):

    dat = data[['WeekT','DayN','DayW', 'F'+str(C)]]
    return dat


    
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




def pm(dat, no_weeks):
    wholePM=[]
    for day in range(1,8):
        l=list(dat.columns)
        dat
        ii=l[3]
        #set up past observations
        G1 = dat.loc[dat.DayW == day][ii]

        G=np.zeros((48, no_weeks))
       # G[:,0]=G1[-48:]
        for i in range(0, no_weeks):
            
            G[:,i]=G1[-48*(i+2):-48*(i+1)]
       
      
    
        
        N, M = G.shape 
  
        Gmed=np.median(G, axis=1)
       # Gmed=np.zeros((N,1))
        #print(Gmed)
        Gr=nx.DiGraph()
       
    
    #     
    # =============================================================================
    # Create a graph  - n  is the number of historical profiles used,k=0..n, each  node 
    #(k, x, x, ..., x) where x can be 0 or 1,  1 on the position j means permutation
    # of k-1 and k position in the profile j, 0 means no permutation (We assume that
    # w=1, i.e. only adjacent positions can be swapped )
    # =============================================================================
        n=no_weeks
        l=list(itertools.product(range(2), repeat=n)) #create all nodes
        ll=list(itertools.product(range(1,N), l))
        Gr.add_nodes_from(ll)    
        
        lzeros=list(itertools.product(range(1), repeat=n))
        lones=list(itertools.product(range(1,2), repeat=n))
       
        Gr.add_node((0, lzeros[0]))
        Gr.add_node((N, lzeros[0]))
        
        for k in range(N):
            sum_weight_0=0
            for i in range(n):
                sum_weight_0=sum_weight_0+abs(Gmed[k]-G[k, i])
                Gr.add_edge((k,lzeros[0]), (k+1,lzeros[0]), weight=sum_weight_0)
              
        for k in range(1,N-1):
            sum_weight_1=0
            sum_weight_2=0
            for i in range(n):
                sum_weight_2=sum_weight_2+abs(Gmed[k-1]-G[k, i])
                sum_weight_1=sum_weight_1+abs(Gmed[k]-G[k-1, i])
         
            Gr.add_edge((k,lones[0]), (k+1,lzeros[0]), weight=sum_weight_2)
            Gr.add_edge((k,lzeros[0]), (k+1,lones[0]), weight=sum_weight_1)
    
        sum_weight_2=sum_weight_0=0
        for i in range(n):
             sum_weight_2=sum_weight_2+abs(Gmed[N-2]-G[N-1, i])
             sum_weight_0=sum_weight_0+abs(Gmed[1]-G[0, i])
        Gr.add_edge((N-1,lones[0]), (N,lzeros[0]), weight=sum_weight_2)
        Gr.add_edge((0,lzeros[0]), (1,lones[0]), weight=sum_weight_0)
        for i in l:
                ia=np.array(i)
                if np.count_nonzero(ia)<n:  
                    sum_weight_0=sum_weight_1=0
                    for j in range(n):
                        if i[j]==1:
                            sum_weight_0=sum_weight_0+ abs(Gmed[0]-G[1, j])
                            sum_weight_1=sum_weight_1+ abs(Gmed[N-1]-G[N-2, j])
                        else:
                          
                            sum_weight_0=sum_weight_0+abs(Gmed[0]-G[0, j])
                            sum_weight_1=sum_weight_0+abs(Gmed[N-1]-G[N-1, j])
                    Gr.add_edge((0,lzeros[0]), (1,i), weight=sum_weight_0)
                    Gr.add_edge((N-1,i), (N, lzeros[0]), weight=sum_weight_1)
                 
                    
                  
       
        for k in range(1,N-1):   
            for i in l:
                ia=np.array(i)
                
                if np.count_nonzero(ia)<n:
                    sum_weight_0=sum_weight_1=0
                    for j in range(n):
                        if i[j]==1:
                            sum_weight_1=sum_weight_1+ abs(Gmed[k-1]-G[k, j])
                            sum_weight_0=sum_weight_0+ abs(Gmed[k]-G[k-1, j])
                        else:
                            sum_weight_0=sum_weight_0+ abs(Gmed[k]-G[k, j])
                            sum_weight_1=sum_weight_1+abs(Gmed[k]-G[k, j])
                    Gr.add_edge((k,lzeros[0]), (k+1,i), weight=sum_weight_0)
                    Gr.add_edge((k, i), (k+1, lzeros[0]), weight=sum_weight_1)
                    
                    for ii in l:
                         sum_weight_0=sum_weight_1=0
                        
                         if np.count_nonzero(np.logical_and(i,ii))==0:
                                 for j in range(n):
                                     if i[j]==1:
                                         sum_weight_1=sum_weight_1+ abs(Gmed[k-1]-G[k, j])
                                         sum_weight_0=sum_weight_0+ abs(Gmed[k]-G[k-1, j])
                                     else:
                                         sum_weight_0=sum_weight_0+ abs(Gmed[k]-G[k, j])
                                         sum_weight_1=sum_weight_1+abs(Gmed[k]-G[k, j])
                                 Gr.add_edge((k, i), (k+1, ii), weight=sum_weight_0)
                                 Gr.add_edge((k, ii), (k+1, i), weight=sum_weight_1)
    
      
        sum_weights=0
        p=nx.dijkstra_path(Gr, source=(0,lzeros[0]), target=(N,lzeros[0]), weight='weight')
       
      
        PM1=np.zeros((N,1))
       
        for k in range(0, N):
       
            sum_weights=sum_weights + Gr.edges[p[k], p[k+1]]['weight']
            for i in range(n):
                if p[k][1][i]==0:
                    PM1[k]=PM1[k]+G[k,i]
                    
                else:
                    PM1[k-1]=PM1[k-1]-G[k-1, i]+G[k,i]
                    PM1[k]=PM1[k]+G[k-1, i]
         
        PM1=PM1/n
    
        
   
        Gr.clear
        PM=PM1
        wholePM.extend(PM)


    ppm=np.hstack(wholePM)
  
    return ppm[-336:]



def main():
   
   
    FULL = pd.read_csv("my_epm1.csv",header=0) #,skiprows=[1])
    FULL.DateTime = pd.to_datetime(FULL.DateTime,format='%d/%m/%Y %H:%M')
    FULL = FULL[(FULL.DateTime >= pd.to_datetime('19/07/2015',format='%d/%m/%Y')) &(FULL.DateTime < pd.to_datetime('21/09/2015',format='%d/%m/%Y'))]

    forecast = np.zeros((336,226))
    MAPE = np.zeros((226))
    MSE = np.zeros((226))
    MAE = np.zeros((226))
    MAD = np.zeros((226))
    for C in range(1,227):
        dat = mycustomer(FULL,C)
        pred_pm = pm(dat, 4)
        
        ii=str(dat.columns.values[3])
        OBS = np.array(dat[-336:][ii])
        MAPE[C-1] = NMAPE(pred_pm,OBS)
        MSE[C-1] = NPMSE(pred_pm,OBS,4)
        MAE[C-1] = NMAE(pred_pm,OBS)
        MAD[C-1] = NMAD(pred_pm,OBS)
        forecast[:,C-1] = pred_pm

    np.savetxt('PM_4_forecast-new1.csv',forecast,delimiter=',')
    print('MAPE', np.mean(MAPE),'MAE',np.mean(MAE),'4th norm',np.mean(MSE),'MAD',np.mean(MAD))
    plt.hist(MSE)
    plt.show()
     

main()
