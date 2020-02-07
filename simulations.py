#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 08:58:26 2020

@author: harshita
"""

# beta ols is used for actual beta
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import Ridge
from collections import defaultdict

np.random.seed(0)
#parameters
data_size = 250
n=data_size
num_para=500

################################################################################## 

corr_dict=defaultdict()
beta_actual=np.random.randint(1,10,num_para)
for i in [0,0.2,0.4,0.6,0.8,0.9,0.99,0.999,0.9999,0.99999,0.999999]:#correlation values
    print(i)
    mean_=[]
    var_=[]
    
    bi=[]
    var_rid=[]
    
    mse_rid=[]
  
    #generating covariance matrix 
    cor=i
    cov=np.full((num_para,num_para),cor)
    cov=cov-np.identity(num_para)*cor+np.identity(num_para)
    mu=[0]*num_para
    
    #Sampling X from multivariate normal with the cov matrix generated above
    X=np.random.multivariate_normal(mu, cov, size=data_size)
    scaler=StandardScaler()
    model=scaler.fit(X)
    X=model.transform(X)
    c=np.corrcoef(X.T)

    U,D,V_t=np.linalg.svd(X,full_matrices=False) #SVD of X
    V=V_t.T
    D_2=D**2
    D_=np.diag(1/D)
    
    y1=np.matmul(X,beta_actual)

    for j in [0,1,2,3]:#lambda values
        beta_ols_lister=np.zeros((500,num_para))
        beta_rid=np.zeros((500,num_para))
        D1=np.diag(D/(D_2+j))
        print(j)
        for k in range(500):
       
            
            y=y1+np.random.normal(0,1,n)
            beta_ols_lister[k,:]=V@D_@U.T@y
            beta_rid[k,:]=V@D1@U.T@y
        
        mean_beta=np.mean(beta_rid,axis=0) #mean of all beta_ols generated across nsim simulations
        var_beta=np.var(beta_rid,axis=0) 
        bias_=mean_beta-beta_actual
        m=np.sum(bias_**2)+np.sum(var_beta)
        mse_rid.append(m)
    print('\n\n')   
    o=[mse_rid[0]/i for i in mse_rid]#precision
    
    corr_dict[i]=defaultdict()
    
    corr_dict[i]['mse_rid']=mse_rid
    corr_dict[i]['precision']=o
       
#to access value in the dict
#corrdict[i] has infomation abt mse rid, bias of each predictor, var of each predictor, sum of bias square, sum of var for different 
#values of lambda
plt.figure(figsize=(12,8))    
qw=[str(i) for i in [0,1,2,3]]
for i in [0,0.2,0.4,0.6,0.8,0.9,0.99,0.999,0.9999,0.99999,0.999999]:
    plt.plot(corr_dict[i]['precision'],'-o')
    print(corr_dict[i]['precision'])
locs,labels=plt.xticks()
plt.xticks([0,1,2,3])
plt.title('mse_ols/mse_rid')
plt.ylabel('precision')
plt.xlabel('lambda')
plt.legend([0,0.2,0.4,0.6,0.8,0.9,0.99,0.999,0.9999,0.99999,0.999999])
plt.show()       

plt.plot(corr_dict[0]['precision'])
plt.xticks([0,1,2,3])