#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 21:17:48 2020

@author: harshita
"""

'''
This code generates precision values i.e. mse_ols/mse_ridge for different values of correlation and lambda.

'''
# beta ols is used for actual beta
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import Ridge
from collections import defaultdict

np.random.seed(0)
n = 1000
num_para=250

corr_dict=defaultdict()
beta_actual=np.random.randint(1,10,num_para)

for i in [0,0.2,0.4,0.6,0.8,0.9,0.99,0.999,0.9999,0.99999,0.999999]:#correlation values
    mse=[]
    mse_actual=[]
  
    #generating covariance matrix 
    cov=np.full((num_para,num_para),i)
    cov=cov-np.identity(num_para)*cor+np.identity(num_para)
    mu=[0]*num_para
    
    #Sampling X from multivariate normal with the cov matrix generated above
    X=np.random.multivariate_normal(mu, cov, size=data_size)
    m=np.mean(X,axis=0)
    s=np.std(X,axis=0)
    X=(X-m)/s
#    
    U,D,V_t=np.linalg.svd(X,full_matrices=False) #SVD of X
    V=V_t.T
    D_2=D**2
    D_=np.diag(1/D)
    
    y1=np.matmul(X,beta_actual)

    for j in [0,1,2,3]:#lambda values
        beta_rid=np.zeros((500,num_para))
        D1=np.diag(D/(D_2+j))
        for k in range(500):
       
            y=y1+np.random.normal(0,1,n)
            beta_rid[k,:]=V@D1@U.T@y
            
            
        mean_beta=np.mean(beta_rid,axis=0) #mean of all beta_ols generated across nsim simulations
        var_beta=np.var(beta_rid,axis=0) 
        bias_=mean_beta-beta_actual
        m_=np.sum(bias_**2)+np.sum(var_beta)
        mse.append(m_)
        
        r1=beta_rid-beta_actual
        mse_actual.append(np.sum(r1**2)/500)
  
    precision=[mse[0]/i for i in mse]#precision
    
    corr_dict[i]=defaultdict()
    
    corr_dict[i]['mse']=mse
    corr_dict[i]['precision']=precision
    corr_dict[i]['mse_actual']=mse_actual


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

