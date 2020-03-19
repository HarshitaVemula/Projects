#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:00:16 2020

@author: harshita
"""

'''
1) using variance and bias derivations to compute precision. (as error is geenrated from N(0,1), var(y) is taken 
to be equal to 1)

2) precision is calculated using var(y)= mae and var(y)=var(train y)
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

def bias_square(U,D,V,y,lambda_):
    D_2=D**2
    d=np.diag(-lambda_/((D_2+lambda_)*D))
    d=d**2
    bias_=y.T@U@d@U.T@y
    return(bias_)
    
def var_r_A(D,lambda_):
    D_2=D**2
    return(np.sum(D_2/(D_2+lambda_)**2))
    
#case 1 var(y)=1
for i in [0,0.2,0.4,0.6,0.8,0.9,0.99,0.999,0.9999,0.99999,0.999999]:#correlation values
    mse=[]
  
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
    
    y=np.matmul(X,beta_actual)+np.random.normal(0,1,n)
    var_y=1

    for j in [0,1,10,100,1000]:#lambda values
        D1=np.diag(D/(D_2+j))
        
        beta_rid=V@D1@U.T@y
        y_hat=X@beta_rid.T
        
        var_beta=var_y*var_r_A(D,j)
        bias_sq=bias_square(U,D,V,y,j)
        m_=bias_sq+var_beta
        mse.append(m_)
        
  
    precision=[mse[0]/i for i in mse]#precision
    
    corr_dict[i]=defaultdict()
    
    corr_dict[i]['mse']=mse
    corr_dict[i]['precision']=precision


plt.figure(figsize=(12,8))    
qw=[str(i) for i in [0,1,10,100,1000]]
for i in [0.2,0.4,0.6,0.8,0.9,0.99,0.999,0.9999,0.99999,0.999999]:
    plt.plot(qw,corr_dict[i]['precision'],'-o')
    print(corr_dict[i]['precision'])
locs,labels=plt.xticks()
plt.xticks(qw)
plt.title('mse_ols/mse_rid')
plt.ylabel('precision')
plt.xlabel('lambda')
plt.legend([0.2,0.4,0.6,0.8,0.9,0.99,0.999,0.9999,0.99999,0.999999])
plt.show()    
 
#case 2 var(y)=mae
for i in [0,0.2,0.4,0.6,0.8,0.9,0.99,0.999,0.9999,0.99999,0.999999]:#correlation values
    mse=[]
  
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
    
    y=np.matmul(X,beta_actual)+np.random.normal(0,1,n)


    for j in [0,1,10,100,1000]:#lambda values
        D1=np.diag(D/(D_2+j))
        
        beta_rid=V@D1@U.T@y
        y_hat=X@beta_rid.T
        
        e=y-y_hat
        median_=np.median(e)
        var_y=(np.sum(np.abs(e-median_))/0.6745)**2
        
        var_beta=var_y*var_r_A(D,j)
        bias_sq=bias_square(U,D,V,y,j)
        m_=bias_sq+var_beta
        mse.append(m_)
        
  
    precision=[mse[0]/i for i in mse]#precision
    
    corr_dict[i]=defaultdict()
    
    corr_dict[i]['mse']=mse
    corr_dict[i]['precision']=precision


plt.figure(figsize=(12,8))    
qw=[str(i) for i in [0,1,10,100,1000]]
for i in [0.2,0.4,0.6,0.8,0.9,0.99,0.999,0.9999,0.99999,0.999999]:
    plt.plot(qw,corr_dict[i]['precision'],'-o')
    print(corr_dict[i]['precision'])
locs,labels=plt.xticks()
plt.xticks(qw)
plt.title('mse_ols/mse_rid')
plt.ylabel('precision')
plt.xlabel('lambda')
plt.legend([0.2,0.4,0.6,0.8,0.9,0.99,0.999,0.9999,0.99999,0.999999])
plt.show()    

#case 2 var(y)=var(ytrain)
for i in [0,0.2,0.4,0.6,0.8,0.9,0.99,0.999,0.9999,0.99999,0.999999]:#correlation values
    mse=[]
  
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
    
    y=np.matmul(X,beta_actual)+np.random.normal(0,1,n)
    var_y=np.var(y)

    for j in [0,1,10,100,1000]:#lambda values
        D1=np.diag(D/(D_2+j))
        
        beta_rid=V@D1@U.T@y
        y_hat=X@beta_rid.T
        
        var_beta=var_y*var_r_A(D,j)
        bias_sq=bias_square(U,D,V,y,j)
        m_=bias_sq+var_beta
        mse.append(m_)
        
  
    precision=[mse[0]/i for i in mse]#precision
    
    corr_dict[i]=defaultdict()
    
    corr_dict[i]['mse']=mse
    corr_dict[i]['precision']=precision


plt.figure(figsize=(12,8))    
qw=[str(i) for i in [0,1,10,100,1000]]
for i in [0.2,0.4,0.6,0.8,0.9,0.99,0.999,0.9999,0.99999,0.999999]:
    plt.plot(qw,corr_dict[i]['precision'],'-o')
    print(corr_dict[i]['precision'])
locs,labels=plt.xticks()
plt.xticks(qw)
plt.title('mse_ols/mse_rid')
plt.ylabel('precision')
plt.xlabel('lambda')
plt.legend([0.2,0.4,0.6,0.8,0.9,0.99,0.999,0.9999,0.99999,0.999999])
plt.show()      