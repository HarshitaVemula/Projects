#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 12:53:06 2020

@author: harshita
"""

# beta ols is used for actual beta
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import Ridge
from collections import defaultdict

#test    
data_size = 100
num_para=2

cor_dict=defaultdict()
beta_actual=np.random.randint(1,10,num_para)

cor=1
cov=np.full((num_para,num_para),cor)
cov=cov-np.identity(num_para)*cor+np.identity(num_para)
mu=[0]*num_para

X=np.random.multivariate_normal(mu, cov, size=data_size)
x_transpose_x=X.T@X
w=np.linalg.inv(x_transpose_x)


D,U=np.linalg.eig(x_transpose_x)
U,D1,V_t=np.linalg.svd(X)
V=V_t.T
x_transpose_x_inv=V@np.diag(D**-2)@V.T

def plotting(array_num):
    i = array_num.shape[0]
    for k in range(i):
        plt.plot(array_num[k,:])
#condition number of (XTX)inv
#case 1 without lambda
data_size = 1000
num_para=200
beta_actual=np.random.randint(1,10,num_para)

condition_number_dict={}
condition_number_dict['n>p']={}
condition_number_dict['n=p']={}
condition_number_dict['n<p']={}

condition_number_n_greater_p=np.zeros((10,8))
for i in range(10):
    c=[]
    for cor in [0,0.2,0.4,0.6,0.8,0.9,0.95,0.98]:
        cov=np.full((num_para,num_para),cor)
        cov=cov-np.identity(num_para)*cor+np.identity(num_para)
        mu=[0]*num_para
        
        X=np.random.multivariate_normal(mu, cov, size=data_size)
        #y=X@beta_actual+np.random.normal(0,1,data_size)
        
        U,S,V_t=np.linalg.svd(X)
        D=S**2
        XTXinv_eigen_values=1/D
        c.append(XTXinv_eigen_values[-1]/XTXinv_eigen_values[0])
    condition_number_n_greater_p[i,:]=c

plotting(condition_number_n_greater_p)

data_size = 1000
num_para=1000
beta_actual=np.random.randint(1,10,num_para)

condition_number=[]
condition_number_n_greater_p=np.zeros((10,8))
for i in range(10):
    c=[]
    for cor in [0,0.2,0.4,0.6,0.8,0.9,0.95,0.98]:
        cov=np.full((num_para,num_para),cor)
        cov=cov-np.identity(num_para)*cor+np.identity(num_para)
        mu=[0]*num_para
        
        X=np.random.multivariate_normal(mu, cov, size=data_size)
        #y=X@beta_actual+np.random.normal(0,1,data_size)
        
        U,S,V_t=np.linalg.svd(X)
        D=S**2
        XTXinv_eigen_values=1/D
        c.append(np.max(XTXinv_eigen_values)/np.min(XTXinv_eigen_values))
    condition_number_n_greater_p[i,:]=c
plotting(condition_number_n_greater_p)

    
data_size = 1000
num_para=1500
beta_actual=np.random.randint(1,10,num_para)

condition_number=[]
condition_number_n_greater_p=np.zeros((10,8))
for i in range(10):
    c=[]
    for cor in [0,0.2,0.4,0.6,0.8,0.9,0.95,0.98]:
        cov=np.full((num_para,num_para),cor)
        cov=cov-np.identity(num_para)*cor+np.identity(num_para)
        mu=[0]*num_para
        
        X=np.random.multivariate_normal(mu, cov, size=data_size)
        #y=X@beta_actual+np.random.normal(0,1,data_size)
        
        U,S,V_t=np.linalg.svd(X)
        D=S**2
        XTXinv_eigen_values=1/D
        c.append(np.max(XTXinv_eigen_values)/np.min(XTXinv_eigen_values))
       
    condition_number_n_greater_p[i,:]=c
plotting(condition_number_n_greater_p)
    
'''
without adding lambda when n=p the trend in condition number is not consistent.
but when n>p or n<p the condition number seems to increase with correlation
'''

'''
Now let us analyze this trend by using a positive lambda value.

Observations:
    As expected increasing the lambda value has decreased the condition number.
    
    1)what is the effect of condition number on var and bias of the estimates?
    
High variance will effect the interpretability of the linear models.

'''
#with lambda
lambda_=100
data_size = 1000
num_para=200
beta_actual=np.random.randint(1,10,num_para)

condition_number_dict={}
condition_number_dict['n>p']={}
condition_number_dict['n=p']={}
condition_number_dict['n<p']={}

condition_number_n_greater_p1=np.zeros((10,8))
for i in range(10):
    c=[]
    for cor in [0,0.2,0.4,0.6,0.8,0.9,0.95,0.98]:
        cov=np.full((num_para,num_para),cor)
        cov=cov-np.identity(num_para)*cor+np.identity(num_para)
        mu=[0]*num_para
        
        X=np.random.multivariate_normal(mu, cov, size=data_size)
        #y=X@beta_actual+np.random.normal(0,1,data_size)
        
        U,S,V_t=np.linalg.svd(X)
        D=S**2+lambda_
        XTXinv_eigen_values=1/D
        c.append(XTXinv_eigen_values[-1]/XTXinv_eigen_values[0])
    condition_number_n_greater_p1[i,:]=c
plotting(condition_number_n_greater_p1)    

lambda_=100
data_size = 1000
num_para=1000
beta_actual=np.random.randint(1,10,num_para)

condition_number_dict={}
condition_number_dict['n>p']={}
condition_number_dict['n=p']={}
condition_number_dict['n<p']={}

condition_number_n_greater_p1=np.zeros((10,8))
for i in range(10):
    c=[]
    for cor in [0,0.2,0.4,0.6,0.8,0.9,0.95,0.98]:
        cov=np.full((num_para,num_para),cor)
        cov=cov-np.identity(num_para)*cor+np.identity(num_para)
        mu=[0]*num_para
        
        X=np.random.multivariate_normal(mu, cov, size=data_size)
        #y=X@beta_actual+np.random.normal(0,1,data_size)
        
        U,S,V_t=np.linalg.svd(X)
        D=S**2+lambda_
        XTXinv_eigen_values=1/D
        c.append(XTXinv_eigen_values[-1]/XTXinv_eigen_values[0])
    condition_number_n_greater_p1[i,:]=c    
plotting(condition_number_n_greater_p1)      

lambda_=100
data_size = 1000
num_para=1500
beta_actual=np.random.randint(1,10,num_para)

condition_number_dict={}
condition_number_dict['n>p']={}
condition_number_dict['n=p']={}
condition_number_dict['n<p']={}

condition_number_n_greater_p1=np.zeros((10,8))
for i in range(10):
    c=[]
    for cor in [0,0.2,0.4,0.6,0.8,0.9,0.95,0.98]:
        cov=np.full((num_para,num_para),cor)
        cov=cov-np.identity(num_para)*cor+np.identity(num_para)
        mu=[0]*num_para
        
        X=np.random.multivariate_normal(mu, cov, size=data_size)
        #y=X@beta_actual+np.random.normal(0,1,data_size)
        
        U,S,V_t=np.linalg.svd(X)
        D=S**2+lambda_
        XTXinv_eigen_values=1/D
        c.append(XTXinv_eigen_values[-1]/XTXinv_eigen_values[0])
    condition_number_n_greater_p1[i,:]=c    
plotting(condition_number_n_greater_p1)  

'''
affect of condition number on bias and variance
'''    

'''
Bounds on mse

mse<= [(n*lambda^2)/((emin+lambda)^2*emin) + (p*emax/lambda^2)] *var(y)
var(y)=(sum(mod(error_i-median(error_i))))/0.6745
'''

'''
finding optimal lambda
'''
data_size = 1000
num_para=50

cor_dict=defaultdict()
beta_actual=np.random.randint(1,100,num_para)

cor=0.999
cov=np.full((num_para,num_para),cor)
cov=cov-np.identity(num_para)*cor+np.identity(num_para)
mu=[0]*num_para

X=np.random.multivariate_normal(mu, cov, size=data_size)
mean=np.mean(X,axis=0)
std=np.std(X,axis=0)
X=(X-mean)/std
y=X@beta_actual

mean_y=np.mean(y)
var_y=np.sum((y-mean_y)**2)
var_y=np.var(y)

#betas
s=X.T@X
inv_=np.linalg.inv(X.T@X)
det=np.linalg.det(s)
beta_ols=inv_@X.T@y
beta_rid=np.linalg.inv(X.T@X+0.01*np.identity(num_para))@X.T@y

def mse_beta(beta,betahat):
    e=beta-betahat
    s=np.sum(e**2)/len(e)
    return(s)
    
print(mse_beta(beta_actual,beta_ols))
print(mse_beta(beta_actual,beta_rid))

def bias(U,D,V,y,lambda_):
    D_2=D**2
    d=np.diag(-lambda_/((D_2+lambda_)*D))
    bias_=V@d@U.T@y
    return(bias_)
    
def var_r_A(D,lambda_):
    D_2=D**2
    return(np.sum(D_2/(D_2+lambda_)**2))
    
U,D,V_t=np.linalg.svd(X)

var=var_r_A(D,0.01)







