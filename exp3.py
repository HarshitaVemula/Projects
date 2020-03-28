#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:55:13 2020

@author: harshita
"""

# beta ols is used for actual beta
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import Ridge
from collections import defaultdict

#np.random.seed(0)
n = 50
num_para=10

corr_dict=defaultdict()
beta_actual=np.random.randint(1,10,num_para)

#bias_square=B.T@A@B
#A is obtained using the fucntion bias_square_A
def bias_square_A(U,D,V,y,lambda_):
    D_2=D**2
    d=np.diag(lambda_**2/((D_2+lambda_)**2))
    #print(d)
    A=V@d@V.T
    return(A)
    
def bias_square_A1(U,D,V,y,lambda_):
     D_2=D**2
     d=np.diag(2*lambda_*D_2/((D_2+lambda_)**3))
     A_1=V@d@V.T
     return(A_1)

def bias_square_A2(U,D,V,y,lambda_):
     D_2=D**2
     d=np.diag(4*D_2*(D_2-lambda_)/((D_2+lambda_)**4))
     A_2=V@d@V.T
     return(A_2)   
    
def var_r_A(U,D,V,y,lambda_):
    D_2=D**2
    d=np.diag(D_2/((D_2+lambda_)**2))
    return(V@d@V.T)
    
def var_der1(U,D,V,y,lambda_):
    D_2=D**2
    d=np.diag(-2*D_2/((D_2+lambda_)**3))
    return(V@d@V.T)
    
def var_der2(U,D,V,y,lambda_):
    D_2=D**2
    d=np.diag(6*D_2/((D_2+lambda_)**4))
    return(V@d@V.T)
    
import math    
def newton_raphson(U,D,V,y,X):
    
    epsilon=0.001
    lambda_=1
    beta=np.linalg.inv(X.T@X)@X.T@y
    beta_lambda=np.append(beta,lambda_)
    
    delta=epsilon+1
    i=0
    var_y=np.var(y)
    while i<100:
        
#        yhat=X@beta.T
#        e=y-y_hat
#        median_=np.median(e)
#        var_y=(np.sum(np.abs(e-median_))/0.6745)**2
        #lambda_
        first_der=beta.T@bias_square_A1(U,D,V,y,lambda_)@beta+var_y*np.sum(np.diagonal(var_der1(U,D,V,y,lambda_)))
        second_der=beta.T@bias_square_A2(U,D,V,y,lambda_)@beta+var_y*np.sum(np.diagonal(var_der2(U,D,V,y,lambda_)))
        #print(first_der,second_der)
        #beta
        first_der_bias=bias_square_A1(U,D,V,y,lambda_)@beta
        second_der_bias=bias_square_A1(U,D,V,y,lambda_)
        #print(first_der_bias,second_der_bias)
        #print('\n\n')
        
        #der1=np.append(first_der_bias,first_der)
        #der2=np.append(second_der_bias,second_der)
        
        #grad=der1/der2
        #beta_lambda=beta_lambda-grad
        mse=beta.T@bias_square_A(U,D,V,y,lambda_)@beta+var_y*np.sum(np.diagonal(var_der1(U,D,V,y,lambda_)))
        #print('1')
        lambda_=lambda_-mse/first_der
        #print('2')
        beta=beta-mse/first_der_bias
        #print(beta.shape)
        #print(first_der_bias.shape)
        #print(mse.shape)
        #print('3')
        #print(np.linalg.inv(second_der_bias)@first_der_bias)
        i=i+1
 
    
    return(beta,lambda_)

data_size=50
cov=np.full((num_para,num_para),0.999)
cov=cov-np.identity(num_para)*0.999+np.identity(num_para)
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
W= np.zeros((100,num_para))
Q=np.zeros((100,num_para))
y1=np.matmul(X,beta_actual)
for i in range(100):
    print(i)
    y=y1+np.random.normal(0,1,n)    
    W[i,:],_=newton_raphson(U,D,V,y,X)
    Q[i,:]=np.linalg.inv(X.T@X)@X.T@y

m=np.mean(W,axis=0)
var=np.var(W,axis=0)
lambda_=np.mean(W,axis=0)

mse_rid=np.sum((m-beta_actual)**2)+np.sum(var)

m1=np.mean(Q,axis=0)
var1=np.var(Q,axis=0)
mse_ols=np.sum((m1-beta_actual)**2)+np.sum(var1)

