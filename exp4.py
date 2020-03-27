#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:20:30 2020

@author: harshita
"""

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
n = 25
num_para=5

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
def grad_des(U,D,V,y,X):
    lambda_=0
    beta=np.linalg.inv(X.T@X)@X.T@y
    #print('beta',beta)
    i=0
    var_y=np.var(y)
    epsilon=10^(-9)
    while i<1000:
        
        first_der=beta.T@bias_square_A1(U,D,V,y,lambda_)@beta+var_y*np.sum(np.diagonal(var_der1(U,D,V,y,lambda_)))
        second_der=beta.T@bias_square_A2(U,D,V,y,lambda_)@beta+var_y*np.sum(np.diagonal(var_der2(U,D,V,y,lambda_)))
     
        mse=beta.T@bias_square_A(U,D,V,y,lambda_)@beta+var_y*np.sum(np.diagonal(var_r_A(U,D,V,y,lambda_)))
        g=first_der/second_der
        lambda_=lambda_-g
        #print(mse)
        
#        print('l',lambda_)
#        print('first der',first_der)
#        print('sc',second_der)
        i=i+1

        
        if g < epsilon:
            break
        
        
        
    #print('/n/n')
    beta_lambda=np.linalg.inv(X.T@X+np.identity(X.shape[1])*lambda_)@X.T@y
    return(beta_lambda,lambda_,i)

data_size=n
cov=np.full((num_para,num_para),0.9999)
cov=cov-np.identity(num_para)*0.9999+np.identity(num_para)
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
 
W= np.zeros((100,num_para))
Q=np.zeros((100,num_para))
y1=np.matmul(X,beta_actual)
mse_rid_data=[]
mse_ols_data=[]
lam=[]
for i in range(100):
    print(i)
    y=y1+np.random.normal(0,1,n)
    b1,_,j=    grad_des(U,D,V,y,X)
    print('iterations',j)
    lam.append(_)
    W[i,:]=b1
    b2=np.linalg.inv(X.T@X)@X.T@y
    Q[i,:]=b2
    
    yhat_r=X@b1
    yhat=X@b2
    
    mse_rid_data.append(np.sum((y-yhat_r)**2)/len(y))
    mse_ols_data.append(np.sum((y-yhat)**2)/len(y))

m=np.mean(W,axis=0)
var=np.var(W,axis=0)
lambda_=np.mean(W,axis=0)

mse_rid=np.sum((m-beta_actual)**2)+np.sum(var)

m1=np.mean(Q,axis=0)
var1=np.var(Q,axis=0)
mse_ols=np.sum((m1-beta_actual)**2)+np.sum(var1)

