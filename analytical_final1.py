#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 08:43:10 2020

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
data_size = 500
num_para=1000
lambda_=1
n=400
n_t=100

def bias(U,D,V,y,lambda_):
    D_2=D**2
    d=np.diag(-lambda_/((D_2+lambda_)*D))
    #print(d)
    bias_=V@d@U.T@y
    return(bias_)
    
def var_r_A(D,lambda_):
    D_2=D**2
    return(np.sum(D_2/(D_2+lambda_)**2))

################################################################################
#case 1 p<n
data_size = 1000
num_para=250

cor_dict=defaultdict()
beta_actual=np.random.randint(1,10,num_para)

for k in [0,0.2,0.4,0.6,0.8,0.9,0.99,0.999,0.9999,0.99999,0.999999]:#correlation

    cor=k
    cov=np.full((num_para,num_para),cor)
    cov=cov-np.identity(num_para)*cor+np.identity(num_para)
    mu=[0]*num_para

    X=np.random.multivariate_normal(mu, cov, size=data_size)

    scaler=StandardScaler()
    model=scaler.fit(X)
    X=model.transform(X)
    y=np.matmul(X,beta_actual)+np.random.normal(0,1,data_size)
    U,D,V_t=np.linalg.svd(X,full_matrices=False)
    V=V_t.T
    D_2=D**2

    bi=[]
    var_rid=[]
    
    mse_rid=[]
    mse_act=[] 
      
    mse=[]
    
    v1=[]
    v2=[]
    f=[]

    for i in [0,1,2,3]:#lambda
    
        d1=np.diag(D/(D_2+i))
        beta_hat_r=V@(d1@(U.T@y))
    
        y_hat=X@beta_hat_r
        e=y-y_hat
        
        var=np.var(e)
        median_=np.median(e)
        var_=(np.sum(np.abs(e-median_))/0.6745)**2
        
        mse.append(np.sum(e**2)/n)
        
        p1=var_*var_r_A(D,i)
        p2=np.sum(bias(U,D,V,y,i)**2)
        
        v1.append(var_)
        v2.append(var_r_A(D,i))
        
        mse_act.append(np.sum((beta_actual-beta_hat_r)**2))
        mse_rid.append(p1+p2)
        var_rid.append(p1)
        bi.append(p2)
        f.append(p2/p1)
        
    o=[mse_rid[0]/i for i in mse_rid]
    
    
    cor_dict[k]=defaultdict()
    cor_dict[k]['mse']=mse
    cor_dict[k]['mse_act']=mse_act
    cor_dict[k]['mse_rid']=mse_rid
    cor_dict[k]['bi']=bi
    cor_dict[k]['var_rid']=var_rid
    cor_dict[k]['v1']=v1
    cor_dict[k]['v2']=v2
    cor_dict[k]['bias/var']=f
    cor_dict[k]['o']=o

plt.figure(figsize=(12,8))    
qw=[str(i) for i in [0,0,1,10,100,200,500,1000,3000,100000]]
for i in [0.2,0.4,0.6,0.8,0.9,0.99,0.999,0.9999,0.99999,0.999999]:
    plt.plot(cor_dict[i]['o'],'-o')
locs,labels=plt.xticks()
plt.xticks([0,1,2,3])
plt.title('mse_ols/mse_rid')
plt.ylabel('precision')
plt.xlabel('lambda')
plt.legend([0.2,0.4,0.6,0.8,0.9,0.99,0.999,0.9999,0.99999,0.999999])
plt.show()

plt.figure(figsize=(12,8))
for i in [0.2,0.4,0.6,0.8,0.9,0.999,0.9999,0.9999999]: 
    plt.plot(cor_dict[i]['mse_rid'])
locs,labels=plt.xticks()
plt.xticks(locs,qw)
plt.title('mse_rid')
plt.ylabel('mse_rid')
plt.xlabel('lambda')
plt.legend([0.2,0.4,0.6,0.8,0.9,0.999,0.9999,0.9999999])
plt.show()


################################################################################
#gradient descent
#code for optimal lambda
#case 1 p<n
data_size=1000    
cor=0
num_para=250
cov=np.full((num_para,num_para),cor)
cov=cov-np.identity(num_para)*cor+np.identity(num_para)
mu=[0]*num_para

X=np.random.multivariate_normal(mu, cov, size=data_size)
U,D,V_t=np.linalg.svd(X,full_matrices=False)
V=V_t.T
beta_actual=np.random.randint(1,10,num_para)
y=np.matmul(X,beta_actual)+np.random.normal(0,1,data_size)

def bias(U,D,V,y,lambda_):
    D_2=D**2
    d=np.diag(-lambda_/((D_2+lambda_)*D))
    #print(d)
    bias_=V@d@U.T@y
    return(bias_)
    
def var_r_A(D,lambda_):
    D_2=D**2
    return(np.sum(D_2/(D_2+lambda_)**2))
    
def bias_der(U,D,V,y,lambda_):
     D_2=D**2
     d=np.diag(-D/((D_2+lambda_)**2))
     bias_d=V@d@U.T@y
     return(bias_d)
     
def var_der(D,lambda_):
    D_2=D**2
    return(np.sum(-2*D_2/(D_2+lambda_)**3))
    
import math    
def grad_des(U,D,V,y,X):
    lambda_=0
    alpha=0.001
    epsilon=0.001
    
    c=0
    d1=np.diag(D/((D**2)+c))
    beta_hat_r=V@(d1@(U.T@y))
    
    y_hat=X@beta_hat_r
    e=y-y_hat
    
    median_=np.median(e)
    var_=(np.sum(np.abs(e-median_))/0.6745)**2
    
    delta=1
    mse=[bias(U,D,V,y,lambda_).T@bias(U,D,V,y,lambda_)+var_*var_r_A(D,lambda_)]
    i=1
    lambda_list=[lambda_]
    j=0
    precision_list=[1]
    while delta>epsilon:
        j=j+1
        c=lambda_
        
        d1=np.diag(D/((D**2)+c))
        beta_hat_r=V@(d1@(U.T@y))
        y_hat=X@beta_hat_r
        e=y-y_hat
        
        median_=np.median(e)
        var_=(np.sum(np.abs(e-median_))/0.6745)**2
        #var_=np.var(e)
        
        b=2*bias(U,D,V,y,lambda_).T@bias_der(U,D,V,y,lambda_)
        v=var_der(D,lambda_)

        gr=b+var_*v #(did not diff var_with lambda)
        alpha=1/10**(int(math.log10(np.abs(gr))))
        lambda_=lambda_-alpha*(gr)
        lambda_list.append(lambda_)
        p1=bias(U,D,V,y,lambda_).T@bias(U,D,V,y,lambda_)
        p2=var_*var_r_A(D,lambda_)
        precision_list.append(mse[0]/(p1+p2))

        mse.append(p1+p2)
        #print('mse',bias(U,D,V,y,lambda_).T@bias(U,D,V,y,lambda_)+var_*var_r_A(D,lambda_))
    
        if mse[-2]-mse[-1]<=1:
            break
        i=i+1
        if i ==500:
            break
    #print('c',c)
    print('j',j)
    #print('\n\n')
   
    return(c,lambda_list,precision_list)


d,lister,pre_list=grad_des(U,D,V,y,X)
print('d',d)
print(lister)
print(pre_list)

d=[]  
p=[] 
for c in [0,0.1,0.5,0.6,0.7,0.8,0.9,0.99,0.999,0.9999]:
    cor=c
    num_para=250
    cov=np.full((num_para,num_para),cor)
    cov=cov-np.identity(num_para)*cor+np.identity(num_para)
    mu=[0]*num_para
    
    X=np.random.multivariate_normal(mu, cov, size=data_size)
    U,D,V_t=np.linalg.svd(X,full_matrices=False)
    V=V_t.T
    beta_actual=np.random.randint(1,10,num_para)
    y=np.matmul(X,beta_actual)+np.random.normal(0,1,data_size)
    d1,_,p1=grad_des(U,D,V,y,X)
    d.append(d1)
    p.append(p1[-2])
  
l=[0,0.1,0.5,0.6,0.7,0.8,0.9,0.99,0.999,0.9999]
for i in range(len(l)):
    print(l[i],d[i])