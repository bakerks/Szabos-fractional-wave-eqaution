#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import netgen.gui
import math
import numpy as np
from ngsolve import *
from netgen.geom2d import SplineGeometry


# Weights for convolution quadrature:

# In[ ]:


def weights(N,T,Ms,Ks):
    
    
    dt = T/N
    if Ms == 0:
        gam = lambda z: 1-z #BDF1
    elif Ms == 1:
        gam = lambda z: 0.5 * (1-z) * (3-z) #BDF2
    elif Ms == 2:
        gam = lambda z: 2 * (1-z) / (1+z) #Trapezoidal
        
    
    m = np.arange(0,(4*N)+1,1)
    zs = np.exp(-1j * 2 * math.pi * (m) / (4*N+1))
    lam = 10 ** (-15 / (5*N))

    tmp = (np.fft.ifft(Ks(gam(lam * zs)/dt))).real
    
    p = np.arange(0,-N-1,-1)
    g = (lam ** p) * tmp[0:N+1]
  
    return g


# Correction weights $w_{n1}$ and $w_{n0}$:

# In[ ]:


def corr_weights1(n,gamma):
    if gamma<0:
        wn1 = 0    
    elif gamma>0:
        wj = 0
        for k in range (0,n+1):
            wj += stored_weights[n-k] * k
            
        wn1 = (dt**(-gamma)*(n)**(1-gamma))/math.gamma(2-gamma) - wj 
    
    return float(wn1)


# In[ ]:


def corr_weights0(n,gamma):
    
    w = 0
    for k in range (0,n+1):
        w += stored_weights[n-k] 
    
    if gamma<0:
        wn0 = (dt*n)**(-gamma)/math.gamma(1-gamma) - w
    else:
        wn0 = -corr_weights1(n,gamma) - w
    
    return wn0
    


# Gauss Quadrature:

# In[ ]:


def gauss(N):
    
    beta = 0.5 / np.sqrt(1-1/((2*np.arange(1,N))**(2)))

    T = np.diag(beta,1) + np.diag(beta,-1)
    
    D,V = np.linalg.eigh(T) #eigenvalues and eigenvectors
    
    j = np.argsort(D[:]) #index arrangement 
    D.sort(axis=0) #sort D
    
    w = 2*(V[0,j]**2)
    
    return (D,w)


# Gauss-Jacobi Quadrature:

# In[ ]:


def gaussj(n,alf,bet):

    apb = alf + bet

    a1 = (bet-alf)/(apb+2)
    N1 = np.arange(2,n+1)
    aN = (apb)*(bet-alf) / ((apb+2*N1)*(apb+2*N1-2))
    a = np.append(a1,aN)

    b1 = math.sqrt(4*(1+alf)*(1+bet) / ((apb+3)*(apb+2)**2))
    N2 = np.arange(2,n)
    bN = np.sqrt(4*N2*(N2+alf)*(N2+bet)*(N2+apb)/(((apb+2*N2)**2-1)*(apb+2*N2)**2))
    b = np.append(b1,bN)

    if n>1: 
        D,V = np.linalg.eigh(np.diag(a) + np.diag(b,1) + np.diag(b,-1))
    else:
        V = 1
        D = a
    
    c = 2**(apb+1)*math.gamma(alf+1)*math.gamma(bet+1)/math.gamma(apb+2)

    j = np.argsort(D[:]) #index arrangement 
    D.sort(axis=0) #sort D

    if type(V)==int:
        w = c * (V**2)
    else:
        w = c * (V[0,j]**2)

    return (D,w)


# Compute the fractional integral:

# In[ ]:


def simple_fint(t,g,alpha,**nq):
    #t = time
    #g = function
    #alpha = fractional integral power
    
    if nq is not None:
        nq = 40
    
    t = np.array([t])
    f = np.zeros(np.shape(t))
    (xj,wj) = gaussj(nq,alpha-1,0)
    (x,w) = gauss(nq)
    x = (x+1)/2
    w = w/2
    
    gv = np.vectorize(g)
    
    for j in range (0,len(t)):
        t1 = t[j]*3/4
        t1d = t[j]-t1
        f[j] = ((t1d/2)**alpha)*np.dot(wj,gv((xj+1)*t1d/2+t1))
        if t[j]>0:
            f[j] = f[j]+(t1/2)*np.dot(w,(gv(x*t1/2)*(t[j]-t1*x/2)**(alpha-1)))
            f[j] = f[j]+(t1/2)*np.dot(w,(gv((x+1)*t1/2)*(t[j]-t1*(x+1)/2)**(alpha-1)))
    f = (1/math.gamma(alpha))*f
    
    return f[0]

