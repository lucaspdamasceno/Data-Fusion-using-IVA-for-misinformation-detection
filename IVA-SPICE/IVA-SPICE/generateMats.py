#!/usr/bin/env python
# coding: utf-8

# Generates .mat files for each testcase.

# In[2]:


import matlab.engine
import numpy as np
import os

# Runs gen.m with the given testcase
def test_gen(N, K, T):

    global density
    global simulations

    for i in range(simulations):
        print("density = ", density)
        S, A, X, wnit, Inv = eng.gen(float(N), float(K), float(T), density, i+1, nargout=5)

# Vary K from 4 to 2^(power) and hold N, T constant
def increasingK(power, N, T):

    for K in (2.0**k for k in range(2, power+1)):
        test_gen(N,K,T)

# Vary T from 100 to 10^(power) and hold N, K constant
def increasingT(power, N, K):
    
    for T in (10.0**t for t in range(2, power+1)):
        test_gen(N,K,T)

# Vary N from 10 to 10*(mult) and hold K, T constant
def increasingN(multiplier, K, T):  

    for N in (10.0*n for n in range(1, multiplier+1)):
        test_gen(N,K,T)
    

# Start matlab engine in current working directory
eng = matlab.engine.start_matlab()

# Define number of simulations, density
simulations = 1
density = 0.5

# Call functions
#increasingK(6,10,1000)

# Close engine
eng.quit

