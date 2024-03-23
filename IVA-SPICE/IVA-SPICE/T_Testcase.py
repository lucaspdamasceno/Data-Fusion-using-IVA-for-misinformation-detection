#!/usr/bin/env python
# coding: utf-8


import numpy as np
import scipy as sc
from sklearn.datasets import make_sparse_spd_matrix
from sparse_iva_g import iva_spice
from iva_g import iva_g
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def test_gen(N, K, T):
    
    global density
    global simulations
    
    avg_isi_g = 0
    avg_isi_spice = 0
        
    for i in range(simulations):
                
        name = "N="+str(int(N))+"K="+str(int(K))+"T="+str(int(T))+"density="+str(density)
        newname = name+"sim="+str(i+1)+".mat"
        
        data = sc.io.loadmat('matFiles/'+newname)
        S = data['S']
        A = data['A']
        X = data['X']
        wnit = data['wnit']

        # CALL iva
        W, c, sig, isi_g = iva_g(X, opt_approach='gradient', complex_valued=False, circular=False, whiten=True,
                               verbose=False, A=A, W_init=wnit, jdiag_initW=False, max_iter=512,
                               W_diff_stop=1e-6, alpha0=1.0)
        # CALL sparse IVA
        W, c, sig, isi_spice = iva_spice(X, whiten=True,
                  verbose=False, A=A, max_iter=512,
                  W_diff_stop=1e-6, alpha0=1.0)
        
        # Find sum of joint isi across simulations
        avg_isi_g = avg_isi_g + isi_g[-1]
        avg_isi_spice = avg_isi_spice + isi_spice[-1]
    
    return (avg_isi_g / simulations), (avg_isi_spice / simulations)
    
    
def increasingT(power, 2, N, K):
    
    global density
    global simulations

    plt.figure(figsize=(12,8))
    fig, ax = plt.subplots()
    
    plt.xlabel("T")
    plt.ylabel("Joint ISI")
        
    for T in (10**t for t in range(2, power+1)):
                
        joint_isi_g, joint_isi_spice = test_gen(N, K, T)
        
        if T > (10**2):
                ax.plot(T, joint_isi_g, '.', color='pink')
                ax.plot(T, joint_isi_spice, 's', color='orange')
                
                glc = LineCollection([[(T/(10), prev_isi_g), (T, joint_isi_g)]], color='pink', linestyles='dashed')
                slc = LineCollection([[(T/(10), prev_isi_spice),(T, joint_isi_spice)]], color='orange', linestyles='dashed')

                ax.add_collection(glc)
                ax.add_collection(slc)
        else:
            ax.plot(T, joint_isi_g, '.', color='pink', label='IVA-G')
            ax.plot(T, joint_isi_spice, 's', color='orange', label='IVA-SPICE')
                        
        prev_isi_g = joint_isi_g
        prev_isi_spice = joint_isi_spice
                        
    plt.legend(loc="lower left")
    plt.yscale('log', basey=(10))
    plt.xscale('log', basex=(10))
    fig.savefig('Increasing_T=%s_N=%s_K=%s_density=%s_simulations=%s.png' % (T,N,K, density, simulations), bbox_inches='tight')
    plt.close(fig)