#!/usr/bin/env python
# coding: utf-8

# # Load neuron transfer functions and connectivity

# This notebook contains the functions which we use to load the transfer functions of RS and FS cells by using the method explained in [1]. The transfer functions and their parameters are based on a fitting to experimental data, therefore the parameters should be kept fixed. DO NOT CHANGE THIS NOTEBOOK as long as you want to use the transfer functions obtained via the method explained in [1].

# In[ ]:


#export

# Initialize

import numpy as np
from syn_and_connec_library import get_connectivity_and_synapses_matrix
from cell_library import get_neuron_params
from theoretical_tools import pseq_params,TF_my_templateup


# In[ ]:


#export

# Define the transfer functions of RS and FS cells

def ReformatSynParameters(params, M):
    """
        valid only of no synaptic differences between excitation and inhibition
        """
    
    params['Qe'], params['Te'], params['Ee'] = M[0,0]['Q'], M[0,0]['Tsyn'], M[0,0]['Erev']
    params['Qi'], params['Ti'], params['Ei'] = M[1,1]['Q'], M[1,1]['Tsyn'], M[1,1]['Erev']
    params['pconnec'] = M[0,0]['p_conn']
    params['Ntot'], params['gei'] = M[0,0]['Ntot'], M[0,0]['gei']

def LoadTransferFunctions(NRN1, NRN2, NTWK):

    
    # NTWK
    M = get_connectivity_and_synapses_matrix(NTWK, SI_units=True)
    
    # NRN1
    params1 = get_neuron_params(NRN1, SI_units=True)
    ReformatSynParameters(params1, M)
    try:
        
        
        P1 = np.load('data/RS-cell_CONFIG1_fit.npy')
        
        
        params1['P'] = P1
        def TF1(fe, fi,XX):
            return TF_my_templateup(fe, fi,XX, *pseq_params(params1))
    
    
    
    
    
    
    except IOError:
        print('=======================================================')
        print('=====  fit for NRN1 not available  ====================')
        print('=======================================================')

    # NRN1
    params2 = get_neuron_params(NRN2, SI_units=True)
    ReformatSynParameters(params2, M)
    try:
        
        P2 = np.load('data/FS-cell_CONFIG1_fit.npy')
        
        
        
        params2['P'] = P2
        def TF2(fe, fi,XX):
            return TF_my_templateup(fe, fi,XX, *pseq_params(params2))

    except IOError:
        print('=======================================================')
        print('=====  fit for NRN2 not available  ====================')
        print('=======================================================')
    
    return TF1, TF2




# # Bibliography
# 
# [1] Y. Zerlaut, A. Destexhe, A mean-field model for conductance-based networks of adaptive exponential integrate-and-fire neurons,
