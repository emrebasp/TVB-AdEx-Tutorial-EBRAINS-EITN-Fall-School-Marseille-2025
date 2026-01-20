#!/usr/bin/env python
# coding: utf-8

# # SDE integrator

# In[ ]:


#export

# Initialization
import numpy as np
import scipy.stats as ss
from cell_library import get_neuron_params
from syn_and_connec_library import get_connectivity_and_synapses_matrix
from DiffOperator import DifferentialOperator
from NeuronConnectivity import ReformatSynParameters, LoadTransferFunctions
# import derivativesTransferFunctions

# import derivativesTransferFunctions

# def heaviside(x):
#     return 0.5*(1+np.sign(x))

# def double_gaussian(t, t0, T1, T2, amplitude):
    
#     return amplitude*(\
#                       np.exp(-(t-t0)**2/2./T1**2)*heaviside(-(t-t0))+\
#                       np.exp(-(t-t0)**2/2./T2**2)*heaviside(t-t0))


def TimeStepping(NRN1, NRN2, NTWK, T, dt, tstop, stimulus, vDrive, noiseAmplitude, initConditions):


    # amp = 5
    amp = 0
    t0 = 2.
    T1=0.01
    T2=0.2
    
    
    # tstop = 300 #end of simulation
    # tstop = 10 #end of simulation
    
    
    #take network parameters
    # M = get_connectivity_and_synapses_matrix(NTWK, SI_units=True)
    # params = get_neuron_params(NRN1, SI_units=True)
    M = get_connectivity_and_synapses_matrix(NTWK, SI_units=False)
    params = get_neuron_params(NRN1, SI_units=False)
    ReformatSynParameters(params, M)
    
    a, b, tauw = params['a'],\
    params['b'], params['tauw']
    Qe, Te, Ee = params['Qe'], params['Te'], params['Ee']
    Qi, Ti, Ei = params['Qi'], params['Ti'], params['Ei']
    Gl, Cm , El = params['Gl'], params['Cm'] , params['El']
    pconnec,Ntot,gei,ext_drive=params['pconnec'], params['Ntot'] , params['gei'],M[0,0]['ext_drive']



########## FROM JEN's NOTEBOOK###


    # 'g_L':10.0,
    # 'E_L_e':-63.0,
    # 'E_L_i':-65.0,
    # 'C_m':200.0,
    # 'b_e':60.0,
    # 'a_e':0.0,
    # 'b_i':0.0,
    # 'a_i':0.0,
    # 'tau_w_e':500.0,
    # 'tau_w_i':1.0,
    # 'E_e':0.0,
    # 'E_i':-80.0,
    # 'Q_e':1.5,
    # 'Q_i':5.0,
    # 'tau_e':5.0,
    # 'tau_i':5.0,
    # 'N_tot':10000,
    # 'p_connect_e':0.05,
    # 'p_connect_i':0.05,
    # 'g':0.2,
    # 'T':40.0,
    # 'P_e':[-0.0498, 0.00506, -0.025, 0.0014, -0.00041, 0.0105, -0.036, 0.0074, 0.0012, -0.0407],
    # 'P_i':[-0.0514, 0.004, -0.0083, 0.0002, -0.0005, 0.0014, -0.0146, 0.0045, 0.0028, -0.0153],
    # 'external_input_ex_ex':0.315*1e-3,
    # 'external_input_ex_in':0.000,
    # 'external_input_in_ex':0.315*1e-3,
    # 'external_input_in_in':0.000,
    # 'tau_OU':5.0,
    # 'weight_noise': 1e-4, #10.5*1e-5,
    # 'K_ext_e':400,
    # 'K_ext_i':0,

#######



    
    
    #IMPORTANT PARAMETERS
    
    
    #to change network size
    Ntot=10000
    
    
    # extinp=2.5
    # extinp=0.315
    extinp = vDrive
    

    
    # filesave='mean_field.npy'
    
    
    Ne=Ntot*(1-gei)
    Ni=Ntot*gei
    
    TF1, TF2 = LoadTransferFunctions(NRN1, NRN2, NTWK)
    
    t = np.arange(int(tstop/dt))*dt
    fe=0*t
    fi=0*t
    ww=0*t
    v2vec=0*t
    v3vec=0*t
    v4vec=0*t
    
    # extinpnuovo=double_gaussian(t, t0, T1, T2, amp)

    
    
    
    
    
    # ornstein=0
    # ornsteinin2=0
    # extinpnoisex1=0
    # extinpnoisex=0
    # ornsteinA=0.0002 #time decay ornstein process



    # Ornstein-Uhlenbeck (OU) process

    N = int(tstop/dt) # number of time steps
    paths = 1         # number of paths: an independent path is required for each AdEx system.
    tauOU = 5         # time scale of the OU process
    kappa = 1/tauOU   # convergence rate of the process
    theta = 0
    
    sigmaOU = 1 # or this one sigma = 1
    std_asy = np.sqrt(sigmaOU**2 / (2 * kappa))  # asymptotic standard deviation
    
    OU0 = 0
    OU = np.zeros((N, paths))
    OU[0, :] = OU0
    # np.random.normal(0, 1)
    W = ss.norm.rvs(loc=0, scale=1, size=(N - 1, paths))
    
    # Uncomment for Euler Maruyama
    # for t in range(0,N-1):
    #    X[t + 1, :] = X[t, :] + kappa*(theta - X[t, :])*dt + sigma * np.sqrt(dt) * W[t, :]
    
    std_dt = np.sqrt(sigmaOU**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))
    for tInstant in range(0, N - 1):
        OU[tInstant + 1, :] = theta + np.exp(-kappa * dt) * (OU[tInstant, :] - theta) + std_dt * W[tInstant, :]

    # sigma = 10.5
    # # sigma = 1e-4
    # OU = np.zeros(N)
    # tauOU = 1
    # # tauOU = 0.05
    # # tauOU = 0.005
    # # OU[1:-1] = sigma*(1/tauOU)*np.sqrt(dt)*np.random.normal(0, 1, N-2) # generate the intrinsic noise
    # OU[1:-1] = sigma*(1/tauOU)*np.random.normal(0, 1, N-2) # generate the intrinsic noise
    # # (1/T)*np.sqrt(dt)
    # # for t2 in range(0, N - 1):
    # #     OU[t2 + 1, :] = theta + np.exp(-kappa * dt2) * (OU[t2, :] - theta) + std_dt * W[t2, :]
    
    

    
    
    
    
    #initial conditions
    # fecont=.8
    # ficont=50
    # v2=0.5
    # v3=0.5
    # v4=0.5
    # wcont=fecont*b*tauw
    # fecont=0
    # ficont=0
    # v2=0
    # v3=0
    # v4=0
    # wcont=fecont*b*tauw





    # def dX_dt_scalar(X, t=0):
    #     exc_aff = vDrive
    #     inh_aff = exc_aff
    #     # pure_exc_aff = 
    #     pure_exc_aff = stimulus
    #     return DifferentialOperator(TF1, TF2,NRN1, NRN2, NTWK,  Ne=Ne, Ni=Ni, T=T)(X,\
    #                                              exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff,inh_fract=inh_fract)
    
    
    # X0 = [1., 30, .5, .5, .5,1.e-10,0.]
    X0 = initConditions
    X = np.zeros((len(t), len(X0)))
    for i in range(0,6):
        X[0][i]=X0[i]



    inh_fract=1.
    for i in range(len(t)-1):

        stdvexc=np.sqrt(X[i,0] )
        stdvinh=np.sqrt(X[i,1] )
        
        


        if(t[i]<t0):
            adjust=stimulus.max()
        else:
            adjust=0

        sigma = noiseAmplitude
        exc_aff_All = extinp + sigma*OU[i][0]
        pure_exc_aff_All = (stimulus[i]+0*adjust)
        


        
        # sigma_r=0.5
        # X[i+1,:] = sigma_r*np.random.normal(0, 1) + X[i,:] + (t[1]-t[0])*DifferentialOperator(TF1, TF2,NRN1, NRN2, NTWK, \
        #                                                                Ne=Ne, Ni=Ni)(X[i,:], exc_aff=exc_aff, pure_exc_aff=pure_exc_aff,inh_fract=inh_fract)

        X[i+1,:] = X[i,:] + (t[1]-t[0])*DifferentialOperator(TF1, TF2,NRN1, NRN2, NTWK, \
                                                                       Ne=Ne, Ni=Ni)(X[i,:], exc_aff=exc_aff_All, pure_exc_aff=pure_exc_aff_All,inh_fract=inh_fract)
            
            
            
        # for j in range(0,6):
        #     if X[i+1][j]<0:
                
        #         X[i+1][j]=1.e-9


        # for j in range(0,6):
        #     if X[i+1][j]>175.:
        #         X[i+1][j]=175.

        for j in range(0,6):
            if X[i+1][j]<0:
                
                X[i+1][j]=1.e-9


        for j in range(0,6):
            if X[i+1][j]>200:
                X[i+1][j]=200
        
        

        X[i+1,6] = 0
    if vDrive<1:
        fe, fi = 0.1*X[:,0], 0.1*X[:,1] # scale for up-down state
    else:
        fe, fi = X[:,0], X[:,1]
    
    sfe, sfei, sfi = [np.sqrt(X[:,i]) for i in range(2,5)]
    XXe,XXi= X[:,5], X[:,6]

    return np.array([fe, fi])

