#!/usr/bin/env python
# coding: utf-8

# # AdEx double pool SDE system

# In[4]:


#export

# Initialization
import numpy as np
from syn_and_connec_library import get_connectivity_and_synapses_matrix
from cell_library import get_neuron_params
from NeuronConnectivity import ReformatSynParameters


# ## Derivatives of transfer functions with respect to firing rates

# In[5]:



##### Derivatives taken numerically,
## to be implemented analitically ! not hard...

def diff_fe(TF, fe, fi,XX, df=1e-5):
    return (TF(fe+df/2., fi,XX)-TF(fe-df/2.,fi,XX))/df

def diff_fi(TF, fe, fi,XX, df=1e-5):
    return (TF(fe, fi+df/2.,XX)-TF(fe, fi-df/2.,XX))/df

def diff2_fe_fe(TF, fe, fi,XX, df=1e-5):
    return (diff_fe(TF, fe+df/2., fi,XX)-diff_fe(TF,fe-df/2.,fi,XX))/df

def diff2_fi_fe(TF, fe, fi,XX, df=1e-5):
    return (diff_fi(TF, fe+df/2., fi,XX)-diff_fi(TF,fe-df/2.,fi,XX))/df

def diff2_fe_fi(TF, fe, fi,XX, df=1e-5):
    return (diff_fe(TF, fe, fi+df/2.,XX)-diff_fe(TF,fe, fi-df/2.,XX))/df

def diff2_fi_fi(TF, fe, fi,XX, df=1e-5):
    return (diff_fi(TF, fe, fi+df/2.,XX)-diff_fi(TF,fe, fi-df/2.,XX))/df




# def build_up_differential_operator_first_order(TF1, TF2, NRN1,NRN2,NTWK, T=5e-3):
    
#     M = get_connectivity_and_synapses_matrix(NTWK, SI_units=True)
#     params = get_neuron_params(NRN1, SI_units=True)
#     reformat_syn_parameters(params, M)
    
#     a, b, tauw = params['a'],\
#         params['b'], params['tauw']
#     Qe, Te, Ee = params['Qe'], params['Te'], params['Ee']
#     Qi, Ti, Ei = params['Qi'], params['Ti'], params['Ei']
#     Gl, Cm , El = params['Gl'], params['Cm'] , params['El']
#     pconnec,Ntot,gei,ext_drive=params['pconnec'], params['Ntot'] , params['gei'],M[0,0]['ext_drive']




#     def A0(V, exc_aff=0, inh_aff=0, pure_exc_aff=0):
#         return 1./T*(TF1(V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff,V[2])-V[0])
    
#     def A1(V, exc_aff=0, inh_aff=0, pure_exc_aff=0,inh_fract=0):
#         return 1./T*(TF2(V[0]+exc_aff+inh_fract*pure_exc_aff, V[1]+inh_aff,V[3])-V[1])
    
#     def A2(V, exc_aff=0, inh_aff=0, pure_exc_aff=0):
        
#         fe = (V[0]+exc_aff+pure_exc_aff)*(1.-gei)*pconnec*Ntot # default is 1 !!
#         fi = V[1]*gei*pconnec*Ntot
#         muGe, muGi = Qe*Te*fe, Qi*Ti*fi
#         muG = Gl+muGe+muGi
#         muV = (muGe*Ee+muGi*Ei+Gl*El-V[2])/muG
        

        
#         return (-V[2]/tauw+(b)*V[0]+a*(muV-El)/tauw)
    

    
#     def A3(V, exc_aff=0, inh_aff=0, pure_exc_aff=0):
#         return  (-V[3]/1.0+0.*V[1])
    
#     def Diff_OP(V, exc_aff=0, inh_aff=0, pure_exc_aff=0,inh_fract=0):
#         return np.array([A0(V, exc_aff=exc_aff,inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),\
#                          A1(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff,inh_fract=inh_fract),\
#                          A2(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),\
#                          A3(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff)])
#     return Diff_OP


def DifferentialOperator(TF1, TF2, NRN1,NRN2,NTWK,\
                                   Ne=8000, Ni=2000, T=5e-3):

    
    M = get_connectivity_and_synapses_matrix(NTWK, SI_units=True)
    params = get_neuron_params(NRN1, SI_units=True)
    ReformatSynParameters(params, M)
    
    a, b, tauw = params['a'],\
        params['b'], params['tauw']
    Qe, Te, Ee = params['Qe'], params['Te'], params['Ee']
    Qi, Ti, Ei = params['Qi'], params['Ti'], params['Ei']
    Gl, Cm , El = params['Gl'], params['Cm'] , params['El']
    pconnec,Ntot,gei,ext_drive=params['pconnec'], params['Ntot'] , params['gei'],M[0,0]['ext_drive']



    def A0(V, exc_aff=0, inh_aff=0, pure_exc_aff=0):
    
        return 1./T*(\
                 .5*V[2]*diff2_fe_fe(TF1, V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff,V[5])+\
                 .5*V[3]*diff2_fe_fi(TF1, V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff,V[5])+\
                 .5*V[3]*diff2_fi_fe(TF1, V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff,V[5])+\
                 .5*V[4]*diff2_fi_fi(TF1, V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff,V[5])+\
                 TF1(V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff,V[5])-V[0])
        
    def A1(V, exc_aff=0, inh_aff=0, pure_exc_aff=0,inh_fract=0):
                     
        return 1./T*(\
                  .5*V[2]*diff2_fe_fe(TF2, V[0]+exc_aff+inh_fract*pure_exc_aff, V[1]+inh_aff,V[6])+\
                  .5*V[3]*diff2_fe_fi(TF2, V[0]+exc_aff+inh_fract*pure_exc_aff, V[1]+inh_aff,V[6])+\
                  .5*V[3]*diff2_fi_fe(TF2, V[0]+exc_aff+inh_fract*pure_exc_aff, V[1]+inh_aff,V[6])+\
                  .5*V[4]*diff2_fi_fi(TF2, V[0]+exc_aff+inh_fract*pure_exc_aff, V[1]+inh_aff,V[6])+\
                  TF2(V[0]+exc_aff+inh_fract*pure_exc_aff, V[1]+inh_aff,V[6])-V[1])

    def A2(V, exc_aff=0, inh_aff=0, pure_exc_aff=0):
    
        return 1./T*(\
                 1./Ne*TF1(V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff,V[5])*(1./T-TF1(V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff,V[5]))+\
                 (TF1(V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff,V[5])-V[0])**2+\
                 2.*V[2]*diff_fe(TF1, V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff,V[5])+\
                 2.*V[3]*diff_fi(TF1, V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff,V[5])+\
                 -2.*V[2])
        
    def A3(V, exc_aff=0, inh_aff=0, pure_exc_aff=0,inh_fract=0): # mu, nu = e,i, then lbd = e then i
                     
        return 1./T*(\
                  (TF1(V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff,V[5])-V[0])*(TF2(V[0]+exc_aff+inh_fract*pure_exc_aff, V[1]+inh_aff,V[6])-V[1])+\
                  V[2]*diff_fe(TF2, V[0]+exc_aff+inh_fract*pure_exc_aff, V[1]+inh_aff,V[6])+\
                  V[3]*diff_fe(TF1, V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff,V[5])+\
                  V[3]*diff_fi(TF2, V[0]+exc_aff+inh_fract*pure_exc_aff, V[1]+inh_aff,V[6])+\
                  V[4]*diff_fi(TF1, V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff,V[5])+\
                  -2.*V[3])

    def A4(V, exc_aff=0, inh_aff=0, pure_exc_aff=0,inh_fract=0):
    
        return 1./T*(\
                 1./Ni*TF2(V[0]+exc_aff+inh_fract*pure_exc_aff, V[1]+inh_aff,V[6])*(1./T-TF2(V[0]+exc_aff+inh_fract*pure_exc_aff, V[1]+inh_aff,V[6]))+\
                 (TF2(V[0]+exc_aff+inh_fract*pure_exc_aff, V[1]+inh_aff,V[6])-V[1])**2+\
                 2.*V[3]*diff_fe(TF2, V[0]+exc_aff+inh_fract*pure_exc_aff, V[1]+inh_aff,V[6])+\
                 2.*V[4]*diff_fi(TF2, V[0]+exc_aff+inh_fract*pure_exc_aff, V[1]+inh_aff,V[6])+\
                 -2.*V[4])
        
    def A5(V, exc_aff=0, inh_aff=0, pure_exc_aff=0):
        fe = (V[0]+exc_aff+pure_exc_aff)*(1.-gei)*pconnec*Ntot # default is 1 !!
        fi = V[1]*gei*pconnec*Ntot
        muGe, muGi = Qe*Te*fe, Qi*Ti*fi
        muG = Gl+muGe+muGi
        muV = (muGe*Ee+muGi*Ei+Gl*El-V[5])/muG
                                         
  
        return (-V[5]/tauw+(b)*V[0]+a*(muV-El)/tauw)

    def A6(V, exc_aff=0, inh_aff=0, pure_exc_aff=0):
        return  (-V[6]/1.0+0.*V[1])
    
    def DiffOperatorRes(V, exc_aff=0, inh_aff=0, pure_exc_aff=0,inh_fract=0):
        return np.array([A0(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),\
                         A1(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff,inh_fract=inh_fract),\
                         A2(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),\
                         A3(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff,inh_fract=inh_fract),\
                         A4(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff,inh_fract=inh_fract),\
                         A5(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),\
                         A6(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff)])
    return DiffOperatorRes


