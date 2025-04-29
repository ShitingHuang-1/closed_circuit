import numpy as np
import scipy as sp
import sympy
import math
import matplotlib.pyplot as plt
from typing import Union
class PressureSystem:
    def __init__(self, C, Za, R = None):
        self.C = C 
        self.Za = Za 
        self.R = R 
    def dp(self,time, p, qin, qout):
        return (qin-qout)/self.C
    def dqout_L(self, time , qout, pin , pout):
        return (pin - (qout*self.R+pout))/self.Za
    def dqout_windkessel(self, time, qout, qin):
        return (qin - qout)/self.C
    def dp_original(self,time,p,qin,pout):
        return ((qin-self.qout(p,pout))/self.C)
    def qout(self,pin,pout):#q1
        delta_p=pin-pout
        return delta_p/self.R
    def pi(self,qin,p):
        return (p+self.Za*qin)
    def qin(self, pin, pout):
        return (pin - pout)/self.Za
    def qout(self, pin, pout):
        return (pin - pout)/self.R
    
class lumped_resection_unified:
    def __init__(self,C, Za, R, n, N):
        self.C = C
        self.Za = Za
        self.R = R
        self.N = N
        if n == N:
            self.n = N - 10**(-9)
        else:
            self.n = n
    def updated_R(self):
        R_ele = self.R*self.N
        R_update = R_ele/(self.N-self.n)
        return R_update
    def updated_C(self):
        C_ele = self.C/self.N
        C_update = C_ele*(self.N - self.n)
        return C_update
    def updated_Za(self):
        Za_ele = self.Za*self.N
        Za_update = Za_ele/(self.N-self.n)
        return Za_update
    
class ValveinP:
    def __init__(self,density, eff_length, Aann, Kvo,Kvc,p_oc, Kt = None, Ao = None):
        self.Mst=1#healthy
        self.Mrg=1e-6#healthy
        self.rho=density#g/cm3
        self.leff=eff_length
        self.Aann=Aann # in there, constant
        self.Kvo=Kvo
        self.Kvc=Kvc#1/pa*s
        self.poc=p_oc#open delta p and closing delta p
        if Kt == None:
            self.Kt = 0.2
        else:
            self.Kt = Kt
        if Ao == None:
            self.Ao = 0.3
        else:
            self.Ao = Ao
            
#The p is g cm-4 s-2 and Kvc is 1/(pa*s), the dxi is 1/s
    def dxi(self,time,xi,pin,pout):
        delta_p=pin-pout
        if delta_p >= self.poc:
            return (1-xi)*(self.Kvo/10)*(delta_p-self.poc)
        if delta_p < self.poc:
            return xi*(self.Kvc/10)*(delta_p-self.poc)
    
    #Aeff cm2
    def Aeff(self,time,xi):
        Amax=self.Mst*self.Aann
        Amin=self.Mrg*self.Aann
        Aeff = (Amax-Amin)*xi+Amin 
        if Aeff <1e-8:
            Aeff = 1e-8
        return Aeff
    

    def B(self,time,xi):#B g*cm^(-7)
        Aeff=self.Aeff(time,xi)
        return self.rho/(2.*(Aeff)**2)
        #return self.Kt*self.rho/2.*(1./Aeff - 1./self.Ao)**2

    def L(self,time,xi):#L g*cm^(-4)
        Aeff=self.Aeff(time,xi)
        return (self.rho*self.leff)/Aeff
    def dq(self,time,q,xi,pin,pout):#cm^(3)s^(-2)
        return (pin-pout-self.B(time,xi)*q*np.abs(q))/(self.L(time,xi))#q0 is the input flow
#input: dvdt and vt to output pressure
class heart:
    def __init__(self,tau1,tau2,m1,m2,Emax,Emin,V0, T, deltat, delay = None):
        self.tau1 = tau1#s
        self.tau2 = tau2#s
        self.m1 = m1#constant
        self.m2 = m2
        self.Emax = Emax*1333#g cm-4 s-2
        self.Emin = Emin*1333#g cm-4 s-2
        #self.Ks = Ks#10**(-9) s/mL
        self.v0 = V0#mL
        self.T = T
        self.delay = delay
        self.deltat = deltat
        self.t_vec = np.arange(0, self.T, self.deltat)
        if (self.T / self.deltat)%1 > 1e-10:
            raise ValueError('Time period should be integral multiple of delta t')
        g1 = (self.t_vec/self.tau1)**self.m1
        g2 = (self.t_vec/self.tau2)**self.m2
        self.H1 = g1/(1+g1)
        self.H2 = 1./(1+g2)
        k = (self.Emax - self.Emin)/(np.max(self.H1*self.H2) + 1e-6)
        self.E_vec = k*self.H1*self.H2 + self.Emin
        
    def Et(self, t): # mmHg*mL^(-1)
        if self.delay is None:
            inte = 1/self.deltat
            t_int = round(t * inte)
            T_int = round(self.T * inte)
            deltat_int = round(self.deltat * inte)
            num = int(t_int % T_int/deltat_int)
            return self.E_vec[num]
        else:
            t = np.maximum(t - self.delay, 0)
            inte = 1/self.deltat
            t_int = round(t * inte)
            T_int = round(self.T * inte)
            deltat_int = round(self.deltat * inte)
            num = int(t_int % T_int/deltat_int)
            return self.E_vec[num]
            
        
    def Rs(self,vt,t):#133*10^(9)*s^(-1)*g*cm^(-2)
        rs=self.Ks*self.Et(t)*(vt-self.v0)
        return rs
    
    def p(self,vt,t):#g cm-1 s-2
        pt = self.Et(t)*(vt-self.v0)
        return pt
    def dv(self,time,v,qin,qout):
        return qin-qout