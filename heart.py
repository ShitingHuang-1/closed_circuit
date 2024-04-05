#input: dvdt and vt to output pressure
import numpy as np
import scipy as sp
import sympy
import math
import matplotlib.pyplot as plt
class heart:
    def __init__(self,tau1,tau2,m1,m2,Emax,Emin,Ks,V0,T):
        self.tau1=tau1#s
        self.tau2=tau2#s
        self.m1=m1#constant
        self.m2=m2
        self.Emax=Emax#mmHg/mL
        self.Emin=Emin#mmHg/mL
        self.Ks=Ks#10**(-9) s/mL
        self.v0=V0#mL
        self.T=T
    def Et(self,t): # mmHg*mL^(-1) 133g*cm^(-4)*s^(-2)
        t = t%self.T
        g1=(t/self.tau1)**self.m1#constant
        g2=(t/self.tau2)**self.m2
        k=(self.Emax-self.Emin)/max((g1/(1+g1)),(1/(1+g2)))# mmHg*mL^(-1) 133g*cm^(-4)*s^(-2)
        return k*(g1/(1+g1))*(1/(1+g2))+self.Emin
    def Rs(self,vt,t):#133*10^(9)*s^(-1)*g*cm^(-2)
        rs=self.Ks*self.Et(t)*(vt-self.v0)
        return rs
    def p(self,vt,t):#10^(-9)*mmHg
        pt=self.Et(t)*(vt-self.v0)
        return pt
    def qt(self,time,q):
        return -q
    def vt(self,q,time,delta_t):#input the flow to get the volume change with time
        y,v=sp.integrate.quad(self.qt,time,time+delta_t,args=(q))
        v_update=self.v0+y
        return v_update
        