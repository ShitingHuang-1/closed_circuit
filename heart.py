import numpy as np
import scipy as sp
import sympy
import math
import matplotlib.pyplot as plt
class heart:
    def __init__(self,tau1,tau2,m1,m2,Emax,Emin,Ks,V0):
        self.tau1=tau1#s
        self.tau2=tau2#s
        self.m1=m1#constant
        self.m2=m2
        self.Emax=Emax#mmHg/mL
        self.Emin=Emin#mmHg/mL
        self.Ks=Ks#10**(-9) s/mL
        self.v0=V0#mL
    def Et(self,t): # mmHg*mL^(-1) 133g*cm^(-4)*s^(-2)
        t = t%0.8
        g1=(t/self.tau1)**self.m1#constant
        g2=(t/self.tau2)**self.m2
        k=(self.Emax-self.Emin)/max((g1/(1+g1)),(1/(1+g2)))# mmHg*mL^(-1) 133g*cm^(-4)*s^(-2)
        return k*(g1/(1+g1))*(1/(1+g2))+self.Emin
    def Rs(self,vt,t):#133*10^(9)*s^(-1)*g*cm^(-2)
        rs=self.Ks*self.Et(t)*(vt-self.v0)
        return rs
    def p(self,vt,dvdt,t):#10^(-9)*mmHg
        pt=self.Et(t)*(vt-self.v0)-self.Rs(vt,t)*dvdt
        return pt