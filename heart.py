#input: dvdt and vt to output pressure
import numpy as np
import scipy as sp
import sympy
import math
import matplotlib.pyplot as plt
class heart:
    def __init__(self,tau1,tau2,m1,m2,Emax,Emin,V0,T):
        self.tau1 = tau1#s
        self.tau2 = tau2#s
        self.m1 = m1#constant
        self.m2 = m2
        self.Emax = Emax*1333#g cm-4 s-2
        self.Emin = Emin*1333#g cm-4 s-2
        #self.Ks = Ks#10**(-9) s/mL
        self.v0 = V0#mL
        self.T = T
    def Et(self,t,delay = None): # mmHg*mL^(-1)
        if delay is None:
            t = t%self.T
            g1=(t/self.tau1)**self.m1#constant
            g2=(t/self.tau2)**self.m2
            k=(self.Emax-self.Emin)/max((g1/(1+g1)),(1/(1+g2)))# mmHg*mL^(-1) 133g*cm^(-4)*s^(-2)
            return k*(g1/(1+g1))*(1/(1+g2))+self.Emin
            #return 2.4
        else:
            if t-delay <= 0:
                t = 0
            else:
                t = (t-delay)%self.T
            g1=(t/self.tau1)**self.m1#constant
            g2=(t/self.tau2)**self.m2
            k=(self.Emax-self.Emin)/max((g1/(1+g1)),(1/(1+g2)))# mmHg*mL^(-1) 133g*cm^(-4)*s^(-2)
            return k*(g1/(1+g1))*(1/(1+g2))+self.Emin
            #return 0.2
    def Rs(self,vt,t,delay = None):#133*10^(9)*s^(-1)*g*cm^(-2)
        if delay is None:
            rs=self.Ks*self.Et(t)*(vt-self.v0)
            return rs
        else:
            rs=self.Ks*self.Et(t,delay)*(vt-self.v0)
            return rs
    def p(self,vt,t,delay = None):#g cm-1 s-2
        #qout is the ejection flow or aortic flow
        if delay is None:
            #pt = self.Et(t)*(vt-self.v0)-self.Rs(vt,t)*qout
            pt = self.Et(t)*(vt-self.v0)
            return pt
        else:
            #pt=self.Et(t,delay)*(vt-self.v0)-self.Rs(vt,t,delay)*qout
            pt = self.Et(t,delay)*(vt-self.v0)
            return pt
    def dv(self,time,v,qin,qout):
        return qin-qout
        
    
        