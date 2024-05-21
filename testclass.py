#delta_p as input
import numpy as np
import scipy as sp
import sympy
import math
import matplotlib.pyplot as plt
class test_v:
    def __init__(self, a1, a2, n1, n2, T, E, P0, lamda, v0, vd):
        self.a1 = 0.5141 
        self.a2 = 0.3257
        self.n1 = n1
        self.n2 = n2
        self.T = T
        self.Ees = E
        self.P0 = P0
        self.lamda = lamda
        self.v0 = v0
        self.vd = vd
    def et(self, t):
        k=1
        first = k * (((t/(self.a1*self.T))**self.n1)/(1+(t/(self.a1*self.T))**self.n1))
        second = 1/(1+(t/(self.a2*self.T))**self.n2)
        return first*second
    def Ea(self,t):
        return self.Emax*(self.et(t)+self.Emin/self.Emax)
    def pes(self, t, v):
        return self.Ees*(v-self.vd)
    def ped(self, t, v):
        return self.P0*(math.exp(self.lamda*(v-self.v0))-1)
    def p(self,t ,v):
        return self.et(t)* self.pes(t,v)+(1-self.et(t))*self.ped(t,v)
    
    def dv(self,time,v,qin,qout):
        return qin-qout

class test_a:
    def __init__(self,B,C,Emax, Emin, vd):
        self.B = B
        self.C = C
        self.Emax = Emax
        self.Emin = Emin
        self.vd = vd
    def et(self,t):
        return math.exp(-self.B*(t-self.C)**2)
    def Et(self,t):
        return self.Emax*(self.et(t)+self.Emax/self.Emin)
    def p(self,t,v):
        return self.Et(t)*(v-self.vd)
    
    def dv(self,time,v,qin,qout):
        return qin-qout