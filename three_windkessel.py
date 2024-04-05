import numpy as np
import scipy as sp
import sympy
import math
import matplotlib.pyplot as plt
class PressureSystem:
    def __init__(self, C, Za, R):
        self.C = C
        self.Za = Za
        self.R = R
        self.pi=0
    def dp(self,time,p_initial,q,dq):
        return self.Za *dq - p_initial / (self.R * self.C) + (self.R + self.Za) * q/(self.R*self.C)
    def p(self,time,dt,q,dq):
        delta_t=dt
        t=[time,time+delta_t]
        dp_update=sp.integrate.solve_ivp(self.dp,t_span=t,y0=[self.pi],args=[q,dq])
        dp_sol=dp_update.y.flatten()[-1]
        self.pi=dp_sol
        return self.pi
    def q1(self,p):#the flow at the output point of windkessel
        return p/self.R
        