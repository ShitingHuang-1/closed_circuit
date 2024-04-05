#delta_p as input
import numpy as np
import scipy as sp
import sympy
import math
import matplotlib.pyplot as plt
class ValveinP:
    def __init__(self,density, eff_length, Aann, Kvo,Kvc,p_oc,diff_p):
        self.Mst=1#healthy
        self.Mrg=0#healthy
        self.rho=density#g/cm3
        self.leff=eff_length
        self.Aann=Aann # in there, constant
        self.Kvo=Kvo#mmHg^-1s^-1
        self.Kvc=Kvc#mmHg^-1s^-1
        self.poc=p_oc#open delta p and closing delta p
        self.xii=0.9999 #xi, when xi=1 means valve open, xi=0 means close
        self.qii=0
        self.delta_pi=diff_p
    def dxi(self,time,xi):
        if self.delta_pi>=self.poc:
            return (1-xi)*self.Kvo*(self.delta_pi-self.poc)
        if self.delta_pi<self.poc:
            return xi*self.Kvc*(self.delta_pi-self.poc)
    def xi_t(self,time,dt):#rate of opening and closure 
        delta_t=dt
        t=[time,time+delta_t]
        dx_update=sp.integrate.solve_ivp(self.dxi,t_span=t,y0=[self.xii])
        dx_sol=dx_update.y.flatten()[-1]
        self.xii=dx_sol
        return self.xii
    
    def Aeff(self,time,dt):
        Amax=self.Mst*self.Aann
        Amin=self.Mrg*self.Aann
        xi=self.xi_t(time,dt)
        return (Amax-Amin)*xi+Amin    
    

    def B(self,time,dt):#B g*cm^(-1)*s^(-2)
        Aeff=self.Aeff(time,dt)
        return self.rho/(2*Aeff**2)

    def L(self,time,dt):#L g*cm^(-4)
        Aeff=self.Aeff(time,dt)
        return (self.rho*self.leff)/Aeff
    def dflow(self,time,q,dt,p):
        return (1./self.L(time,dt))*(p-self.B(time,dt)*q*abs(q))#q0 is the input flow
    def flow(self,time,dt,p,q_in):
        self.delta_pi=p
        self.qii=q_in
        delta_t=dt
        t=[time,time+delta_t]
        dq_update=sp.integrate.solve_ivp(self.dflow,t_span=t,y0=[self.qii],args=[delta_t,self.delta_pi])
        dq_sol=dq_update.y.flatten()[-1]
        return dq_sol