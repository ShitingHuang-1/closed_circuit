#input q dqdt and time_step to output delta_p, the beginning time point is the equal delta_p, when the valve absolute openning
import numpy as np
import scipy as sp
import sympy
import math
import matplotlib.pyplot as plt
class ValveinQ:
    def __init__(self,density, eff_length, Aann, Kvo,Kvc,p_oc,q,diff_q):
        self.rho=density#g/cm3
        self.leff=eff_length
        self.Aann=Aann # in there, constant
        self.Kvo=Kvo#mmHg^-1s^-1
        self.Kvc=Kvc#mmHg^-1s^-1
        self.q=q
        self.poc=p_oc#open delta p and closing delta p
        self.diff_q=diff_q
        self.xii=0.9999 #xi, when xi=1 means valve open, xi=0 means close
        self.delta_pi=0 #the pressure difference 
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
        Mst=1
        Mrg=0
        Amax=Mst*self.Aann
        Amin=Mrg*self.Aann
        xi=self.xi_t(time,dt)
        return (Amax-Amin)*xi+Amin    
    

    def bernoulli_resistance(self,time,dt):#B g*cm^(-1)*s^(-2)
        Aeff=self.Aeff(time,dt)
        return self.rho/(2*Aeff**2)

    def blood_inertance(self,time,dt):#L g*cm^(-4)
        Aeff=self.Aeff(time,dt)
        return (self.rho*self.leff)/Aeff
       
    def delta_p(self,q,diff_q,time,dt):
        self.q=q
        self.diff_q=diff_q
        B=self.bernoulli_resistance(time,dt)
        L=self.blood_inertance(time,dt)
        self.delta_pi=B*self.q*abs(self.q)+L*self.diff_q
        return self.delta_pi