import numpy as np
import scipy as sp
import sympy
import math
import matplotlib.pyplot as plt
class Valve:
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

    def xi_t(self,delta_t):#rate of opening and closure        
        if self.delta_pi>=self.poc:
            self.xii=self.xii+(1-self.xii)*self.Kvo*(self.delta_pi-self.poc)*delta_t
            return self.xii
        elif self.delta_pi<self.poc:
            self.xii=self.xii+self.xii*self.Kvc*(self.delta_pi-self.poc)*delta_t
            return self.xii
    
    def Aeff(self,delta_t):
        Mst=1
        Mrg=0
        Amax=Mst*self.Aann
        Amin=Mrg*self.Aann
        xi=self.xi_t(delta_t)
        return (Amax-Amin)*xi+Amin    
    

    def bernoulli_resistance(self,delta_t):#B g*cm^(-1)*s^(-2)
        Aeff=self.Aeff(delta_t)
        return self.rho/(2*Aeff**2)

    def blood_inertance(self,delta_t):#L g*cm^(-4)
        Aeff=self.Aeff(delta_t)
        return (self.rho*self.leff)/Aeff
       
    def delta_p(self,q,diff_q,delta_t):
        self.q=q
        self.diff_q=diff_q
        B=self.bernoulli_resistance(delta_t)
        L=self.blood_inertance(delta_t)
        self.delta_pi=B*self.q*abs(self.q)+L*self.diff_q
        return self.delta_pi