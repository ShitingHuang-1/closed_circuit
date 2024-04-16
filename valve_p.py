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
    def dxi(self,time,xi,pin,pout):
        delta_p=pin-pout
        if delta_p >= self.poc:
            return (1-xi)*self.Kvo*(delta_p-self.poc)
        if delta_p < self.poc:
            return xi*self.Kvc*(delta_p-self.poc)
    
    def Aeff(self,time,xi):
        Amax=self.Mst*self.Aann
        Amin=self.Mrg*self.Aann
        return (Amax-Amin)*xi+Amin    
    

    def B(self,time,xi):#B g*cm^(-1)*s^(-2)
        Aeff=self.Aeff(time,xi)
        return self.rho/(2*Aeff**2)

    def L(self,time,xi):#L g*cm^(-4)
        Aeff=self.Aeff(time,xi)
        return (self.rho*self.leff)/Aeff
    def dq(self,time,q,xi,pin,pout):
        return (1/self.L(time,xi))*(pin-pout-self.B(time,xi)*q*abs(q))#q0 is the input flow