#delta_p as input
import numpy as np
import scipy as sp
import sympy
import math
import matplotlib.pyplot as plt
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
        return (Amax-Amin)*xi+Amin    
    

    def B(self,time,xi):#B g*cm^(-7)
        Aeff=self.Aeff(time,xi)
        return self.rho/(2.*(Aeff)**2)
        #return self.Kt*self.rho/2.*(1./Aeff - 1./self.Ao)**2

    def L(self,time,xi):#L g*cm^(-4)
        Aeff=self.Aeff(time,xi)
        return (self.rho*self.leff)/Aeff
    def dq(self,time,q,xi,pin,pout):#cm^(3)s^(-2)
        return (pin-pout-self.B(time,xi)*q*np.abs(q))/(self.L(time,xi))#q0 is the input flow