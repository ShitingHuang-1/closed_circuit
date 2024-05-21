import numpy as np
import scipy as sp
import sympy
import math
import matplotlib.pyplot as plt
class PressureSystem:
    def __init__(self, C, Za, R):
        self.C = C 
        self.Za = Za 
        self.R = R #
    def dp(self,time,p,qin,pout):
        return ((qin-self.qout(p,pout))/self.C)
    def qout(self,pin,pout):#q1
        delta_p=pin-pout
        return delta_p/self.R
    def pi(self,qin,p):
        return (p+self.Za*qin)
        
        