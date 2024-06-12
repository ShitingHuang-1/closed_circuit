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
class lumped_rection:
    def __init__(self,C,trunk,artery,Rother, r_rate, right= True):
        self.C = C
        self.trunk = trunk
        self.artery = artery
        self.Rother = Rother
        self.rr = r_rate
        self.right = right
    def update_surgeryZa(self):
        if self.rr ==0:
            self.rr = 1e-8
        if self.rr ==1:
            self.rr = 1-1e-8
        R_single_artery = self.artery*2 # lung original
        if self.right is True:
            R_lobe = R_single_artery*3
            R_r = R_lobe/(3-3*self.rr)
            R_total = R_r*R_single_artery/(R_r+R_single_artery)
            return R_total
        if self.right is False:
            R_lobe = R_single_artery*2
            R_l = R_lobe/(3-2*self.rr)
            R_total = R_l*R_single_artery/(R_l+R_single_artery)
            return R_total
    def update_surgeryR(self):
        if self.rr ==0:
            self.rr = 1e-8
        if self.rr ==1:
            self.rr = 1-1e-8
        R_single_artery = self.Rother*2
        if self.right is True:
            R_lobe = R_single_artery*3
            R_r = R_lobe/(3-3*self.rr)
            R_total = R_r*R_single_artery/(R_r+R_single_artery)
            return R_total
        if self.right is False:
            R_lobe = R_single_artery*2
            R_l = R_lobe/(3-2*self.rr)
            R_total = R_l*R_single_artery/(R_l+R_single_artery)
            return R_total
    def update_surgeryC(self):
        ori_C = self.C*0.5 #lung original
        C_total = ori_C + (1-self.rr)*ori_C
        return C_total