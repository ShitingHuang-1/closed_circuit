import numpy as np
import scipy as sp
import sympy
import math
import matplotlib.pyplot as plt
from typing import Union
class PressureSystem:
    def __init__(self, C, Za, R = None):
        self.C = C 
        self.Za = Za 
        self.R = R 
    def dp(self,time, p, qin, qout):
        return (qin-qout)/self.C
    def dqout_L(self, time , qout, pin , pout):
        return (pin - (qout*self.R+pout))/self.Za
    def dqout_windkessel(self, time, qout, qin):
        return (qin - qout)/self.C
    def dp_original(self,time,p,qin,pout):
        return ((qin-self.qout(p,pout))/self.C)
    def qout(self,pin,pout):#q1
        delta_p=pin-pout
        return delta_p/self.R
    def pi(self,qin,p):
        return (p+self.Za*qin)
    def qin(self, pin, pout):
        return (pin - pout)/self.Za
    def qout(self, pin, pout):
        return (pin - pout)/self.R
class lumped_resection_seperate:
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
    
class lumped_resection_unified:
    def __init__(self,C, Za, R, n, N):
        self.C = C
        self.Za = Za
        self.R = R
        self.N = N
        if n == N:
            self.n = N - 10**(-9)
        else:
            self.n = n
    def updated_R(self):
        R_ele = self.R*self.N
        R_update = R_ele/(self.N-self.n)
        return R_update
    def updated_C(self):
        C_ele = self.C/self.N
        C_update = C_ele*(self.N - self.n)
        return C_update
    def updated_Za(self):
        Za_ele = self.Za*self.N
        Za_update = Za_ele/(self.N-self.n)
        return Za_update
    
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
        Aeff = (Amax-Amin)*xi+Amin 
        if Aeff <1e-8:
            Aeff = 1e-8
        return Aeff
    

    def B(self,time,xi):#B g*cm^(-7)
        Aeff=self.Aeff(time,xi)
        return self.rho/(2.*(Aeff)**2)
        #return self.Kt*self.rho/2.*(1./Aeff - 1./self.Ao)**2

    def L(self,time,xi):#L g*cm^(-4)
        Aeff=self.Aeff(time,xi)
        return (self.rho*self.leff)/Aeff
    def dq(self,time,q,xi,pin,pout):#cm^(3)s^(-2)
        return (pin-pout-self.B(time,xi)*q*np.abs(q))/(self.L(time,xi))#q0 is the input flow
#input: dvdt and vt to output pressure
class heart:
    def __init__(self,tau1,tau2,m1,m2,Emax,Emin,V0, T, deltat, delay = None):
        self.tau1 = tau1#s
        self.tau2 = tau2#s
        self.m1 = m1#constant
        self.m2 = m2
        self.Emax = Emax*1333#g cm-4 s-2
        self.Emin = Emin*1333#g cm-4 s-2
        #self.Ks = Ks#10**(-9) s/mL
        self.v0 = V0#mL
        self.T = T
        self.delay = delay
        self.deltat = deltat
        self.t_vec = np.arange(0, self.T, self.deltat)
        if (self.T / self.deltat)%1 > 1e-10:
            raise ValueError('Time period should be integral multiple of delta t')
        g1 = (self.t_vec/self.tau1)**self.m1
        g2 = (self.t_vec/self.tau2)**self.m2
        self.H1 = g1/(1+g1)
        self.H2 = 1./(1+g2)
        k = (self.Emax - self.Emin)/(np.max(self.H1*self.H2) + 1e-6)
        self.E_vec = k*self.H1*self.H2 + self.Emin
        
    def Et(self, t): # mmHg*mL^(-1)
        if self.delay is None:
            #num =int(round((round(t,7) % self.T)/self.deltat, 7))
            #if num == 80:
            #    num = 0
            #return self.E_vec[num]
            inte = 1/self.deltat
            t_int = round(t * inte)
            T_int = round(self.T * inte)
            deltat_int = round(self.deltat * inte)
            num = int(t_int % T_int/deltat_int)
            return self.E_vec[num]
        else:
            t = np.maximum(t - self.delay, 0)
            inte = 1/self.deltat
            t_int = round(t * inte)
            T_int = round(self.T * inte)
            deltat_int = round(self.deltat * inte)
            num = int(t_int % T_int/deltat_int)
            return self.E_vec[num]
            
        
    def Rs(self,vt,t):#133*10^(9)*s^(-1)*g*cm^(-2)
        rs=self.Ks*self.Et(t)*(vt-self.v0)
        return rs
    
    def p(self,vt,t):#g cm-1 s-2
        #qout is the ejection flow or aortic flow
        #pt = self.Et(t)*(vt-self.v0)-self.Rs(vt,t)*qout
        pt = self.Et(t)*(vt-self.v0)
        return pt
    def dv(self,time,v,qin,qout):
        return qin-qout
class heart_fk:
    def __init__(self, a, b, c, d, n, m, ppmin, ppmax, theta, neta, typ, ventricle = None, tpmin = None, tpmax = None, phi = None, v = None):
        self.a = a*1333
        self.b = b
        self.c = c*1333
        self.d = d*1333
        self.n = n
        self.m = m
        self.tpmin = tpmin
        self.tpmax = tpmax
        self.v = v
        self.ppmin = ppmin
        self.ppmax = ppmax
        self.theta = theta
        self.neta = neta
        self.typ = typ
        if self.typ == 'ventricle':
            self.k1 = None
            self.k2 = None
            self.ventricle = None
            self.alpha = 0
            self.phi = 1.156
            if tpmin is None or tpmax is None or phi is None or v is None:
                raise ValueError("For 'ventricle', tpmin, tpmax, phi and v should be provided.")
        elif self.typ == 'atrium':
            self.k1 = 0.01
            self.k2 = 0.6
            self.phi = 2.1152
            self.ventricle = ventricle
            if ventricle is None:
                raise ValueError("For 'atrium', ventricle betaH must be provided.")
        else:
            raise ValueError("Invalid type. Must be 'ventricle' or 'atrium'.")
    def tpH(self, H):
        return self.tpmin + (self.theta**self.v/ (H**self.v+self.theta**self.v))*(self.tpmax - self.tpmin)
    def betaH(self, H):
        T0 = 1/H
        if self.typ == 'ventricle':
            return ((self.n + self.m)/self.n)*self.tpH(H)
        elif self.typ == 'atrium':
            betaHv = self.ventricle
            return T0 + self.k1*betaHv
    def alphaH(self, H):
        T0 = 1/H
        betaHv = self.ventricle
        return self.betaH(H) - self.k2*betaHv
    def ppH(self, H):
        return self.ppmin + (H**self.neta/(H**self.neta + self.phi**self.neta)) * (self.ppmax - self.ppmin)
    def ft(self, t, H):
        T0 = 1/H
        t = round(t % T0, 11)
        # t = t - int(t/T0)
        if t < 1e-8 or abs(t - T0) < 1e-8:
            t = 0
        if self.typ =='ventricle':
            if t <= self.betaH(H) and t >= 0:
                upper = ((self.betaH(H) - t)**self.m)*(t**self.n)
                lower = (self.n**self.n)*(self.m**self.m)*(self.betaH(H)/(self.n + self.m))**(self.m + self.n)
                return self.ppH(H)*upper/lower
            elif t > self.betaH(H) and t<= T0:
                return 0
        elif self.typ =='atrium':
            if t <= self.betaH(H) - T0 and t >= 0:
                upper = ((t + T0 - self.alphaH(H))**self.n)*(self.betaH(H) - t - T0)**self.m
                lower = (self.n**self.n)*(self.m**self.m)*((self.betaH(H) - self.alphaH(H))/(self.n + self.m))**(self.m + self.n)
                return self.ppH(H) *upper/lower
            elif t > self.betaH(H) - T0 and t <= self.alphaH(H):
                return 0
            elif t > self.alphaH(H) and t < T0:
                upper = ((t - self.alphaH(H))**self.n) * (self.betaH(H) - t)**self.m
                lower = (self.n**self.n)*(self.m**self.m)*((self.betaH(H) - self.alphaH(H))/(self.n + self.m))**(self.m + self.n)
                return self.ppH(H)*upper/lower
        else:
            raise ValueError(f"Invalid input for self.typ: {self.typ}. Expected 'ventricle' or 'atrium'.")
    def p(self, v, t, H):
        if self.ft(t, H) is None:
            print('t:', t, 'H', H, 'ft:', self.ft(t, H), 'ppH:', self.ppH(H), 'type:', self.typ)
        return (self.a*((v - self.b)**2) + (self.c*v - self.d)*self.ft(t, H))
    def dv(self, time, v, qin, qout):
        return qin-qout

class ParameterBounds:
    def __init__(self, T):
        self.tau1_v = (0.01 * T, 2 * T)
        self.tau2_v = (0.01 * T, 2 * T)
        self.m1_v = (0.1, 50)
        self.m2_v = (0.1, 50)
        self.tau1_a = (0.005 * T, 2 * T)
        self.tau2_a = (0.005 * T, 2 * T)
        self.m1_a = (0.5, 50)
        self.m2_a = (0.5, 50)
        
        self.Emax_lv = (0.01, 10)
        self.Emin_lv = (0.01, 10)
        self.Emax_rv = (0.001, 10)
        self.Emin_rv = (0.001, 10)
        self.Emax_la = (0.001, 10)
        self.Emin_la = (0.001, 10)
        self.Emax_ra = (0.001, 10)
        self.Emin_ra = (0.001, 10)
        
        self.C_s = (0.00001, 0.01)
        self.R_s = (200, 8000)
        self.Za_s = (10, 2000)
        
        self.C_p = (0.00001, 0.01)
        self.R_p = (10, 1000)
        self.Za_p = (0.1, 200)
        self.delay_a = (0,T)
class res_circu_pul:
    def __init__(self, rpc0, vphi, vstar, rpa0, pphi, rpv0, cpa, vpcmax, cpv, cpcb, zpa):
        self.rpc0 = rpc0
        self.vphi = vphi
        self.vstar = vstar
        self.rpa0 = rpa0
        self.pphi = pphi
        self.rpv0 = rpv0
        self.cpa = cpa
        self.vpcmax = vpcmax
        self.cpv = cpv
        self.cpcb = cpcb
        self.zpa = zpa# cpc bar
    # resistance of pulmonary artery
    def rpa(self, va, ppl):
        rpa = (self.vphi*(va - self.vstar)**4 + self.rpa0) *(1 + ppl/self.pphi)
        return rpa
    def drpa(self, va, ppl, dva, dppl):
        drpa = 4*(self.vphi(-self.vstar + va)**3)*(1+(ppl/self.pphi))*dva + dppl*(self.rpa0 + self.vphi*(-self.vstar + va)**4)/self.pphi
        return drpa
    # resistance of pulmonary vein
    def rpv(self, va, ppl):
        rpv = (self.vphi*(va - self.vstar)**4 + self.rpv0) *(1 + ppl/self.pphi)
        return rpv
    def drpv(self, va, ppl, dva, dppl):
        drpv = 4*(self.vphi(-self.vstar + va)**3)*(1+(ppl/self.pphi))*dva + dppl*(self.rpv0 + self.vphi*(-self.vstar + va)**4)/self.pphi
        return drpv
    # pressure of pulmonary capillary
    def ppc(self, pa, ptmb):
        ppc = pa + ptmb
        return ppc
    # the flow between artery and capillary
    def qpac(self, ppa, ppc, rcpa, rpa):
        q = (ppa - ppc)/ (rcpa + rpa)
        return q
    # the flow into the pulmonary capillary
    def dvpc(self, ppam, ppc, rcpa, rpa, ppvm,  rcpv):
        dvpc = ((ppam - ppc)/ (rcpa+rpa)) - ((ppc - ppvm)/ rcpv)
        return dvpc
    def dppam(self, qav, rpa, rcpa, ppc, ppam):
        #flow between the cap and artery 
        qpca = (ppam - ppc)/ (rpa+rcpa)
        qpa = qav - qpca
        dppam = qpa/self.cpa
        return dppam
    def rcpv(self, vpc):
        rcpv = 0.5*self.rpc0*(self.vpcmax/vpc)**2
        return rcpv
    def rcpa(self, vpc):
        rcpa = 0.5*self.rpc0*(self.vpcmax/vpc)**2
        return rcpa
    def dppvm(self, ppc, ppvm, pla, rcpv, rpv):
        #flow between the cap and vein
        qpvc = (ppc - ppvm)/rcpv
        qla = (ppvm - pla)/ rpv
        dppvm = (qpvc - qla)/self.cpv
        return dppvm
    def ptmb(self, vpc):
        if vpc >= 0.5*self.vpcmax and vpc <=self.vpcmax:
            ma = (self.vpcmax - 0.001)/(self.vpcmax - 13.6*self.cpcb - 0.001)
            mb = 13.6/(6.908 + np.log(ma - 0.999))
            mc = 20.4 - 6.908*mb
            ptmb = mc - mb*np.log(((self.vpcmax - 0.001)/ (vpc-0.001)) - 0.999)
            return ptmb
        elif vpc >= 0 and vpc <= 0.5*self.vpcmax:
            mc = 0.124002
            mb = -5.502
            ptmb = mc - mb*(0.7 - ((vpc - 0.001)/ (self.vpcmax-0.001)))**2
            return ptmb
        else:
            return 0
    def qpvc(self, ppc, ppvm, rcpv):
        qpvc = (ppc - ppvm)/rcpv
        return qpvc
    def ppa(self, ppam, qpv):
        ppa = ppam + qpv * self.zpa
        return ppa