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
    def __init__(self,trunk_c, trunk_r, artery_r, artery_c, arterioles_r, arterioles_c, capillaries_r, venules_r, venules_c, vein_r, vein_c, n, N):
        self.trunk_c = trunk_c
        self.trunk_r = trunk_r
        self.art_r = artery_r
        self.art_c = artery_c
        self.arterio_r = arterioles_r
        self.arterio_c = arterioles_c
        self.cap_r = capillaries_r
        self.ven_r = venules_r
        self.ven_c = venules_c
        self.vein_r = vein_r
        self.vein_c = vein_c
        self.n = n
        self.N = N
    def r_art_aseff(self):
        r_art_ele = self.N*self.art_r
        r_art_aseff = r_art_ele/(self.N-self.n)
        return r_art_aseff
    def r_arterio_aseff(self):
        r_arterio_ele = self.N*self.arterio_r
        r_arterio_aseff = r_arterio_ele/(self.N-self.n)
        return r_arterio_aseff
    def r_cap_aseff(self):
        r_cap_ele = self.N * self.cap_r
        r_cap_aseff = r_cap_ele/(self.N-self.n)
        return r_cap_aseff
    def r_ven_aseff(self):
        r_ven_ele = self.N * self.ven_r
        r_ven_aseff = r_ven_ele/(self.N-self.n)
        return r_ven_aseff
    def r_vein_aseff(self):
        r_vein_ele = self.N * self.vein_r
        r_vein_aseff = r_vein_ele/(self.N-self.n)
        return r_vein_aseff
    def r3(self):
        r3 = self.r_art_aseff() + self.trunk_r
        return r3
    def r4(self):
        r4 = self.r_arterio_aseff()+self.r_cap_aseff()+self.r_ven_aseff()+self.r_vein_aseff()
        return r4
    def c_art_aseff(self):
        c_art_ele = self.art_c/self.N
        c_art_aseff = c_art_ele*(self.N - self.n)
        return c_art_aseff
    def c_arterio_aseff(self):
        c_arterio_ele = self.arterio_c/self.N
        c_arterio_aseff = c_arterio_ele*(self.N - self.n)
        return c_arterio_aseff
    def c_ven_aseff(self):
        c_ven_ele = self.ven_c/self.N
        c_ven_aseff = c_ven_ele*(self.N - self.n)
        return c_ven_aseff
    def c_vein_aseff(self):
        c_vein_ele = self.vein_c/self.N
        c_vein_aseff = c_vein_ele*(self.N - self.n)
        return c_vein_aseff
    def c(self):
        c = 0
        c = self.trunk_c + self.c_art_aseff() + self.c_arterio_aseff() + self.c_ven_aseff() + self.c_vein_aseff()
        return c

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
    def __init__(self,tau1,tau2,m1,m2,Emax,Emin,V0,T):
        self.tau1 = tau1#s
        self.tau2 = tau2#s
        self.m1 = m1#constant
        self.m2 = m2
        self.Emax = Emax*1333#g cm-4 s-2
        self.Emin = Emin*1333#g cm-4 s-2
        #self.Ks = Ks#10**(-9) s/mL
        self.v0 = V0#mL
        self.T = T
    def Et(self,t,delay = None): # mmHg*mL^(-1)
        if delay is None:
            t = t%self.T
            g1=(t/self.tau1)**self.m1#constant
            g2=(t/self.tau2)**self.m2
            k=(self.Emax-self.Emin)/max((g1/(1+g1)),(1/(1+g2)))# mmHg*mL^(-1) 133g*cm^(-4)*s^(-2)
            return k*(g1/(1+g1))*(1/(1+g2))+self.Emin
            #return 2.4
        else:
            if t-delay <= 0:
                t = 0
            else:
                t = (t-delay)%self.T
            g1=(t/self.tau1)**self.m1#constant
            g2=(t/self.tau2)**self.m2
            k=(self.Emax-self.Emin)/max((g1/(1+g1)),(1/(1+g2)))# mmHg*mL^(-1) 133g*cm^(-4)*s^(-2)
            return k*(g1/(1+g1))*(1/(1+g2))+self.Emin
            #return 0.2
    def Rs(self,vt,t,delay = None):#133*10^(9)*s^(-1)*g*cm^(-2)
        if delay is None:
            rs=self.Ks*self.Et(t)*(vt-self.v0)
            return rs
        else:
            rs=self.Ks*self.Et(t,delay)*(vt-self.v0)
            return rs
    def p(self,vt,t, delay = None):#g cm-1 s-2
        #qout is the ejection flow or aortic flow
        if delay is None:
            #pt = self.Et(t)*(vt-self.v0)-self.Rs(vt,t)*qout
            pt = self.Et(t)*(vt-self.v0)
            return pt
        else:
            #pt=self.Et(t,delay)*(vt-self.v0)-self.Rs(vt,t,delay)*qout
            pt = self.Et(t,delay)*(vt-self.v0)
            return pt
    def dv(self,time,v,qin,qout):
        return qin-qout
class ode_solve:
    def dydt(self,t,y):
        v_lv = y[0] 
        v_la = y[1]
        q_av = y[2]
        q_mv = y[3]
        xi_av = y[4]
        xi_mv = y[5]
        #pressure of capillaries in systemic circulation
        pa = y[6]
        v_rv = y[7]
        v_ra = y[8]
        q_tv = y[9]
        q_pv = y[10]
        xi_tv = y[11]
        xi_pv = y[12]
        #pressure of capillaries in pulmonary circulation
        pb = y[13]
        #calculate parameters
        #systemic
        p_la = la.p(v_la,t,0.85*T)
        p_lv = lv.p(v_lv,t)
        #p_aa: pressure at the coupling point of av and capillaries
        p_aa = cap_s.pi(q_av,pa)
        #pulmonary
        p_ra = ra.p(v_ra,t,0.85*T)
        p_rv = rv.p(v_rv,t)
        #q_cap2: flow out the capillaries of pulmonary circulation
        q_cap2 = cap_p.qout(pb,p_la)
        #p_pa: pressure at the coupling point of pv and capillaries
        p_pa = cap_p.pi(q_tv,pb)
        #q_cap1: flow out the capillaries of systemic circulation
        q_cap1 = cap_s.qout(pa,p_ra)
        #derivative
        #la
        dv_la = la.dv(t,v_la,q_cap2,q_mv)
        dxi_mv = mv.dxi(t,xi_mv,p_la,p_lv)
        dq_mv = mv.dq(t,q_mv,xi_mv,p_la,p_lv)
        #lv
        dv_lv = lv.dv(t,v_lv,q_mv,q_av)
        dxi_av = av.dxi(t,xi_av,p_lv,p_aa)
        dq_av = av.dq(t,q_av,xi_av,p_lv,p_aa)
        #cap sys
        dpa = cap_s.dp(t,pa,q_av,p_ra)
        #ra
        dv_ra = ra.dv(t,v_ra,q_cap1,q_tv)
        dxi_tv = tv.dxi(t,xi_tv,p_ra,p_rv)
        dq_tv = tv.dq(t,q_tv,xi_tv,p_ra,p_rv)
        #rv
        dv_rv = rv.dv(t,v_rv,q_tv,q_pv)
        dxi_pv = pv.dxi(t,xi_pv,p_rv,p_pa)
        dq_pv = pv.dq(t,q_pv,xi_pv,p_rv,p_pa)
        #cap pul
        dpb = cap_p.dp(t,pb,q_pv,p_la)
        #derivative vector
        dy = np.array([dv_lv, dv_la, 
                   dq_av, dq_mv, 
                   dxi_av, dxi_mv, 
                   dpa, 
                   dv_rv, dv_ra, 
                   dq_tv, dq_pv, 
                   dxi_tv, dxi_pv, 
                   dpb])
        return dy
    def ode_sol(self,t,y0,t_eval):
        sol = sp.integrate.solve_ivp(self.dydt, t_span=t, y0=y0, t_eval=t_eval, method='LSODA')
        return sol

    
        