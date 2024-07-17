import scipy as sp
import sympy
import math
import matplotlib.pyplot as plt
from closed_circuit import *
import pyswarms as ps
import numpy as np
t_end=10
t = [0,t_end]
t_span=np.arange(0,t_end,0.01)
aeff_av_values = []
et_lv = []
p_lv_values=[]
time=[]
def dydt(t, y, 
    tau1_lv, tau2_lv, m1_lv, m2_lv, Emax_lv, Emin_lv,
    tau1_rv, tau2_rv, m1_rv, m2_rv, Emax_rv, Emin_rv,
    tau1_la, tau2_la, m1_la, m2_la, Emax_la, Emin_la,
    tau1_ra, tau2_ra, m1_ra, m2_ra, Emax_ra, Emin_ra,
    C_s, R_s, Za_s,
    C_p, R_p, Za_p,
    l_av, A_av,
    l_pv, A_pv):
    lv = heart(tau1 = tau1_lv, tau2 = tau2_lv, # tau1,2 (s)
         m1 = m1_lv, m2 = m2_lv, # m1,2
         Emax = Emax_lv,Emin = Emin_lv, # Emax,min (mmHg/mL)
        V0 = 10,T = T)# V0 (mL), T (s)

    la = heart(tau1 = tau1_la, tau2 = tau2_la, #tau1,2
             m1 = m1_la,m2 = m2_la, # m1,2
             Emax = Emax_la, Emin = Emin_la, #Emax,min(mmHg/mL)
             V0 = 3,T = T) # V0 (mL), T (s)

    av = ValveinP(density = 1.06, eff_length = l_av,Aann = A_av, # density, eff_length(cm), Aann
                Kvo = 0.12,Kvc = 0.15, # Kvo,Kvc
                p_oc = 0) #poc

    mv = ValveinP(density = 1.06,eff_length =1.9,Aann = 5, # density, eff_length(cm), Aann
                Kvo = 0.3,Kvc = 0.4, # Kvo,Kvc
                p_oc = 0) #poc

    #cap_s=PressureSystem(0.0008, 90, 700)#C,Za,R
    cap_s=PressureSystem(C_s, Za_s, R_s)

    rv=heart(tau1 = tau1_rv,tau2 = tau2_rv,  # tau1, 2
             m1 = m1_rv,m2 = m2_rv, # m1,2
             Emax = Emax_rv,Emin = Emin_rv, #Emax, min
             V0 = 10,T = T) # Ks, V0 ,T

    ra=heart(tau1 = tau1_ra,tau2 = tau2_ra, # tau1, 2
             m1 = m1_ra,m2 = m2_ra, # m1,2
             Emax = Emax_ra,Emin = Emin_ra, # Emax,min
             V0 = 3, T = T) # Ks, V0, T

    tv=ValveinP(1.06,2,6, # density, eff_length, Aann(cm2)
                0.3,0.4, # Kvo,Kvc
                0) #poc (mmHg)

    pv=ValveinP(1.06,l_pv,A_pv, # density, eff_length, Aann(cm2)
                0.2,0.2, # Kvo,Kvc
                0) #poc

    #cap_p=PressureSystem(0.0017, 10, 71.25)#C,Za,R
    cap_p=PressureSystem(C_p, Za_p, R_p)
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
    dpa = cap_s.dp_original(t,pa,q_av,p_ra)
    #ra
    dv_ra = ra.dv(t,v_ra,q_cap1,q_tv)
    dxi_tv = tv.dxi(t,xi_tv,p_ra,p_rv)
    dq_tv = tv.dq(t,q_tv,xi_tv,p_ra,p_rv)
    #rv
    dv_rv = rv.dv(t,v_rv,q_tv,q_pv)
    dxi_pv = pv.dxi(t,xi_pv,p_rv,p_pa)
    dq_pv = pv.dq(t,q_pv,xi_pv,p_rv,p_pa)
    #cap pul
    dpb = cap_p.dp_original(t,pb,q_pv,p_la)
    
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
def obj_fun1(params):
    tau1_lv = params[0]
    tau2_lv = params[1]
    m1_lv = params[2]
    m2_lv = params[3]
    Emax_lv = params[4]
    Emin_lv = params[5]
    
    tau1_rv = params[6]
    tau2_rv = params[7]
    m1_rv = params[8]
    m2_rv = params[9]
    Emax_rv = params[10]
    Emin_rv = params[11]
    
    tau1_la = params[12]
    tau2_la = params[13]
    m1_la = params[14]
    m2_la = params[15]
    Emax_la = params[16]
    Emin_la = params[17]
    
    tau1_ra = params[18]
    tau2_ra = params[19]
    m1_ra = params[20]
    m2_ra = params[21]
    Emax_ra = params[22]
    Emin_ra = params[23]
    
    C_s = params[24]
    R_s = params[25]
    Za_s = params[26]
    
    C_p = params[27]
    R_p = params[28]
    Za_p = params[29]
    
    l_av = params[30]
    A_av = params[31]
    l_pv = params[32]
    A_pv = params[33]
    
    params_tuple = (
    tau1_lv, tau2_lv, m1_lv, m2_lv, Emax_lv, Emin_lv,
    tau1_rv, tau2_rv, m1_rv, m2_rv, Emax_rv, Emin_rv,
    tau1_la, tau2_la, m1_la, m2_la, Emax_la, Emin_la,
    tau1_ra, tau2_ra, m1_ra, m2_ra, Emax_ra, Emin_ra,
    C_s, R_s, Za_s,
    C_p, R_p, Za_p,
    l_av, A_av,
    l_pv, A_pv)

    sol = sp.integrate.solve_ivp(dydt , t , y0 = y_initial , t_eval = t_span, args = params_tuple, method='LSODA')
    
    lv = heart(tau1 = tau1_lv, tau2 = tau2_lv, # tau1,2 (s)
         m1 = m1_lv, m2 = m2_lv, # m1,2
         Emax = Emax_lv,Emin = Emin_lv, # Emax,min (mmHg/mL)
        V0 = 10,T = T)# V0 (mL), T (s)

    la = heart(tau1 = tau1_la, tau2 = tau2_la, #tau1,2
             m1 = m1_la,m2 = m2_la, # m1,2
             Emax = Emax_la, Emin = Emin_la, #Emax,min(mmHg/mL)
             V0 = 3,T = T) # V0 (mL), T (s)
    rv=heart(tau1 = tau1_rv,tau2 = tau2_rv,  # tau1, 2
             m1 = m1_rv,m2 = m2_rv, # m1,2
             Emax = Emax_rv,Emin = Emin_rv, #Emax, min
             V0 = 10,T = T) # Ks, V0 ,T
    cap_s=PressureSystem(C_s, Za_s, R_s)
    
    v_lv = sol.y[0]
    v_la = sol.y[1]
    q_av = sol.y[2]
    pa = sol.y[6]
    v_rv = sol.y[7]
    v_ra = sol.y[8]
    
    p_lv = []
    p_la = []
    p_ao = []
    p_rv = []
    for i,j in zip(sol.t,range(len(sol.t))):
        p_lv.append(lv.p(v_lv[j], i))
        p_la.append(la.p(v_la[j], i, 0.85*T))
        p_ao.append(cap_s.pi(q_av[j],pa[j]))
        p_rv.append(rv.p(v_rv[j],i))
    
    p_lv = list(map(lambda x: x / 1333, p_lv))
    p_la = list(map(lambda x: x / 1333, p_la))
    p_ao = list(map(lambda x: x / 1333, p_ao))
    
    max_volume_rv = max(v_rv[880:980])
    min_volume_rv = min(v_rv[880:980])
    sv_rv = max_volume_rv - min_volume_rv
    max_volume_lv = max(v_lv[880:980])
    min_volume_lv = min(v_lv[880:980])
    p_nor = 300
    v_nor = 250
    vsv_nor = 150
    sv_lv = (max_volume_lv - min_volume_lv)/vsv_nor
    rvef = sv_rv / max_volume_rv
    lvef = sv_lv / max_volume_lv
    psys_lv = max(p_lv[880:980])/p_nor
    psys_rv = max(p_rv[880:980])/p_nor
    ao_sys = max(p_ao[880:980])/p_nor
    ao_dia = min(p_ao[880:980])/p_nor
    v_lv_max = max(v_lv[880:980])/v_nor
    v_rv_max = max(v_rv[880:900])/v_nor
    #standard
    sv_lv_s = 70/vsv_nor
    lvef_s = 0.59
    v_lv_s = 123/v_nor
    rvef_s = 0.61
    v_rv_s = 111/v_nor
    p_lv_s = 120/p_nor
    p_ao_smax = 120/p_nor
    p_ao_smin = 80/p_nor
    
    target = np.array([
        sv_lv_s,
        lvef_s,
        rvef_s,
        v_lv_s,
        v_rv_s,
        p_lv_s,
        p_ao_smax,
        p_ao_smin
    ])
    current = np.array([
        sv_lv,
        lvef,
        rvef,
        v_lv_max,
        v_rv_max,
        psys_lv,
        ao_sys,
        ao_dia
    ])
    err = current - target
    return np.sum(err**2)
T = 0.8
y_initial = np.array([135, 27, #lv, la
                      150, 10, #qav, qmv
                      0.01, 0.5, #xi_av, xi_mv
                      5*1333, #pa
                      180, 40, #v_rv, v_ra 
                      150, 10, #q_tv, q_pv
                      0.01, 0.5, #xi_tv, xi_pv
                      5*1333])

def objective_function(params):
    all_err = []
    for param in params:
        all_err.append(obj_fun1(param))
    return all_err
if __name__ == "__main__":
    lb = [0.1*T, 0.401*T, 0.5, 1, 0.5, 0.01, 0.1*T, 0.401*T, 0.5, 1, 0.1, 0.01, 0.05*T, 0.16*T, 0.5,1,0.1,0.01, 0.05*T, 0.16*T, 0.5,1,0.1,0.01,0.0005,200,10,0.0005,50,10,1,2,1,2]   # low boundary
    ub = [0.4*T, 0.7*T, 10,20,10,0.49, 0.4*T, 0.7*T, 10,20,2,0.09, 0.15*T, 0.3*T, 10,20,1,0.09, 0.15*T, 0.3*T, 10,20,1,0.09,0.002,7000,500,0.002,200,200,4,8,4,8]  # up boundary
    bounds = (lb, ub)
    
    options = {'c1': 2, 'c2': 2, 'w': 0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=34, options=options, bounds = bounds, ftol= 0.001)
    cost, pos = optimizer.optimize(objective_function, iters=2000, n_processes = 32)
    print(f"Optimal Parameters:")
    print(f"m1_lv = {pos[0]}, m2_lv = {pos[1]}, Emax_lv = {pos[2]}, Emin_lv = {pos[3]}")
    print(f"m1_rv = {pos[4]}, m2_rv = {pos[5]}, Emax_rv = {pos[6]}, Emin_rv = {pos[7]}")
    print(f"m1_la = {pos[8]}, m2_la = {pos[9]}, Emax_la = {pos[10]}, Emin_la = {pos[11]}")
    print(f"m1_ra = {pos[12]}, m2_ra = {pos[13]}, Emax_ra = {pos[14]}, Emin_ra = {pos[15]}")
    print(f"C_s = {pos[16]}, R_s = {pos[17]}, Za_s = {pos[18]}")
    print(f"C_p = {pos[19]}, R_p = {pos[20]}, Za_p = {pos[21]}")
    print(f"l_av = {pos[22]}, A_av = {pos[23]}, l_pv = {pos[24]}, A_pv = {pos[25]}")
    print(f"Optimal Value: {cost}")