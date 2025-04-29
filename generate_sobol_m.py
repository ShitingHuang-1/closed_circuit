import numpy as np
import scipy as sp
import sympy
import math
import matplotlib.pyplot as plt
from SALib.sample import sobol as sobol_sam
from tqdm import tqdm 
from p_tqdm import p_map
import sys
import os
sys.path.append(os.path.abspath("/home/postgrads/2654133H/code/closed_loop/MCMC/"))
from closed_circuit import *
def ODE(params):
    C_p, Za_p, R_p, Emax_rv, Emin_rv = params
    rv=heart(tau1 = 0.269*T, tau2 = 0.452*T, m1 = 1.32, m2 = 27.4, Emax = Emax_rv, Emin = Emin_rv, V0 = 10, T = T, deltat = 0.01) # Ks, V0 ,T
    cap_p=PressureSystem(C_p, Za_p, R_p)
    
    def dydt(t, y):
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
        p_la = la.p(v_la,t)
        p_lv = lv.p(v_lv,t)
        #p_aa: pressure at the coupling point of av and capillaries
        p_aa = cap_s.pi(q_av,pa)
        #pulmonary
        p_ra = ra.p(v_ra,t)
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

    y_initial = np.array([135, 27, #lv, la
                      150, 10, #qav, qmv
                      0.01, 0.5, #xi_av, xi_mv
                      5*1333, #pa
                      180, 40, #v_rv, v_ra 
                      150, 10, #q_tv, q_pv
                      0.01, 0.5, #xi_tv, xi_pv
                      5*1333]) # pb
    
    sol = sp.integrate.solve_ivp(dydt , t , y0 = y_initial , t_eval = t_span, method='LSODA')
    
    sol.t = sol.t[int(T*1000):int(T*1100)]
    v_lv = sol.y[0][int(T*1000):int(T*1100)]
    v_la = sol.y[1][int(T*1000):int(T*1100)]
    q_av = sol.y[2][int(T*1000):int(T*1100)]
    q_mv = sol.y[3][int(T*1000):int(T*1100)]
    xi_av = sol.y[4][int(T*1000):int(T*1100)]
    xi_mv = sol.y[5][int(T*1000):int(T*1100)]
    pa = sol.y[6][int(T*1000):int(T*1100)]
    v_rv = sol.y[7][int(T*1000):int(T*1100)]
    v_ra = sol.y[8][int(T*1000):int(T*1100)]
    q_tv = sol.y[9][int(T*1000):int(T*1100)]
    q_pv = sol.y[10][int(T*1000):int(T*1100)]
    xi_tv = sol.y[11][int(T*1000):int(T*1100)]
    xi_pv = sol.y[12][int(T*1000):int(T*1100)]
    pb = sol.y[13][int(T*1000):int(T*1100)]
    
    p_lv, p_rv, p_la, p_ra, p_ao, p_pa = [], [], [], [], [], []
    for i,j in zip(sol.t,range(len(sol.t))):
        p_lv.append(lv.p(v_lv[j], i))
        p_la.append(la.p(v_la[j], i))
        p_ra.append(ra.p(v_ra[j], i))
        p_rv.append(rv.p(v_rv[j], i))
        p_ao.append(cap_s.pi(q_av[j], pa[j]))
        p_pa.append(cap_p.pi(q_pv[j], pb[j]))
    p_lv, p_rv, p_ao, p_pa, p_la, p_ra= np.array(p_lv)/1333, np.array(p_rv)/1333, np.array(p_ao)/1333, np.array(p_pa)/1333, np.array(p_la)/1333, np.array(p_ra)/1333
    rvedv = max(v_rv)
    rvesv = min(v_rv)
    lvedv = max(v_lv)
    lvesv = min(v_lv)
    lvsv = lvedv - lvesv
    rvsv = rvedv - rvesv
    lvef = lvsv/lvedv
    rvef = rvsv/rvedv
    lvsp = max(p_lv)
    rvsp = max(p_rv)
    pasp = max(p_pa)
    padp = min(p_pa)
    return lvsp, rvsp, pasp, padp, lvedv, lvesv, rvedv, rvesv, lvsv, rvsv, lvef, rvef
if __name__ == "__main__":
    T = 0.8    
    problem = {
        'num_vars': 5,
        'names': ['C_p', 'Za_p', 'R_p', 'Emax_rv','Emin_rv'],
        'bounds': [[0.0005,0.01], [1, 60], [50, 500], [0.2, 2], [0.01, 0.16]] 
    }
    
    param_values =  sobol_sam.sample(problem, 65536*8, calc_second_order = False) #65536*8
    
    t_end = 8.9
    num_cpu = 96
    t_step = 0.01
    t = [0,t_end]
    t_span=np.arange(0, t_end, t_step)

    lv=heart(tau1 = 0.269*T, tau2 = 0.452*T, m1 = 1.32, m2 = 27.4, Emax = 3, Emin = 0.08, V0 = 10,T = T, deltat = 0.01)# V0 (mL), T (s)
    la=heart(tau1 = 0.110*T, tau2 = 0.180*T, m1 = 1.32, m2 = 13.1, Emax = 0.17, Emin = 0.08, V0 = 3,T = T, deltat = 0.01, delay = 0.85*T) # V0 (mL), T (s)
    ra=heart(tau1 = 0.110*T, tau2 = 0.180*T, m1 = 1.32, m2 = 13.1, Emax = 0.15, Emin = 0.04, V0 = 3, T = T, deltat = 0.01, delay = 0.85*T) # Ks, V0, T
    cap_s=PressureSystem(0.00061      , 103.44867    , 1471.26289)
    
    av=ValveinP(density = 1.06, eff_length = 2.2,Aann = 5, Kvo = 0.12,Kvc = 0.15, p_oc = 0)
    mv=ValveinP(density = 1.06,eff_length =1.9,Aann = 5, Kvo = 0.3,Kvc = 0.4, p_oc = 0)
    tv=ValveinP(1.06,2,6, 0.3,0.4, 0) #poc (mmHg)
    pv=ValveinP(1.06,1.9,2.8, 0.2,0.2, 0) #poc
    
    results = np.array(p_map(ODE, param_values, num_cpus = num_cpu))
    label = ['lvsp', 'rvsp', 'pasp', 'padp', 'lvedv', 'lvesv', 'rvedv', 'rvesv', 'lvsv', 'rvsv', 'lvef', 'rvef']
    for i,j in zip(label, range(results.shape[1])):
        globals()[f'{i}'] = results[:,j]
    data_to_save = {
    'param_value_all': param_values,
    'lvsp': lvsp,
    'rvsp': rvsp,
    'pasp': pasp,
    'padp': padp,
    'lvedv': lvedv,
    'lvesv': lvesv,
    'rvedv': rvedv,
    'rvesv': rvesv,
    'lvsv': lvsv,
    'rvsv': rvsv,
    'lvef': lvef,
    'rvef': rvef}
    np.savez_compressed('results_m.npz', **data_to_save)