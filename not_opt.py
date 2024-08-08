import numpy as np
import scipy as sp
import sympy
import math
import matplotlib.pyplot as plt
from closed_circuit import *
from SALib.sample import sobol as sobol_sam
from tqdm import tqdm 
from p_tqdm import p_map
def dydt(t, y, params):
    C_s, Za_s, R_s, C_p, Za_p, R_p = params
    lv=heart(tau1 = 0.269*T, tau2 = 0.452*T, m1 = 1.32, m2 = 27.4, Emax = 3, Emin = 0.08, V0 = 10,T = T, deltat = 0.01)# V0 (mL), T (s)
    la=heart(tau1 = 0.110*T, tau2 = 0.180*T, m1 = 1.32, m2 = 13.1, Emax = 0.17, Emin = 0.08, V0 = 3,T = T, deltat = 0.01, delay = 0.85*T) # V0 (mL), T (s)
    rv=heart(tau1 = 0.269*T, tau2 = 0.452*T, m1 = 1.32, m2 = 27.4, Emax = 0.8, Emin = 0.04, V0 = 10, T = T, deltat = 0.01) # Ks, V0 ,T
    ra=heart(tau1 = 0.110*T, tau2 = 0.180*T, m1 = 1.32, m2 = 13.1, Emax = 0.15, Emin = 0.04, V0 = 3, T = T, deltat = 0.01, delay = 0.85*T) # Ks, V0, T
    
    cap_s = PressureSystem(C_s, Za_s, R_s)
    cap_p = PressureSystem(C_p, Za_p, R_p)
    
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
def ODE(params):
    C_s, Za_s, R_s, C_p, Za_p, R_p = params
    
    lv=heart(tau1 = 0.269*T, tau2 = 0.452*T, m1 = 1.32, m2 = 27.4, Emax = 3, Emin = 0.08, V0 = 10,T = T, deltat = 0.01)# V0 (mL), T (s)
    la=heart(tau1 = 0.110*T, tau2 = 0.180*T, m1 = 1.32, m2 = 13.1, Emax = 0.17, Emin = 0.08, V0 = 3,T = T, deltat = 0.01, delay = 0.85*T) # V0 (mL), T (s)
    rv=heart(tau1 = 0.269*T, tau2 = 0.452*T, m1 = 1.32, m2 = 27.4, Emax = 0.8, Emin = 0.04, V0 = 10, T = T, deltat = 0.01) # Ks, V0 ,T
    ra=heart(tau1 = 0.110*T, tau2 = 0.180*T, m1 = 1.32, m2 = 13.1, Emax = 0.15, Emin = 0.04, V0 = 3, T = T, deltat = 0.01, delay = 0.85*T) # Ks, V0, T
    cap_s=PressureSystem(C_s, Za_s, R_s)
    cap_p=PressureSystem(C_p, Za_p, R_p)
    y_initial = np.array([135, 27, #lv, la
                      150, 10, #qav, qmv
                      0.01, 0.5, #xi_av, xi_mv
                      5*1333, #pa
                      180, 40, #v_rv, v_ra 
                      150, 10, #q_tv, q_pv
                      0.01, 0.5, #xi_tv, xi_pv
                      5*1333]) # pb
    sol = sp.integrate.solve_ivp(dydt , t , y0 = y_initial , t_eval = t_span, method='LSODA', args = [params])
    
    v_lv = sol.y[0]
    v_la = sol.y[1]
    q_av = sol.y[2]
    q_mv = sol.y[3]
    xi_av = sol.y[4]
    xi_mv = sol.y[5]
    pa = sol.y[6]
    v_rv = sol.y[7]
    v_ra = sol.y[8]
    q_tv = sol.y[9]
    q_pv = sol.y[10]
    xi_tv = sol.y[11]
    xi_pv = sol.y[12]
    pb = sol.y[13]
    
    p_lv, p_rv, p_la, p_ra, p_ao, p_pa = [], [], [], [], [], []
    for i,j in zip(sol.t,range(len(sol.t))):
        p_lv.append(lv.p(v_lv[j], i))
        #p_la.append(la.p(v_la[j], i))
        #p_ra.append(ra.p(v_ra[j], i))
        p_rv.append(rv.p(v_rv[j], i))
        p_ao.append(cap_s.pi(q_av[j], pa[j]))
        #p_pa.append(cap_p.pi(q_pv[j], pb[j]))
    p_lv, p_rv, p_ao = np.array(p_lv)/1333, np.array(p_rv)/1333, np.array(p_ao)/1333
    max_volume_rv = max(v_rv[int(T*1000):int(T*1100)])
    min_volume_rv = min(v_rv[int(T*1000):int(T*1100)])
    sv_rv = max_volume_rv - min_volume_rv
    rvef = sv_rv / max_volume_rv
    max_volume_lv = max(v_lv[int(T*1000):int(T*1100)])
    
    min_volume_lv = min(v_lv[int(T*1000):int(T*1100)])
    sv_lv = max_volume_lv - min_volume_lv
    lvef = sv_lv / max_volume_lv
    maxplv = max(p_lv[int(T*1000):int(T*1100)])
    maxprv = max(p_rv[int(T*1000):int(T*1100)])
    minaop = min(p_ao[int(T*1000):int(T*1100)])
    return sv_rv, sv_lv, rvef, lvef, maxplv, maxprv, minaop
def loss(results):
    svrv, svlv, rvef, lvef, maxplv, maxprv, minaop = results[:,0],  results[:,1],  results[:,2], results[:,3], results[:,4], results[:,5], results[:,6]
    svrv_st = 71.7
    svlv_st = 73
    lvef_st = 0.54
    rvef_st = 0.61
    plv_max = 120
    prv_max = 30
    pao_min = 80
    diff_svrv = ((svrv_st - svrv)/svrv_st)**2
    diff_svlv = ((svlv_st - svlv)/svlv_st)**2
    diff_rvef = ((rvef_st - rvef)/rvef_st)**2
    diff_lvef = ((lvef_st - lvef)/lvef_st)**2
    diff_pao = ((pao_min - minaop)/ pao_min)**2
    diff_lvp = ((plv_max - maxplv)/plv_max)**2
    diff_rvp = ((prv_max - maxprv)/prv_max)**2
    err = diff_svrv + diff_svlv + diff_rvef + diff_lvef + diff_pao + diff_lvp + diff_rvp
    return err
    
if __name__ == "__main__":
    T = 0.8    
    problem = {
    'num_vars': 6,
    'names': ['C_s', 'Za_s','R_s','C_p','Za_p','R_p'],
    'bounds': [[0.0002, 0.006], [20, 600],[300,9000],[0.001,0.04], [3, 100], [30, 900] ]}
    param_values =  sobol_sam.sample(problem, 32768, calc_second_order = False) 
    param_point = np.array([0.001, 120, 1500, 0.002, 20, 200])
    param_values = np.vstack((param_values, param_point))
    t_end=10
    t_step = 0.01
    t = [0,t_end]
    t_span=np.arange(0, t_end, t_step)

    av=ValveinP(density = 1.06, eff_length = 2.2,Aann = 5, Kvo = 0.12,Kvc = 0.15, p_oc = 0)
    mv=ValveinP(density = 1.06,eff_length =1.9,Aann = 5, Kvo = 0.3,Kvc = 0.4, p_oc = 0)
    tv=ValveinP(1.06,2,6, 0.3,0.4, 0) #poc (mmHg)
    pv=ValveinP(1.06,1.9,2.8, 0.2,0.2, 0) #poc
    
    results = np.array(p_map(ODE, param_values, num_cpus=32))
    loss = loss(results)
    min_indice = np.argsort(loss)[:10]
    min_loss_value = loss[min_indice]
    param_value = param_values[min_indice]
    print(min_loss_value)
    print(param_value)
        
    data_to_save = {
        'param_value_all': param_values ,
        'ODE_results': results,
        'loss': min_loss_value,
        'opt_param': param_value
    }

    np.savez_compressed('not_opt.npz', **data_to_save)