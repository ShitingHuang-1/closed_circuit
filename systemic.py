from components import *

LV = Elastance(T = 1, Ts1 = 0.3, Ts2 = 0.45, Emin = 0.1, Emax = 3.3, Vini = 300)
LA = Elastance(T = 1, Ts1 = 0.92, Ts2 = 0.99, Emin = 0.15, Emax = 0.25, Vini = 10)

AV = Valve(density = 1, Ames = 0.2, Leff = 0.05, Kt = 0.2, Ao = 0.3, Kvo = 0.2, Kvc = 0.2)
MV = Valve(density = 1, Ames = 0.2, Leff = 0.05, Kt = 0.2, Ao = 0.3, Kvo = 0.3, Kvc = 0.4)

R1 = Resistance(R = 0.005)
R2 = Resistance(R = 0.65)
C = Capacitance(C = 2.6)

t = 0
initial_states = [350, 20, 10, 10, 1, 0.9, 0.2]
def get_rates(t, states, return_all_states = False):
    V_LV, V_LA, P_R, Q_AV, Q_MV, zeta_AV, zeta_MV = states
    LV_state = LV.update(t, V = V_LV, dVdt = Q_MV-Q_AV)
    R1_state = R1.update(t, q = Q_AV, Pout = P_R)
    LA_state = LA.update(t, V = V_LA, dVdt = 0) #use zero flow as the first step
    R2_state = R2.update(t,Pin=P_R, Pout = LA_state['P'])
    LA_state = LA.update(t, V = V_LA, dVdt = R2_state['q']-Q_MV) #then use the calculated flow rate QoRC
    C_state = C.update(t, q = Q_AV - R2_state['q'], dPoutdt = 0)
    AV_state = AV.update(t,zeta = zeta_AV, Pin = LV_state['P'], Pout = R1_state['Pin'], Q = Q_AV)
    MV_state = MV.update(t, zeta = zeta_MV, Pin = LA_state['P'], Pout = LV_state['P'], Q = Q_MV)
    if return_all_states:
        all_states = {}
        for key in LV_state.keys():
            all_states['LV_'+key] = LV_state[key]
        for key in LA_state.keys():
            all_states['LA_'+key] = LA_state[key]
        for key in R1_state.keys():
            all_states['R1_'+key] = R1_state[key]
        for key in R2_state.keys():
            all_states['R2_'+key] = R2_state[key]
        for key in C_state.keys():
            all_states['C_'+key] = C_state[key]
        for key in AV_state.keys():
            all_states['AV_'+key] = AV_state[key]
        for key in MV_state.keys():
            all_states['MV_'+key] = MV_state[key]
        return all_states
    rates = [Q_MV - Q_AV, R2_state['q'] - Q_MV, C_state['dPin_dt'], AV_state['dQ_dt'], MV_state['dQ_dt'], AV_state['dzeta_dt'], MV_state['dzeta_dt']]
    return rates

from scipy.integrate import solve_ivp
tspan = [0, 20]
sol = solve_ivp(get_rates, tspan, initial_states)

from matplotlib import pyplot as plt
keys = ['V_LV', 'V_LA', 'P_R', 'Q_AV', 'Q_MV', 'zeta_AV', 'zeta_MV']
fig, axs = plt.subplots(7,1) 
for i in range(7):
    axs[i].plot(sol.t, sol.y[i,:])
    axs[i].set_ylabel(keys[i])
plt.show()
#t_all = np.linspace(0,1,50)
#E = []
#for t in t_all:
#    E.append(LV.E(t))
#E2 = []
#for t in t_all:
#    E2.append(LA.E(t))
#plt.plot(t_all,E, t_all, E2)
#plt.show()

print(get_rates(sol.t[-1], sol.y[:,-1], return_all_states = True))

LV_vol = sol.y[0]
LV_pressure = []
for i,t in enumerate(sol.t):
    LV_pressure.append(LV.P(t,LV_vol[i]))
plt.plot(LV_vol[300:], LV_pressure[300:])
plt.show()
