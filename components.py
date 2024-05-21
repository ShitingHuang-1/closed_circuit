from math import cos, pi
import numpy as np

class Elastance:
    '''
    Class representing a time-varying elastance.
    E = E(t) defined by a piecewise periodic function:
    E = Emin + (Emax-Emin)/2*(1-cos(pi*t/Ts1/T)) for 0<=t<=Ts1*T
    E = Emin + (Emax-Emin)/2*(1+cos(pi*(t-Ts1*T)/(Ts2-Ts1)/T)) for Ts1*T<t<=Ts2*T
    P = E(t)*(V-Vini)
    V = P/E(t) + Vini
    dP/dt = dV/dt*E(t) = q*E(t) (here the derivative of E is being ignored?)
    Parameters:
        T: period of the elastance function
        Ts1: time at which the elastance begins to increase
        Ts2: time at which the elastance begins to decrease
        Emin: minimum elastance
        Emax: maximum elastance
        Vini: initial volume (default 0)
        Pini: initial pressure (default 0)
    '''
    def __init__(self, T, tau1, tau2, m1, m2, Emin, Emax, Vini=0, Pini=0):
        self.T = T
        self.tau1 = tau1
        self.tau2 = tau2
        self.Emin = Emin
        self.Emax = Emax
        self.Vini = Vini
        self.Pini = Pini
        self.m1 = m1
        self.m2 = m2
        self.state = {}

    def update(self, t, V, dVdt):
        '''
        Update the state of the elastance.
        '''
        self.state = {}
        self.state['t'] = t
        self.state['V'] = V
        self.state['dVdt'] = dVdt
        self.state['q'] = dVdt
        self.state['P'] = self.P(t, V)
        self.state['dPdt'] = self.dPdt(t, dVdt)
        self.state['E'] = self.E(t)
        return self.state

    def E(self, t):
        t = t%self.T
        g1=(t/self.tau1)**self.m1#constant
        g2=(t/self.tau2)**self.m2
        k=(self.Emax-self.Emin)/max((g1/(1+g1)),(1/(1+g2)))# mmHg*mL^(-1) 133g*cm^(-4)*s^(-2)
        return k*(g1/(1+g1))*(1/(1+g2))+self.Emin
    def V(self, t, P):
        return P/self.E(t) + self.Vini
    
    def P(self, t, V):
        return self.E(t)*(V-self.Vini)
    
    def dVdt(self, t, dPdt):
        return dPdt/self.E(t)
    
    def dPdt(self, t, dVdt):
        return dVdt*self.E(t)

    def q(self, t, dPdt):
        return self.dVdt(t, dPdt)
    

class Capacitance:
    '''
    Class representing a constant capacitance.
    V = P*C + Vini
    q = dV/dt = dP/dt*C
    Parameters:
        C: capacitance
        Vini: initial volume
    '''
    def __init__(self, C, Vini=0):
        self.C = lambda t: C
        self.dCdt = lambda t: 0
        self.Vini = Vini
        self.state = {}

    def update(self, t, q = None, dPindt = None, dPoutdt = None):
        '''
        Update the state of the capacitance.
        '''
        self.state = {}
        self.state['t'] = t
        if q is None:
            if dPindt is None or dPoutdt is None:
                raise ValueError('Either q or dPindt and dPoutdt must be specified.')
            self.state['dP_dt'] = dPindt - dPoutdt
            self.state['dPin_dt'] = dPindt
            self.state['dPout_dt'] = dPoutdt
            self.state['q'] = self.q(t, self.state['dP_dt'])
            self.state['dVdt'] = self.state['q']
        else:
            self.state['q'] = q
            self.state['dV_dt'] = q
            self.state['dP_dt'] = self.dPdt(t, q)
            if dPindt is None:
                if dPoutdt is None:
                    raise ValueError('Either dPindt or dPoutdt must be specified.')
                self.state['dPin_dt'] = dPoutdt + self.state['dP_dt']
                self.state['dPout_dt'] = dPoutdt
            else:
                if dPoutdt is not None:
                    raise ValueError('Only one of dPindt or dPoutdt may be specified along with q')
                self.state['dPin_dt'] = dPindt
                self.state['dPout_dt'] = dPindt - self.state['dP_dt']
            
            return self.state

    def V(self, t, P):
        '''
        Volume as a function of time and pressure. 
        This is also the integral of the flow rate.
        '''
        return P*self.C(t) + self.Vini
    
    def P(self, t, V):
        '''
        Pressure as a function of volume (integral of flow rate)
        '''
        return (V-self.Vini)/self.C(t)

    def dPdt(self, t, dVdt):
        '''
        Derivative of pressure with respect to time.
        '''
        return dVdt/self.C(t)
    
    def dVdt(self, t, dPdt):
        '''
        Derivative of volume with respect to time.
        '''
        return dPdt*self.C(t)
    
    def q(self, t, dPdt):
        '''
        Flow rate as a function of time and pressure.
        '''
        return self.dVdt(t, dPdt)
    
class Resistance:
    '''
    Class representing a constant resistance.
    P = q*R = dV/dt*R
    Parameters:
        R: resistance
    '''
    def __init__(self, R):
        self.R = lambda t: R
        self.state = {}

    def update(self, t, q = None, Pin = None, Pout = None):
        '''
        Update the state of the resistance.
        '''
        self.state = {}
        self.state['t'] = t
        if q is None:
            if Pin is None or Pout is None:
                raise ValueError('Either q or Pin and Pout must be specified.')
            self.state['deltaP'] = Pin - Pout
            self.state['Pin'] = Pin
            self.state['Pout'] = Pout
            self.state['q'] = self.q(t, self.state['deltaP'])
            self.state['dVdt'] = self.state['q']
        else:
            self.state['q'] = q
            self.state['dVdt'] = q
            self.state['deltaP'] = self.P(t, q)
            if Pin is None:
                if Pout is None:
                    raise ValueError('Either Pin or Pout must be specified.')
                self.state['Pin'] = Pout + self.state['deltaP']
                self.state['Pout'] = Pout
            else:
                if Pout is not None:
                    raise ValueError('Only one of Pin or Pout may be specified along with q')
                self.state['Pin'] = Pin
                self.state['Pout'] = Pin - self.state['deltaP']
        return self.state

    def P(self, t, q):
        '''
        Pressure as a function of flow rate.
        '''
        return q*self.R(t)
    
    def dVdt(self, t, P):
        '''
        Derivative of volume as a function of pressure.
        '''
        return P/self.R(t)
    
    def q(self, t, P):
        '''
        Flow rate as a function of pressure.
        '''
        return self.dVdt(t,P)

class Valve:
    def __init__(self, density, Ames, Leff, Kt, Ao, Kvo, Kvc):
        self.density = density
        self.Ames = Ames
        self.Leff = Leff
        self.Kt = Kt
        self.Ao = Ao
        self.Kvo = Kvo
        self.Kvc = Kvc
        self.state = {}

    def valve_rate(self, zeta, deltaP):
        if deltaP > 0:
            return self.Kvo*(1-zeta)*deltaP
        else:
            return self.Kvc*zeta*deltaP

    def rates(self, zeta, deltaP, Q):
        '''
        Calculate the rates of change of zeta and Q.
        '''
        Aeff = self.Ames*zeta + 1e-6
        L = self.density*self.Leff/Aeff
        beta = self.Kt*self.density/2.*(1./Aeff - 1./self.Ao)**2
        dzeta_dt = self.valve_rate(zeta, deltaP)
        dQ_dt = (deltaP - beta*Q*np.abs(Q))/L
        return dzeta_dt, dQ_dt
    def BL(self, zeta, deltaP, Q):
        Aeff = self.Ames*zeta + 1e-6
        L = self.density*self.Leff/Aeff
        beta = self.Kt*self.density/2.*(1./Aeff - 1./self.Ao)**2
        return beta,L
    def update(self, t, zeta, Pin, Pout, Q):
        '''
        Update the valve state
        '''
        self.state = {}
        self.state['t'] = t
        #zeta = 1./(1+np.exp(-zeta))
        self.state['zeta'] = zeta
        self.state['Pin'] = Pin
        self.state['Pout'] = Pout
        self.state['deltaP'] = Pin - Pout
        self.state['Q'] = Q
        self.state['B'], self.state['L'] = self.BL(zeta, self.state['deltaP'], Q)

        self.state['dzeta_dt'], self.state['dQ_dt'] = self.rates(zeta, self.state['deltaP'], Q)
        #self.state['dzeta_dt'] = self.state['dzeta_dt']/(1-zeta)/zeta
        return self.state
