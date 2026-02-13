# filename: blockbeamDynamics

import numpy as np
import blockbeamParam as BP

class blockbeamDynamics:
    def __init__(self,state0 = None, alpha=0.0):
        '''
        Initialize the blockbeam dynamic system. 
        
        :param alpha: Description
        '''
        if state0 is None:    
            self.state = np.array([
                BP.z0,
                BP.theta0,
                BP.zdot0,
                BP.thetadot0
            ])
        else:
            self.state = state0
        # From parameter file
        
        self.m1 = BP.m1
        self.m2 = BP.m2
        self.ell = BP.length
        self.g = BP.g
        self.dt = BP.dt
        self.torque_limit = BP.tau_max


    def update(self,u):
        '''
        Docstring for update
        
        :param u: raw torque input given to the dynamics. 
        '''
        u = np.clip(u,-self.torque_limit,self.torque_limit) # Not necessary if the controller saturates the input first
        self.rk4_step(u) # dynamics of the step
        y = self.h() # extract y = C*x style "readable states"
        return y
    
    def f(self, state, u):
        '''
        Docstring for f
        
        :param state: current state of the dynamic system (extracts things for readability)
        :param u: input given to the system
        '''
        # State is in the form [z, theta, zdot, thetadot]
        z = state[0]
        theta = state[1]
        zdot = state[2]
        thetadot = state[3]

        thetaddot = (u - self.g * (self.m2 * self.ell / 2 + self.m1 * z) * np.cos(theta) - 2 * self.m1 * z * zdot * thetadot) / (self.m1 * z**2 + self.m2 * self.ell**2 / 3)

        zddot = (self.m1 * z * thetadot**2 - self.m1 * self.g * np.sin(theta))/self.m1

        return np.array([zdot, thetadot, zddot,thetaddot]) # xdot = [zdot, thetadot, zddot, thetaddot]

    def h(self):
        return self.state[:2] # Return the first two measurable states (z, theta)
    
    def rk4_step(self,u):
        '''
        Integrate ODE using Runge-Kutta RK4 algorithm
        
        :param u: input 
        '''
        F1 = self.f(self.state, u)
        F2 = self.f(self.state + self.dt / 2 * F1, u)
        F3 = self.f(self.state + self.dt / 2 * F2, u)
        F4 = self.f(self.state + self.dt * F3, u)
        # Update actual state using RK4 result
        self.state = self.state + self.dt / 6 * (F1 + 2*F2 + 2*F3 + F4)
