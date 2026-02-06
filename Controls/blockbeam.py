# filename:     blockbeam.py
# author:       Joseph Heal 
# version       0.1
# date:         2026.02.04
# purpose:      The purpose of this file is to demonstrate simulation and control capabilities
#               by controlling a simple blockbeam using various control methods. 

## import libraries
import numpy as np
import matplotlib.pyplot as plt
import blockbeamParam as P

## create classes

class blockbeamDynamics:
    def __init__(self,state0 = None, alpha=0.0):
        '''
        Docstring for __init__
        
        :param alpha: Description
        '''
        if state0.all() == None:    
            self.state = np.array([
                P.z0,
                P.theta0,
                P.zdot0,
                P.thetadot0
            ])
        else:
            self.state = state0
        # From parameter file
        
        self.m1 = P.m1
        self.m2 = P.m2
        self.ell = P.length
        self.g = P.g
        self.Ts = P.Ts
        self.torque_limit = P.torque_Max


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
        F2 = self.f(self.state + self.Ts / 2 * F1, u)
        F3 = self.f(self.state + self.Ts / 2 * F2, u)
        F4 = self.f(self.state + self.Ts * F3, u)
        # Update actual state using RK4 result
        self.state = self.state + self.Ts / 6 * (F1 + 2*F2 + 2*F3 + F4)

class PD:
    def __init__(self):
        '''
        Docstring for __init__
        
        :param self: Description
        
        This will initialize the controller PD values given using 
        simple rise time analysis
        Initially I just guess...
        '''

        pass

    def update(self,state,ref):
        '''
        Docstring for update
        
        :param self: Description
        :param state: Description
        :param ref: Description
        '''
        pass

class PID:
    def __init__(self):
        '''
        Docstring for __init__
        
        :param self: Description
        
        This will initialize the controller PID values given using 
        simple rise time analysis
        '''
        




        pass

    def update(self,state,ref):
        '''
        Docstring for update
        
        :param self: Description
        :param state: Description
        :param ref: Description
        '''
        pass

class SMC:
    def __init__(self):
        '''
        Docstring for __init__
        
        :param self: Description
        '''
        pass

    def update(self, state, ref):
        '''
        Docstring for update
        
        :param self: Description
        :param state: Description
        :param ref: Description
        '''
        pass

class MPC:
    def __init__(self):
        '''
        Docstring for __init__
        
        :param self: Description
        '''
        pass

    def update(self, state, ref):
        '''
        Docstring for update
        
        :param self: Description
        :param state: Description
        :param ref: Description
        '''
        pass


## define functions
# linearized equation + parameters:

# A41 = P.g * (P.m1**2 *P.p_steady**2 + P.m1 * P.m2 * P.length**2 / 3) / (P.m1 * P.p_steady**2 + P.m2 * P.length**2 / 3)**2
# A = np.array([[0.0, 0.0, 1.0, 0.0],
#               [0.0, 0.0, 0.0, 1.0],
#               [0.0, -P.g, 0.0, 0.0],
#               [A41, 0.0, 0.0, 0.0]])

# B = np.array([0,0,0,1/(P.m1 * P.p_steady**2 + P.m2 * P.length**2 / 3)])

# C = np.array([[1,0,0,0],
#               [0,1,0,0]])

# def stateSpace_f(state, A, B, C, tau):
#     next_state = A@state + B@tau
#     y = C @ next_state
#     return y, next_state

# def rk4_stateSpace(state,u,Ts):
#         '''
#         Integrate ODE using Runge-Kutta RK4 algorithm
        
#         :param u: input 
#         '''
#         F1 = stateSpace_f(state, A,B,C,u)
#         F2 = stateSpace_f(state + Ts / 2 * F1, A,B,C, u)
#         F3 = stateSpace_f(state + Ts / 2 * F2, A,B,C, u)
#         F4 = stateSpace_f(state + Ts * F3, A,B,C, u)
#         # Update actual state using RK4 result
#         return state + Ts / 6 * (F1 + 2*F2 + 2*F3 + F4)

state0 = np.array([P.z0, P.theta0, P.zdot0, P.thetadot0])

tau0 = P.g * (P.m2 * P.length / 2)
plant = blockbeamDynamics(state0)

# time_vals = np.arange(P.t_start,P.t_end,P.Ts)
time_vals = np.arange(0,20,P.Ts)

tauVals = np.array([tau0]*len(time_vals))
stateVals = np.zeros((len(state0),len(time_vals)))
ddot = np.zeros(np.shape(stateVals))

# print(f"state0: {state0}")
# print(f"length time, tauvals: {np.shape(time_vals), np.shape(tauVals)}")
# print(f"states shape: {np.shape(stateVals)}")

for i, (t, u) in enumerate(zip(time_vals,tauVals)):
    y = plant.update(u)
    # ddot[:,i] = plant.f(plant.state,u)
    stateVals[:,i] = plant.state

fig = plt.figure()
ax1 = fig.add_subplot(1,3,1) # z, zdot
ax1.plot(time_vals,stateVals[0,:],label='z')
ax1.plot(time_vals,stateVals[2,:],label='zdot')
ax1.legend()
ax2 = fig.add_subplot(1,3,2) # z, zdot
ax2.plot(time_vals,stateVals[1,:],label='theta')
ax2.plot(time_vals,stateVals[3,:],label='thetaDot')
ax2.legend()
ax3 = fig.add_subplot(1,3,3) # z, zdot
ax3.plot(time_vals,tauVals,label='Tau')
ax3.legend()

plt.show()



## First step: simulate with static input:

