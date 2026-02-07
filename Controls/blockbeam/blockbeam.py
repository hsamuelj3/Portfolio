# filename:     blockbeam.py
# author:       Joseph Heal 
# version       0.1
# date:         2026.02.04
# purpose:      The purpose of this file is to demonstrate simulation and control capabilities
#               by controlling a simple blockbeam using various control methods. 

## import libraries
import numpy as np
import matplotlib.pyplot as plt
import blockbeamParam as BP # This could easily be brought in here, but I keep it separate.
import blockbeamDynamics as bbDynamics
import blockbeamControllers as bbControllers

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

state0 = np.array([BP.z0, BP.theta0, BP.zdot0, BP.thetadot0])

tau0 = BP.g * (BP.m2 * BP.length / 2)
plant = bbDynamics.blockbeamDynamics(state0)
# controller = bbControllers.PD()

# time_vals = np.arange(P.t_start,P.t_end,P.Ts)
time_vals = np.arange(0,BP.t_end,BP.Ts)

# Only useful for steady state so I can validate my model 
tauVals = np.array([tau0]*len(time_vals)) 
# tauVals = np.zeros_like(time_vals)

stateVals = np.zeros((len(state0),len(time_vals)))

# print(f"state0: {state0}")
# print(f"length time, tauvals: {np.shape(time_vals), np.shape(tauVals)}")
# print(f"states shape: {np.shape(stateVals)}")

for i in range(len(time_vals)):
    # u = controller.update(np.array([0,0]),plant.state)
    u = tauVals[i]
    y = plant.update(u)
    stateVals[:,i] = plant.state
    # tauVals[i] = u

fig = plt.figure()
# Plot position and 
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

