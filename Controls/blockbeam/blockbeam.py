# filename:     blockbeam.py
# author:       Joseph Heal 
# version       0.1
# date:         2026.02.04
# purpose:      The purpose of this file is to demonstrate simulation and control capabilities
#               by controlling a simple blockbeam using various control methods. 

## import libraries
import numpy as np
import control as cnt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import blockbeamParam as BP # This could easily be brought in here, but I keep it separate.
import blockbeamDynamics as bbDynamics 
import blockbeamControllers as bbControllers


## define functions

# linearized equation + parameters:

A41 = BP.g * (BP.m1**2 *BP.p_steady**2 + BP.m1 * BP.m2 * BP.length**2 / 3) / (BP.m1 * BP.p_steady**2 + BP.m2 * BP.length**2 / 3)**2
A = np.array([[0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0],
              [0.0, -BP.g, 0.0, 0.0],
              [A41, 0.0, 0.0, 0.0]])

B = np.array([0,0,0,1/(BP.m1 * BP.p_steady**2 + BP.m2 * BP.length**2 / 3)])

C = np.array([[1,0,0,0],
              [0,1,0,0]])

state0 = np.array([BP.z0, BP.theta0, BP.zdot0, BP.thetadot0])

plant = bbDynamics.blockbeamDynamics(state0)
controller = bbControllers.PID()

time_vals = np.arange(0,BP.t_end,BP.Ts)
tauVals = np.zeros_like(time_vals) 
n = len(time_vals)
stateVals = np.zeros((len(state0),len(time_vals)))
y = plant.h()
stateVals[:,0] = plant.state
ref_sig = np.ones_like(tauVals) * 1.0
ref_sig[int(n/3):] = ref_sig[0]*-1

for i in range(len(time_vals)-1):
    u = controller.update(y,ref_sig[i])
    tauVals[i] = u
    y = plant.update(u)
    stateVals[:,i+1] = plant.state
tauVals[-1] = u
# u = tauVals[i]
if True: # to make this a collapsing section and optional plotting
    fig = plt.figure()
    # Plot position and 
    ax1 = fig.add_subplot(3,1,1) # z, zdot
    ax1.plot(time_vals,stateVals[0,:],label='z')
    ax1.plot(time_vals,stateVals[2,:],label='zdot')
    ax1.plot(time_vals,ref_sig,'g--', label= 'ref val')
    ax1.legend()

    ax2 = fig.add_subplot(3,1,2) # z, zdot
    ax2.plot(time_vals,stateVals[3,:]*180/np.pi,'-', label='thetaDot')
    ax2.plot(time_vals,stateVals[1,:]*180/np.pi,label='theta')
    ax2.legend()

    ax3 = fig.add_subplot(3,1,3) # z, zdot
    ax3.plot(time_vals,tauVals,label='Tau')
    ax3.legend()

    plt.show()


