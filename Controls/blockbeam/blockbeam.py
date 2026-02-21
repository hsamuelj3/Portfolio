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

state0 = np.array([BP.z0, BP.theta0, BP.zdot0, BP.thetadot0])

plant = bbDynamics.blockbeamDynamics(state0)
controller = bbControllers.PID()
controller2 = bbControllers.LQR()

time_vals = np.arange(0,BP.t_end,BP.dt)
tauVals = np.zeros_like(time_vals) 
n = len(time_vals)
stateVals = np.zeros((len(state0),len(time_vals)))
y = plant.h()
stateVals[:,0] = plant.state

# If using Kalman filter: 
stateVals_EKF = np.copy(stateVals)


ref_sig = np.ones_like(tauVals) * 1.0
ref_fil = np.ones_like(tauVals) * 1.0

ref_filtered = 0
alpha_smooth = .01
if True:
    for i in range(len(time_vals)-1):
        # if i < n/3:
        #     ref = 1
        # elif i < n/2:
        #     ref = -1
        # else:
        #     ref = 1
        ref = 1
        ref_filtered = (1-alpha_smooth)*ref_filtered + alpha_smooth * ref
        u = controller2.update(y,ref_filtered)
        tauVals[i] = u
        y = plant.update(u)
        stateVals[:,i+1] = plant.state
        ref_sig[i] = ref
        ref_fil[i] = ref_filtered
        stateVals_EKF[:,i+1] = controller2.x

    tauVals[-1] = u

if True: # to make this a collapsing section and optional plotting
    fig = plt.figure(figsize=(12,12))
    # Plot position and 
    ax1 = fig.add_subplot(3,1,1) # z, zdot
    ax1.plot(time_vals,stateVals[0,:],label='z')
    ax1.plot(time_vals,stateVals[2,:],label='zdot')

    ax1.plot(time_vals,stateVals_EKF[0,:],'k--', label='z_est')
    ax1.plot(time_vals,stateVals_EKF[2,:],'b--', label='zdot_est')
    
    ax1.plot(time_vals,ref_sig,'g--', label= 'ref val')
    ax1.plot(time_vals,ref_fil,'r--', label= 'ref filtered')
    ax1.legend()

    ax2 = fig.add_subplot(3,1,2) # z, zdot
    ax2.plot(time_vals,stateVals[1,:]*180/np.pi,label='theta')
    ax2.plot(time_vals,stateVals[3,:]*180/np.pi, label='thetaDot')
    
    ax2.plot(time_vals,stateVals_EKF[1,:]*180/np.pi,'--',label='theta_est')
    ax2.plot(time_vals,stateVals_EKF[3,:]*180/np.pi,'--', label='thetaDot_est')
    
    ax2.legend()

    ax3 = fig.add_subplot(3,1,3) # z, zdot
    ax3.plot(time_vals,tauVals,label='Tau')
    ax3.legend()

    plt.show()

