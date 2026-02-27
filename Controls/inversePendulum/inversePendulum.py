# filename:     inversePendulum.py
# Author:       Joseph Heal
# Date:         2026.02.06
# Purpose:      The purpose of this file is to demonstrate simulation and control capabilities
#               by controlling a simple inverse pendulum using various control methods. 

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as FuncAnimation

import inverseParameters as iP
import inverseDynamics as ipDynamics

# plant and controller creation
state0 = np.array([iP.z0, iP.theta0, iP.zdot0, iP.thetadot0]) # start in an equilibrium position
state0 = np.array([0.0, 0.0, 1.0, 0.0]) # create your own
plant = ipDynamics.inverseDynamics(state0=state0,alpha=0.0,doubleFriction=True)


# Storage containers
tVals = np.arange(iP.t_start, iP.t_end, iP.dt)
forceVals = np.zeros_like(tVals)
stateVals = np.zeros((len(state0),len(tVals)))
y = plant.h()
stateVals[:,0] = plant.state

if True:
    for i in range(len(tVals)-1):
        u = 3.0 * np.sin(tVals[i])
        forceVals[i] = u
        y = plant.update(u)
        stateVals[:,i+1] = plant.state

    forceVals[-1] = u
if True: # to make this a collapsing section and optional plotting
    fig = plt.figure(figsize=(12,12))
    # Plot position and 
    ax1 = fig.add_subplot(3,1,1) # z, zdot
    ax1.plot(tVals,stateVals[0,:],label='z')
    ax1.plot(tVals,stateVals[2,:],label='zdot')
    ax1.legend()

    ax2 = fig.add_subplot(3,1,2) # z, zdot    
    ax2.plot(tVals,stateVals[1,:],label='theta')
    ax2.plot(tVals,stateVals[3,:], label='thetaDot')    
    ax2.legend()

    ax3 = fig.add_subplot(3,1,3) # z, zdot
    ax3.plot(tVals,forceVals,label='Force')
    ax3.legend()

    plt.show()