# filename:     inverseParameters.py
# Author:       Joseph Heal
# Date:         2026.02.27
# Purpose:      This file will hold all parameters related to the model
#               which I have analyzed and will simulate. 

# Import libraries

import numpy as np


#Parameters

# Cart and pole parameters: 
mPendulum = 0.30 # mass of the pendulum stick, center of gravity at length/2
mCart = 2.0 # mass of the cart
# mBall = 1.0 # optional mass of a ball at the end of the pendulum
mTotal = mPendulum + mCart # +mBall # Total mass is often used in the calculations. 
length = 1 # Rod length

# Friction terms (from lagrange formulation)
# This may be better placed inside the dynamics file, they can be changed there. 

a = 0.1
b = 0.1

# initial conditions

theta0 = 0
z0 = 0
thetadot0 = 0
zdot0 = 0

# World/Sim conditions:

g = 9.81 # Gravity
dt = 0.01
fMax = 1e10 # just for the start

t_start = 0.0
t_end = 30