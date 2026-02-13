# Ball on Beam Parameter File
import numpy as np
# import control as cnt

# Physical parameters of the  ballbeam known to the controller
m1 =        0.35            # Mass of the block kg
m2 =        2.0             # mass of beam, kg
length =    2               # length of beam, m
g =         9.81            # gravity at sea level, m/s^2

p_steady = 0

# parameters for animation
width =     0.05            # width of block
height =    width*0.25      # height of block

# Initial Conditions
z0 =        0               # initial block position,m
theta0 =    0*np.pi/180     # initial beam angle,rads
zdot0 =     0               # initial speed of block along beam, m/s
thetadot0 = 0               # initial angular speed of the beam,rads/s

tau_steady = (m2 * g * length/2 + m1 * g * p_steady)

# Simulation Parameters
t_start =   0               # Start time of simulation
t_end =     30.0            # End time of simulation
dt =        0.01            # sample time for simulation
t_plot =    dt*10           # the plotting and animation is updated at this rate

# saturation limits
tau_max = 30                # Max Torque, N*m
