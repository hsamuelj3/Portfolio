# filename:     inverseDynamics.py
# Author:       Joseph Heal
# Date:         2026.02.27
# Purpose:      This document will handle the physics of the inverse pendulum
#               and calculate the nonlinear motion I have determined by my 
#               pen and paper calculations

# import libraries
import numpy as np

# import other files
import inverseParameters as iP

class inverseDynamics:
    def __init__(self, state0 = None, alpha = 0.0, doubleFriction = False):
        '''
        This class will take care of the dynamics of the inverse pendulum. 

        :state0:            Initial state, if different from the parameter file
        :alpha:             uncertainty on parameter values
        :doubleFriction:    Whether to include rotational and lateral friction terms
        '''
        if state0 is None:    
            self.state = np.array([
                iP.z0,
                iP.theta0,
                iP.zdot0,
                iP.thetadot0
            ])
        else:
            self.state = state0

        # Cart Parameters with potential uncertainty
        self.mC = iP.mCart * (1 + np.random.standard_normal()* alpha)
        self.mP = iP.mPendulum * (1 + np.random.standard_normal()* alpha)
        self.mT = iP.mTotal * (1 + np.random.standard_normal()* alpha)
        self.length = iP.length * (1 + np.random.standard_normal()* alpha)

        # Friction
        self.a = iP.a # linear damping
        self.b = 0
        if doubleFriction: 
            self.b = iP.b # rotational damping

        
        # World/sim parameters
        self.g = iP.g
        self.dt = iP.dt
        self.forceLimit = iP.fMax

        # Variables which will be updated. Lighter than creating a new matrix each time and calculating the inverse
        # self.inverseMatrix is reshaped as a 2x2 after the effects of the changing state are accounted for
        self.inverseMatrix = np.array([self.length**2 / 3 * self.mP, 
                                       -self.length / 2 * self.mP * np.cos(self.state[1]),
                                       -self.length / 2 * self.mP * np.cos(self.state[1]),
                                       self.mT])
        self.inverseDivisor = self.length**2 / 3 * self.mP * self.mT - self.length**2 / 4 * self.mP**2 * (np.cos(self.state[1])**2)

    def update(self, u):
        self.rk4_step(u) # dynamics of the step
        # y = self.h() # extract y = C*x style "readable states"
        return self.h()
    
    def f(self, state,u):
        z, theta, zdot, thetadot = state # for readability
        self.inverseMatrix[1:2] = -self.length / 2 * self.mP * np.cos(theta) # This may not work because of reshaping...?
        self.inverseDivisor = self.length**2 / 3 * self.mP * self.mT - self.length**2 / 4 * self.mP**2 * (np.cos(self.state[1])**2)
        ddot = self.inverseMatrix.reshape(2,2) / self.inverseDivisor \
                @ np.array([self.length / 2 * self.mP *thetadot**2 * np.sin(theta) - self.a * zdot + u,
                            self.length / 2 * self.mP * self.g * np.sin(theta) - self.b * thetadot])
        
        return np.array([zdot,thetadot,ddot[0], ddot[1]])

    def h(self):
        return self.state[:2] # return only measurable states (z,theta)
    
    def rk4_step(self,u):
        '''
        Integrate ODE using Runge-Kutta RK4 algorithm

        Copied from my other files
        
        :param u: input 
        '''
        F1 = self.f(self.state, u)
        F2 = self.f(self.state + self.dt / 2 * F1, u)
        F3 = self.f(self.state + self.dt / 2 * F2, u)
        F4 = self.f(self.state + self.dt * F3, u)
        # Update actual state using RK4 result
        self.state = self.state + self.dt / 6 * (F1 + 2*F2 + 2*F3 + F4)
