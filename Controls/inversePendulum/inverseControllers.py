# filename:     inverseControllers.py
# author:       Joseph Heal
# date:         2026.02.27
# purpose:      Design controllers to stabilize and force 
#               reference tracking for the inverse pendulum

import scipy as sc
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize
import numpy as np
import inverseParameters as iP
import casadi as ca

class PID:
    def __init__(self):
        '''
        Basic PID (proportional, integral, derivative) control
        for the inverse pendulum problem. This will be layered 
        to deal with the coupling of dynamics (we use force to 
        influence both position of the cart and the pole angle)
        '''

        pass

    def update(self):
        pass

class LQR:
    def __init__(self):
        '''
        LQR with kalman filter for state estimation
        
        '''
        pass

    def update(self):

        pass


class MPC:
    def __init__(self, n = 5):
        '''
        MPC with variable receding horizon. 
        :param n: Time step horizon
        Optimize the control over many timesteps
        Use Casadi for optimizer (leads into real-time integration)
        Use Kalman Filter for state estimation
        
        '''
        self.N = n # time horizon
        self.predicted_states = np.zeros((4,self.N)) # state prediction vector which gets updated in MPC
        self.u0 = np.zeros(self.N) # 1d array of single inputs


        # Casadi optimization things
        self.optimization = ca.Opti()
        self.X = self.optimization.variable(4,self.N+1)
        self.U = self.optimization.variable(1,self.N)
        # Cost weights 
        self.Q = np.diag([1.0, 0.1, 10.0, 0.1]) # State costs
        self.R = np.array([[0.1]]) # input cost

        self.cost = 0

        self.pastU = 0 # store one last u to warm-start the optimization (maybe unecessary)

    def dynamics(self):

        return ca.vertcat()
    
    def cost(self,u_flat, x0, x_ref):
        u_seq = u_flat.reshape(self.N, 1)
        states = self.predictions(u_seq, x0)
        
        J = 0
        for k in range(self.N):
            dx = states[k] - x_ref
            J += dx @ self.Q @ dx + u_seq[k] @ self.R @ u_seq[k]
        return J

    def update(self, state, ref):
        '''
        Creates an optimal sequence of control inputs ofver the desired 
        time horizon. 
        '''
        # Assume all states are known (kalman filter used later)
        z, theta, zdot, thetadot = state

        # Use current state and reference 
        self.optimization.subject_to(self.X[:,0] == state)
        self.cost = 0

        # loop through future:
        for i in range(self.N):
            dx = self.X[:,i] - ref
            self.cost += dx.T @ self.Q @ dx
        


        return # Return only the first value of the sequence

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

