
import scipy as sc
from scipy.linalg import solve_continuous_are
import numpy as np
import blockbeamParam as BP

## Controller classes
class PID:
    def __init__(self, tr_th = None, tr_z = None, scale = None, zeta_th = None, zeta_z = None):
        '''
        Initializes a PID controller
        
        This will calculate proper PID values using cascaded 
        loops for the blockbeam system. The inner loop will
        stabilize the beam angle, and the outer loop will 
        control the position of the block on the beam. 

        The inner loop will use a PD controller for fast action,
        while the outer loop will use PID control for accuracy. 
        '''
        
        ## Inner Loop - PD control of theta - fast response
        # Tuning parameters
        scale = 10.0             # Time scale separation
        tr_th = 0.3         # Rise time for theta (inner loop) --> This must be FAST to use a simplified model
        zeta_th = 0.707     # Damping ratio
        wn_th = 2.2 / tr_th # Desired atural frequency of the theta loop

        # Compute inner loop gains - use 2nd order characteristic equation = 1 + Control * Plant
        self.Kd_th = 2 * wn_th * zeta_th * (BP.m1 * BP.p_steady**2 + BP.m2 * BP.length**2 / 2) 
        self.Kp_th = wn_th**2 * (BP.m1 * BP.p_steady**2 + BP.m2 * BP.length**2 / 2)

        ## Outer loop - PID control of position (z) - slower response (5-10x)
        # Tuning parameters
        zeta_z = 0.707          # Damping ratio
        tr_z = scale * tr_th    # rise time for position z (outer loop)
        wn_z = 2.2 / tr_z       # Desired natural frequency of positional loop

        # Compute outer loop Gains - third order system (introduces alpha) = 1 + control * plant
        alpha = wn_z 
        self.Kp_z = -(alpha * 2 * wn_z * zeta_z + wn_z**2) / BP.g
        self.Ki_z = -(alpha * wn_z**2) / BP.g
        self.Kd_z = -(alpha + 2 * wn_z * zeta_z) / BP.g

        # Initial conditions for integrator and differentiators (not reading state directly)
        self.integrator_z = 0.
        self.error_z_prev = 0.
        self.z_dot = BP.zdot0 
        self.z_prev = BP.z0
        self.theta_dot = BP.thetadot0
        self.theta_prev = BP.theta0

        self.Ts = BP.dt
        self.sigma = 0.05 # cutoff frequency for dirty derivative

        self.m1 = BP.m1
        self.m2 = BP.m2
        self.g = BP.g
        self.length = BP.length

        # saturation limits
        self.Tau_max = BP.tau_max        # maximum torque (from parameters) 
        error_max = 1                       # maximum error allowed during calculations (maybe unused)
        self.theta_max = 30 * np.pi/180     # maximum theta ref command
        self.max_i_term = 10 * np.pi/180    # Maximum effect of the integral term... 

        # Alternate outer loop gains (plant = outer plant * cascaded inner plant = G_o * (C_i * G_i)/(1 + C_i * G_i))

        # Do I need a zero canceling filter? my gain for both of these is infinite...

        print('kp_th: ', self.Kp_th)
        print('kd_th: ', self.Kd_th)
        print('kp_z: ', self.Kp_z)
        print('ki_z: ', self.Ki_z)
        print('kd_z: ', self.Kd_z)


    def update(self,state,ref):
        '''
        Docstring for update
        
        :param state: Description
        :param ref: Description
        '''
        # Extract only "visible states" from the state
        z = state[0]
        theta = state[1] 

        # Update outer loop (position)
        error_z = ref - z

        # dirty derivative of z:
        self.z_dot = (2.0*self.sigma - self.Ts) / (2.0*self.sigma + self.Ts) * self.z_dot \
            + (2.0 / (2.0*self.sigma + self.Ts)) * ((z - self.z_prev))
        
        self.integrator_z += error_z*self.Ts
        ki_term = np.clip(self.Ki_z * self.integrator_z,-self.max_i_term,self.max_i_term)
        theta_r = self.Kp_z * error_z \
                + ki_term \
                - self.Kd_z * self.z_dot # This may be negative... I calculated them with all positive signs
        # Saturate the reference theta
        theta_r = np.clip(theta_r,-self.theta_max,self.theta_max)

        # Filtering and left half plane canceling? 

        ## Update inner Loop (angle)
        error_theta = theta_r - theta

        # Dirty derivative of theta
        self.theta_dot = (2.0*self.sigma - self.Ts) / (2.0*self.sigma + self.Ts) * self.theta_dot \
            + (2.0 / (2.0*self.sigma + self.Ts)) * ((theta - self.theta_prev))
        
        # Control input: 
        tau = self.Kp_th * error_theta - self.Kd_th * self.theta_dot \
                + (self.m2 * self.length / 2 + self.m1 * z) * self.g * np.cos(theta) # compensate for where it is currently (feed forward term)

        # Update storage variables
        self.error_z_prev = error_z
        self.z_prev = z
        self.theta_prev = theta

        return np.clip(tau,-self.Tau_max,self.Tau_max)

class SMC:
    def __init__(self, A, B, C= None, K0 = None):
        '''
        
        :param A: System dynamics matrix A 
        :param B: Input effects
        :param C: Base gains for the sliding surface (optional input)
        :param K0: Switching gain (optional input)
        '''
        self.A = A
        self.B = B

        self.C = C if C is not None else np.identity(len(A[0])) 

        self.K0 = K0 if K0 is not None else  np.identity(len(A[0]))
        self.refArray = np.zeros(len(A[0]))

    def update(self, state, ref):
        '''
        Docstring for update
        
        :param state: Description
        :param ref: Only the reference for one of the states: x
        '''

        self.refArray[0] = ref
        error_array = state - self.refArray
        s = self.C @ error_array

        
        pass

## Very quick modification of another project's sliding mode control. Didn't work
class slidingModeControl: 
    def __init__(self, alpha = 0.3):
        '''
        Docstring for __init__
        m*xdd = fx(x,xd,y,yd) + ux + dx
        m*ydd = fy(x,xd,y,yd) + uy + dy
        f_ = dynamics *See update function

        :param alpha: random uncertainty to parameters

        A disturbance is also included in the update term. 
        '''
        
        self.name = "slidingModeControl"
        
        # Uncertainty: 
        self.alpha_uncertainty = (1 + np.random.standard_normal() * alpha) # Measurement error parameter
        print(f"Alpha values: {self.alpha_uncertainty, alpha}")

        # Sliding surface for x,y
        self.lamda_z = 3.0
        self.lamda_theta = 3.0

        # Gains for x,y directions
        self.K_z = 30.0
        self.K_theta = 30.0

        # Boundary layer for the sliding surface
        self.phi = 0.1

        # Filter params + storage values
        self.alpha_filter = 0.7
        self.beta_filter = 0.3
        self.first_run = True

        self.zdot_filtered = 0.0
        self.thetadot_filtered = 0.0

        self.zdot_level = 0.0
        self.thetadot_level = 0.0
        self.zdot_trend = 0.0
        self.thetadot_trend = 0.0

        self.m1 = BP.m1 * self.alpha_uncertainty
        self.m2 = BP.m2 * self.alpha_uncertainty
        self.length = BP.length * self.alpha_uncertainty
        self.g = BP.g


        # # parameters for fx and fy
        # # self.m = (BP.a_val + BP.b_val) * self.alpha_uncertainty
        # self.m_x = BP.a_val * self.alpha_uncertainty
        # self.m_y = BP.b_val * self.alpha_uncertainty
        # self.minv = BP.abinv * self.alpha_uncertainty 
        # self.bx = BP.bx * self.alpha_uncertainty # damping
        # self.by = BP.by * self.alpha_uncertainty  # damping
        # self.cx = BP.cx * self.alpha_uncertainty  # spring
        # self.cy = BP.cy * self.alpha_uncertainty  # spring

    def get_double_ema_velocity(self, zdot_raw, thetadot_raw):
        '''
        Applies Holt's Linear Trend method (Double Smoothing)
        Returns: (filtered_vx, filtered_vy)
        '''
        if self.first_run:
            # Initialize Level to current value, Trend to 0
            self.zdot_level = zdot_raw
            self.thetadot_level = thetadot_raw
            self.zdot_trend = 0.0
            self.thetadot_trend = 0.0
            self.first_run = False
            return zdot_raw, thetadot_raw

        # --- X Direction ---
        # 1. Update Level (Standard EMA logic + adding the previous trend)
        prev_level_x = self.zdot_level
        self.zdot_level = (self.alpha_filter * zdot_raw) + (1 - self.alpha_filter) * (self.zdot_level + self.zdot_trend)

        # 2. Update Trend (Change in level)
        self.zdot_trend = (self.beta_filter * (self.zdot_level - prev_level_x)) + (1 - self.beta_filter) * self.zdot_trend

        # --- Y Direction ---
        prev_level_y = self.thetadot_level
        self.thetadot_level = (self.alpha_filter * thetadot_raw) + (1 - self.alpha_filter) * (self.thetadot_level + self.thetadot_trend)

        self.thetadot_trend = (self.beta_filter * (self.thetadot_level - prev_level_y)) + (1 - self.beta_filter) * self.thetadot_trend

        # The result is Level + Trend (Lag Corrected)
        return (self.zdot_level + self.zdot_trend), (self.thetadot_level + self.thetadot_trend)

    def update (self, ref, state_measured):
        '''
        Docstring for update
        ux = m*(-fx + xdd_desired - lamda_x * exdot - Kx * sat(sx/phi))
        uy = m*(-fy + ydd_desired - lamda_y * eydot - Ky * sat(sy/phi))

        TODO: Change xdd and xdot so they are derived numerically
        Maybe numerical derivation
        
        :param self: Description
        :param ref: Description
        :param refdot: Derivative of reference!!! [xdot, ydot]
        :param refddot: derivative 2 of reference!!! [xddot, yddot]
        :param x: Description 
        '''
        '''
        z    = x[0]
        theta    = x[1]
        zdot = x[2]
        thetadot = x[3]
        '''
        # print(f"state: {x}")
        state = state_measured * (1 + np.random.standard_normal()* 0.005) # Adds a small disturbance value to each measurement

        self.zdot_filtered, self.thetadot_filtered = self.get_double_ema_velocity(state[2],state[3])

        e_z = state[0] - ref[0]
        e_theta = state[1] - ref[1]

        # Original unfiltered
        # e_xdot = x[2] - refdot[0]
        # e_ydot = x[3] - refdot[1]

        e_zdot = self.zdot_filtered
        e_thetadot = self.thetadot_filtered

        s_z = e_zdot + self.lamda_z * e_z
        s_theta = e_thetadot + self.lamda_theta * e_theta

        # # Original
        # fx = self.bx * x[2] - self.cx * x[1] # Known model functions x
        # fy = self.by * x[3] - self.cy * x[0] # Known model functions y

        # f_z = self.bx * self.zdot_filtered - self.cx * state[1] # Known model functions x
        f_theta =  -2 * self.m1 * state[0] * self.zdot_filtered * self.thetadot_filtered - self.m2 * self.g * self.length / 2 - self.m1 * self.g * state[0] * np.cos(state[1])# Known model functions theta... (tau = )
        # f_theta = 0
        # # Original 
        # ux = -fx + self.m_x*(refddot[0] - self.lamda_x * e_xdot - self.K_x * sat(s_x,self.phi))
        # uy = -fy + self.m_y*(refddot[1] - self.lamda_y * e_ydot - self.K_y * sat(s_y,self.phi)) 

        # ux = -fx + self.m_x*(refddot[0] - self.lamda_x * e_xdot - (self.K_x + 1.5*abs(x[2])) * np.tanh(s_x/self.phi))
        # uy = -fy + self.m_y*(refddot[1] - self.lamda_y * e_ydot - (self.K_y + 1.5*abs(x[3])) * np.tanh(s_y/self.phi))

        u_z = 0 #-f_z + self.m_x*(refddot[0] - self.lamda_z * e_zdot - (self.K_z + 5*abs(self.zdot_filtered)) * sat(s_z,self.phi))
        
        # u_theta = -f_theta - self.m_y*(self.lamda_theta * e_thetadot + (self.K_theta + 1.5 * abs(self.thetadot_filtered)) * sat(s_theta,self.phi))
        u_theta = -f_theta - 1*(self.lamda_theta * e_thetadot + (self.K_theta + 1.5 * abs(self.thetadot_filtered)) * sat(s_theta,self.phi))

        # if abs(u_z) > BP.uLimit or abs(u_theta) > BP.uLimit:
        #     print(f"Warning: Saturation! Req: {u_z:.1f}, {u_theta:.1f}")

        return np.clip(u_theta,-BP.tau_max,BP.tau_max)

def sat(s,phi):
    '''
    Saturation function used in SMC
    
    :param s: sliding area value s
    :param phi: boundary layer thickness

    Saturates input based on where you are related to the sliding surface
    '''
    return np.clip(s/phi, -1.0, 1.0)

class LQR:
    def __init__(self, x0=np.zeros(4)):
        A41 = BP.g * (BP.m1**2 *BP.p_steady**2 + BP.m1 * BP.m2 * BP.length**2 / 3) \
            / (BP.m1 * BP.p_steady**2 + BP.m2 * BP.length**2 / 3)**2
        # linearized model used for state prediction xdot = Ax+Bu
        self.A = np.array([[0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, -BP.g, 0.0, 0.0],
                    [-A41*0, 0.0, 0.0, 0.0]])

        self.B = np.array([[0],[0],[0],[1/(BP.m1 * BP.p_steady**2 + BP.m2 * BP.length**2 / 3)]])

        C = np.array([[1,0,0,0],
                    [0,1,0,0]])
        
        # Used for LQR calculation
        Q = np.diag([1.0,0.1,0.1,0.1]) # State cost matrix
        R = np.array([[1]]) # Control cost matrix
        P = solve_continuous_are(self.A,self.B,Q,R)
        self.K_LQR = (np.linalg.inv(R) @ self.B.T @ P).flatten()

        self.refArray = np.zeros(4) # to ease state - ref in the update function
        
        # Use for kalman filter: 
        self.x = x0 # prediction state (initialized at x0)
        self.P_k = np.eye(len(self.x))*0.1 # Covariance matrix
        self.Q_k = np.diag([0.1, 0.01,0.1,0.01])*0.01 # Process noise matrix
        self.R_k = np.diag([0.05, 0.02]) # Measurement covariance
        self.H = C # measurement matrix
        self.K_k = self.P_k @ self.H.T @ (self.H @ self.P_k @ self.H.T + self.R_k) # Initialize kalman gain
        
        self.predict(u=np.array([0]))
        # print(f"After prediction: ")
        # print(f"shape x: {np.shape(self.x)}")
        # print(f"shape P_k: {np.shape(self.P_k)}")
        # print(f"shape Q_k: {np.shape(self.Q_k)}")
        # print(f"shape R_k: {np.shape(self.R_k)}")
        # print(f"shape H: {np.shape(self.H)}")
        # print(f"shape K: {np.shape(self.K_k)}")
        # print()


    def jacobian(self, u):
        # u_val = float(np.asarray(u).flatten()[0]) # Safely extract the scalar value
        u_val = float(u)
        z, theta, zdot, thetadot = self.x
        den = (BP.m1 * z**2 + BP.m2 * BP.length**2 /3)
        A41 = (den*(2 * BP.m1 * zdot - BP.g * BP.m1 * np.cos(theta))\
               -(u_val - 2 * BP.m1 * z * zdot * thetadot - (BP.g * BP.m1 * z - BP.g * BP.m2 * BP.length / 2 )* np.cos(theta))*(2 * BP.m1 * z))\
               / (den**2)
        A42 = (BP.g * BP.m1 * z * np.sin(theta) + BP.g * BP.m2 * BP.length/2*np.sin(theta)) / den
        A43 = (-2 * BP.m1 * z * thetadot) / den
        A44 = (-2 * BP.m1 * z * zdot) / den
        return np.array([[0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0],
                         [thetadot**2, -BP.g * np.cos(theta), 0.0, 2 * z * thetadot],
                         [A41, A42, A43, A44]])
    
    def predict(self,u):
        F = np.eye(4) + self.jacobian(u) * BP.dt

        z, theta, zdot, thetadot = self.x
        zdd = z * thetadot**2 - BP.g * np.sin(theta)
        thetadd = (float(u) - 2 * BP.m1 * z * zdot * thetadot - BP.g * BP.m1 * z * np.cos(theta) - BP.g*BP.m2*BP.length/2*np.cos(theta))/ (BP.m1 * z**2 + BP.m2 * BP.length**2 /3)
        xdot = np.array([zdot, thetadot, zdd, thetadd])
        self.x = self.x + xdot*BP.dt
        # self.x = F @ self.x + (self.B * float(u) * BP.dt).flatten()
        
        self.P_k = F @ self.P_k @ F.T + self.Q_k

    def correct(self, state):
        self.x = self.x + self.K_k @ (state - self.H @ self.x)

        self.K_k = self.P_k @ self.H.T @ np.linalg.inv(self.H @ self.P_k @ self.H.T + self.R_k)

        self.P_k = (np.eye(4) - self.K_k @ self.H) @ self.P_k @ (np.eye(4) - self.K_k @ self.H).T

    def update(self,state,ref):
        self.refArray[0] = ref
        self.correct(state)
        error = self.x - self.refArray
        u_gravity = (BP.m1 * self.x[0] + BP.m2 * BP.length /2) * BP.g * np.cos(self.x[1])
        u = u_gravity - np.dot(self.K_LQR,error)
        self.predict(u)
        return u

class LQRI:
    def __init__(self):
        '''
        Docstring for update
        '''
        # linearized equation + parameters:

        A41 = BP.g * (BP.m1**2 *BP.p_steady**2 + BP.m1 * BP.m2 * BP.length**2 / 3) \
            / (BP.m1 * BP.p_steady**2 + BP.m2 * BP.length**2 / 3)**2
        self.A = np.array([[0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, -BP.g, 0.0, 0.0],
                    [-A41, 0.0, 0.0, 0.0]])

        self.B = np.array([[0],[0],[0],[1/(BP.m1 * BP.p_steady**2 + BP.m2 * BP.length**2 / 3)]])

        self.C = np.array([[1,0,0,0],
                    [0,1,0,0]])
        
        self.A_aug = np.vstack([
                    np.hstack([self.A, np.zeros((4, 1))]), 
                    np.array([-1, 0, 0, 0, 0])      
                ])
        
        self.B_aug = np.vstack([self.B,[0]])

        Q_aug = np.diag([100.0, 1.0, 1.0, 1.0, 10.0]) # High penalty on Integral = Stiff tracking
        R = np.array([[0.1]]) # Penalty on control effort

        self.P = solve_continuous_are(self.A_aug,self.B_aug,Q_aug,R)
        self.K_aug = (np.linalg.inv(R) @ self.B_aug.T @ self.P).flatten()

        # Kalman filter (observer)
        Vd = np.diag([0.1, 0.01, 0.01, 0.01]) 
        Vn = np.diag([0.1, 0.1])
        self.P_est = solve_continuous_are(self.A.T, self.C.T,Vd,Vn)
        self.L = self.P_est @ self.C.T @ np.linalg.inv(Vn)

        print(f"A_aug (shape): \n{np.shape(self.A_aug)}")
        print(f"B_aug (shape): \n{np.shape(self.B_aug)}")
        print(f"LQI Gains: {self.K_aug}")
        print(f"Kalman Gains: {self.L}")

        self.dt = BP.dt
        self.x_hat = np.zeros((4, 1)) # Estimated state
        self.integrator_state = 0.0   # Integral of error
        self.u_prev = 0

        self.tau_max = BP.tau_max

    def update(self, state, ref):
        '''
        Docstring for update
        
        :param self: Description
        :param state: Description
        :param ref: Description
        '''
        # Prediction step (state space model)
        x_pred = self.x_hat + (self.A @ self.x_hat + self.B * self.u_prev) * self.dt
        
        # Correction step
        y_pred = self.C @ x_pred
        residual = state[:2].reshape(-1,1) - y_pred
        self.x_hat = x_pred + self.L @ residual

        # LQI control
        z_est = self.x_hat[0,0]
        error = ref - z_est

        # Anti-windup (optional)
        if abs(self.u_prev) < self.tau_max or np.sign(error) != np.sign(self.u_prev):
            self.integrator_state += error * self.dt
            # TODO - Change to simpson's rule or another method using the previous 4 points (+current) 
            # e.g. to give 4 equal partitions for more accurate numerical integration... 
            # self.integrator_state = self.integrator_state - self.K_aug[-1] * self.integrator_state #- self.Kd_z * self.z_dot
            # print(f"Used")

        # Augmented state: 
        x_aug_current = np.vstack([self.x_hat,[self.integrator_state]])

        # control output: 
        u = -np.dot(self.K_aug,x_aug_current).item()
        self.u_prev = np.clip(u,-self.tau_max,self.tau_max)
        return self.u_prev

class MPC:
    def __init__(self):
        '''
        Docstring for __init__
        
        '''
        pass

    def update(self, state, ref):
        '''
        Docstring for update
        
        :param state: Description
        :param ref: Description
        '''
        pass
