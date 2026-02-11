

import numpy as np
import blockbeamParam as BP

## Controller classes
class PD:

    def __init__(self):
        '''
        Initializes a PD controller. 

        This will calculate proper PD values using simple
        rise time analysis
        '''

        pass

    def update(self,state,ref):
        '''
        Docstring for update
        
        :param self: Description
        :param state: Description
        :param ref: Description
        '''
        # Calculate derivative (hold past state)

        pass

class PID:
    def __init__(self):
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
        tr_th = 0.2         # Rise time for theta (inner loop) --> This must be FAST to use a simplified model
        zeta_th = 0.8     # Damping ratio
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

        self.Ts = BP.Ts
        self.sigma = 0.05 # cutoff frequency for dirty derivative

        self.m1 = BP.m1
        self.m2 = BP.m2
        self.g = BP.g
        self.length = BP.length

        # saturation limits
        self.Tau_max = BP.torque_Max        # maximum torque (from parameters) 
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
        
        # If z_dot is small integrate z:
        # This ensures it only happens when it's slowed down to cancel out steady state errors
        # if np.abs(self.z_dot) < 0.05:
        #     self.integrator_z += error_z*self.Ts
        #     # self.integrator_z = self.integrator_z + self.Ki_z * self.integrator_z - self.Kd_z * self.z_dot
        
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

        return np.clip(u_theta,-BP.torque_Max,BP.torque_Max)


def sat(s,phi):
    '''
    Saturation function used in SMC
    
    :param s: sliding area value s
    :param phi: boundary layer thickness

    Saturates input based on where you are related to the sliding surface
    '''
    return np.clip(s/phi, -1.0, 1.0)



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
