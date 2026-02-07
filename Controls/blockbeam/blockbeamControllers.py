

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
        
        This will calculate proper PID values using simple 
        rise time analysis
        '''
        




        pass

    def update(self,state,ref):
        '''
        Docstring for update
        

        :param state: Description
        :param ref: Description
        '''
        pass

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
