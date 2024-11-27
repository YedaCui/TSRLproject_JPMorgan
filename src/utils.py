import numpy as np
import tensorflow as tf

def leapfrog(get_acceleration, initial_state, dt, num_lf):
    """
    Leapfrog integration.

    Args:
    get_acceleration : python callable which takes an argument "currernt_state" and returns the acceleration at this state.
    initial_state : 'np.array' of the postion and the momentum.
    dt : 'float' of the time step for leapfrog integration.
    num_lf : 'int' Number of the leapfrog steps.
    """
    dim = len(initial_state)
    states = np.zeros(num_lf+1, dim)
    states[0,:] = initial_state
    anew = get_acceleration(initial_state)
    for idx_step in range(1, num_lf+1):
        aold = anew
        states[idx_step,0:dim//2] = states[idx_step-1,0:dim//2] + dt * states[idx_step-1,dim//2:] + 0.5 * dt**2 * aold
        anew = get_acceleration(states[idx_step,:])
        states[idx_step,dim//2:] = states[idx_step-1,dim//2:] + 0.5 * dt * (aold + anew)
    return states
    
def choose_acti(acti):
    """
    Choose activation function.

    Args:
        acti : 'str' the name of activation function.
    """

    if acti == "sin":
        return tf.math.sin
    elif acti == "tanh":
        return tf.nn.tanh
    elif acti == "relu":
        return tf.nn.relu
    
