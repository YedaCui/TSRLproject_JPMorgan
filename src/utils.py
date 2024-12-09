import numpy as np
import tensorflow as tf
from scipy.interpolate import RegularGridInterpolator

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
    states = np.zeros((num_lf+1, dim))
    states[0,:] = initial_state
    anew = get_acceleration(initial_state) # Initialize the acceleration.
    for idx_step in range(1, num_lf+1):
        aold = anew
        
        states[idx_step,0:dim//2] = states[idx_step-1,0:dim//2] + dt * states[idx_step-1,dim//2:] + 0.5 * dt**2 * aold # Update the postion.
        anew = get_acceleration(states[idx_step,:]) # Update the acceleration.
        states[idx_step,dim//2:] = states[idx_step-1,dim//2:] + 0.5 * dt * (aold + anew) # Update the momentum.
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

def get_timegrad_fn(H_function):
    """
    Returns the timegrad_fn which takes an argument 'state' and return the \partial state / \partial t at this state.

    Args:
        H_function : python callable which takes an argument "state" and returns the log-density.
    """
    def timegrad_fn(state):
        # Calculate the \partial state / \partial t at this state.
        state = tf.convert_to_tensor(state)
        with tf.GradientTape() as tape:
            tape.watch(state)
            H = H_function(state)
        grad = tape.gradient(H,state)
        dim = len(grad)
        return np.concat([grad[dim//2:], -grad[0:dim//2]])
    return timegrad_fn


def get_trajectory(timegrad_fn, initial_state, dt, num_lf, require_grads=True):
    """
    Calculate the trajectory of Hamiltonian system.

    Args:
        timegrad_fn : python callable which takes an argument 'state' and return the \partial state / \partial t at this state.
        initial_state : 'np.array' of the postion and the momentum.
        dt : 'float' of the time step for leapfrog integration.
        num_lf : 'int' Number of the leapfrog steps.
        require_grads : 'bool' If True, return the time gradients at each points.
    """
    def get_acceleartion(state):
        dim = len(state)
        return timegrad_fn(state)[dim//2:]

    t_eval = np.linspace(0, num_lf*dt, num_lf+1)
    states = leapfrog(get_acceleartion, initial_state, dt, num_lf)
    
    if require_grads:
        timegrad = [timegrad_fn(states[idx_lf,:]) for idx_lf in range(num_lf+1)]
        timegrad = np.stack(timegrad)
        return states, timegrad, t_eval
    else:
        return states, t_eval

def get_pdesensor(num=50,seed=0):
    """
    Get the sensor in the 2D elliptic pde experiments in Dhulipala et al 2022.

    Args:
        num : 'int' the integer number of the sensor.
    """
    np.random.seed(seed)
    return np.random.uniform(size=(num,2), low=0, high=3)

def get_pdef(seed=0):
    """
    Get the f function values with corruption at the 50 sensors in the 2D elliptic pde experiments in Dhulipala et al 2022.
    """
    np.random.seed(seed)
    x, y = np.linspace(0, 3, 301), np.linspace(0, 3, 301)
    xg, yg = np.meshgrid(x,y, indexing='ij', sparse=True)
    fg = 2 * np.cos(2*xg) - (xg + yg) * 4 * np.sin(2*xg) + 2 * np.cos(2*yg) - (xg + yg) * 4 * np.sin(2*yg)
    fg = fg + np.random.normal(size=fg.shape)
    
    sensor = get_pdesensor(seed=seed)
    fval = RegularGridInterpolator((x,y), fg)(sensor)
    return fval, (x,y, fg)

# def get
    