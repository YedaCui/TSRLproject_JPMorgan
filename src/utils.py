# Code by Yeda CUI at department of SEEM of The Chinese Unviersity of Hong Kong

import numpy as np
import tensorflow as tf
from scipy.interpolate import RegularGridInterpolator
import pickle

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
    Returns a function `timegrad_fn` that computes the time derivative of the state 
    (\partial state / \partial t) for a given Hamiltonian system.

    Args:
        H_function : callable
            A Python callable representing the Hamiltonian function. It takes 
            a state `state` as input and returns the Hamiltonian value (log-density).
    """
    def timegrad_fn(state):
        """
        Computes the time derivative of the state (\partial state / \partial t) 
        based on the Hamiltonian system equations.

        Args:
            state : array-like
                A numpy array or TensorFlow tensor representing the current state 
                (positions and momenta concatenated).

        Returns:
            timegrad : numpy array
                The time derivative of the state as a numpy array, where:
                [d(position)/dt, d(momentum)/dt].
        """
        state = tf.convert_to_tensor(state)
        with tf.GradientTape() as tape:
            tape.watch(state)
            H = H_function(state)
        grad = tape.gradient(H,state)
        dim = len(grad)
        return np.concat([grad[dim//2:], -grad[0:dim//2]]) # (d(position)/dt, d(momentum)/dt) =  (∂H/∂momentum, -∂H/∂position)
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

def getZ(seed=0, T=500, n=6, p=8):
    """
    Generate Z from standard normal distribution in the general linear mixture model of section 4.3 in Alenlov 2021.
    """
    np.random.seed(seed)
    return np.random.normal(size=(T,n,p))

def getglmmdata(seed=0, T=500, w = np.array([0.8,0.2]), mu = np.array([0,3]), la = np.array([10, 3]), 
            beta = np.array([-1.1671, 2.4665, -0.1918, -1.0080, 0.6212, 0.6524, 1.5410, 0.2653]), Z = getZ()):
    """
    Get simulated observatons of the general linear mixture model in section 4.3 of Alenlöv et al 2021.
    """
    np.random.seed(seed)
    # generate X
    samples_normal = np.stack([np.random.normal(loc=_mu,scale=1/_la,size=T) for _mu, _la in zip(mu, la)]).T
    X = np.sum(np.random.multinomial(1, w, T) * samples_normal, axis=-1, keepdims=True)
    # generate Y
    p = 1/(1+np.exp(- X - Z @ beta))
    Y = np.random.binomial(1, p=p)
    return Y

def H_A_sol(theta, u, rho, p, dt):
    """
    Get the explicit solution of Hamiltonian A in equation (17) of Alenlov 2021.

    Args:
        theta : array Position variable for Hamiltonian dynamics.
        u : array Auxiliary momentum variable for `theta`.
        rho : array Additional position variable for Hamiltonian dynamics.
        p : array Auxiliary momentum variable for `rho`.
        dt : float Time step for integration.
    """
    theta1 = theta + dt * rho
    rho1 = rho
    u1 = p * np.sin(dt) + u * np.cos(dt)
    p1 = p * np.cos(dt) - u * np.sin(dt)
    return theta1, u1, rho1, p1

def H_B_sol(theta, u, rho, p, pmgrad_fn, dt):
    """
    Get the soltion of Hamiltonian B in equation (18) of Alenlov 2021.

    Args:
        theta : array Position variable for Hamiltonian dynamics.
        u : array Auxiliary momentum variable for `theta`.
        rho : array Additional position variable for Hamiltonian dynamics.
        p : array Auxiliary momentum variable for `rho`.
        pmgrad_fn : callable A function that computes the gradients of the Hamiltonian B 
                              with respect to `rho` and `p`.
        dt : float Time step for integration.
        num_int : int Number of integration steps.
    """
    grad_rho, grad_p = pmgrad_fn(theta, u, rho, p)
    theta1 = theta
    rho1 = rho + dt * grad_rho
    u1 = u
    p1 = p + dt * grad_p
    return theta1, u1, rho1, p1

def getpmgrad_fn(H_B):
    """
    Get the \partial rho / \partial t and \partial p / \partial t in equation (16) of Alenlov 2021.
    """
    def pmgrad_fn(theta, u, rho, p):
        """
        Args:
            theta : array Position variable for Hamiltonian dynamics.
            u : array Auxiliary momentum variable for `theta`.
            rho : array Additional position variable for Hamiltonian dynamics.
            p : array Auxiliary momentum variable for `rho`.
        """
        state = tf.convert_to_tensor(tf.concat([theta,u,rho,p],axis=0))
        with tf.GradientTape() as tape:
            tape.watch(state)
            H = H_B(state)
        grad = tape.gradient(H,state)
        grad_rho, grad_p = -grad[0:len(theta)], -grad[len(theta):len(u)+len(theta)]
        grad_rho, grad_p = np.array(grad_rho), np.array(grad_p)
        return grad_rho, grad_p
    return pmgrad_fn

def pmintegrator(theta,u,rho,p, pmgrad_fn, dt, num_int, require_grads=False):
    """
    Implement the splitting operator pseudo-marginal HMC integrator as described in Alenlov 2021.
    
    The function integrates the system using a splitting method. It alternates between 
    solving for Hamiltonian A (H_A) and Hamiltonian B (H_B) over the given number of 
    integration steps, `num_int`.

    Args:
        theta : array Position variable for Hamiltonian dynamics.
        u : array Auxiliary momentum variable for `theta`.
        rho : array Additional position variable for Hamiltonian dynamics.
        p : array Auxiliary momentum variable for `rho`.
        pmgrad_fn : callable A function that computes the gradients of the Hamiltonian B 
                              with respect to `rho` and `p`.
        dt : float Time step for integration.
        num_int : int Number of integration steps.
        require_grads : bool If True, returns the gradients of Hamiltonian B at each 
                              integration step for diagnostic purposes.

    Returns:
        states : array An array of shape (num_int + 1, len(theta) + len(u) + len(rho) + len(p))
                        containing the state vectors at each integration step.
        If `require_grads` is True:
            states_for_grads (array): An array containing the state vectors where gradients were computed.
            time_grads (array): An array of gradients of Hamiltonian B at each step.

    Integration Details:
        1. The function uses a symmetric splitting operator:
            - First, solve for H_A for half a time step (`dt/2`).
            - Then, solve for H_B for a full time step (`dt`).
            - Finally, solve for H_A for another half time step (`dt/2`).
        2. At each step, the auxiliary variable `u` is clipped to avoid excessively large values.
        3. The system state is stored at each step for tracking.
    """
    states = np.zeros((num_int+1, len(np.concat([theta,u,rho,p], axis=0))))
    states[0,:] = np.concat([theta,u,rho,p], axis=0)
    states_for_grads, time_grads = [],[]
    for i in range(num_int):
        theta, u, rho, p = H_A_sol(theta, u, rho, p, dt/2)
        u = np.clip(u, -30, 30)
        if require_grads:
            states_for_grads.append(np.concat([theta,u,rho,p], axis=0))
            time_grads.append(np.concat(pmgrad_fn(theta, u, rho, p), axis=0))
        theta, u, rho, p = H_B_sol(theta, u, rho, p, pmgrad_fn, dt)
        u = np.clip(u, -30, 30)
        theta, u, rho, p = H_A_sol(theta, u, rho, p, dt/2)
        u = np.clip(u, -30, 30)
        states[i+1,:] = np.concat([theta,u,rho,p], axis=0)
    if require_grads:
        return states, np.stack(states_for_grads), np.concat([np.zeros((len(time_grads), len(theta)+len(u))), np.stack(time_grads)],axis=1)
    else:
        return states

def get_marginal_initial():
    """
    Get the initial marginal variables for PMHMC in section 4.3 of Alenlov 2021.
    """
    return np.array([0.5838, 0.3805, -1.5062, -0.0442, 0.4717, -0.1435, 0.6371, -0.0522, 0, 0, 1, 1, 0.5])
    # return np.array([0.5838, 0.3805, -1.5062, -0.0442, 0.4717, -0.1435, 0.6371, -0.0522, 0, 3, np.log(10), np.log(3), 0.8])

def get_latent_u(dim_u, seed=0):
    """
    Get the initial u variables for PMHMC in section 4.3 of Alenlov 2021.
    
    Args:
        dim_u : int Number of the dimension of u.
    """
    np.random.seed(seed)
    return np.random.normal(size=dim_u)