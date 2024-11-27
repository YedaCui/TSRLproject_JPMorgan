import numpy as np


def functions(dist_name):
    """
    Return the function which calculates the log-density.

    Args:
        dist_name : 'str' the name of the disribution.
    """
    
    #******** 1D Gaussian Mixture #********
    if dist_name == '1D_Gauss_mix':
        def H_function(state):
            q, p = state[0], state[1]
            mu1 = 1.0
            mu2 = -1.0
            sigma = 0.35
            term1 = -np.log(0.5*(np.exp(-(q-mu1)**2/(2*sigma**2)))+0.5*(np.exp(-(q-mu2)**2/(2*sigma**2))))
            H = term1 + p**2/2 # Normal PDF
            return H

    # #******** 2D Gaussian Four Mixtures #********
    elif dist_name == '2D_Gauss_mix':
        def H_function(state):
            q1, q2, p1, p2 = state[0], state[1], state[2], state[3]
            sigma_inv = np.array([[1.,0.],[0.,1.]])
            term1 = 0.
            
            mu = np.array([3.,0.])
            y = np.array([q1-mu[0],q2-mu[1]])
            tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
            term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
            
            mu = np.array([-3.,0.])
            y = np.array([q1-mu[0],q2-mu[1]])
            tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
            term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
            
            mu = np.array([0.,3.])
            y = np.array([q1-mu[0],q2-mu[1]])
            tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
            term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
            
            mu = np.array([0.,-3.])
            y = np.array([q1-mu[0],q2-mu[1]])
            tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
            term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
            
            term1 = -np.log(term1)
            term2 = p1**2/2+p2**2/2
            H = term1 + term2
            return H
    else:
        raise ValueError("probability distribution name not recognized")

    return H_function