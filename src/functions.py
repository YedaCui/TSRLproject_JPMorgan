import tensorflow as tf
import utils


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
            term1 = -tf.math.log(0.5*(tf.math.exp(-(q-mu1)**2/(2*sigma**2)))+0.5*(tf.math.exp(-(q-mu2)**2/(2*sigma**2))))
            H = term1 + p**2/2 # Normal PDF
            return H
    #*********** nD_Rosenbrock #**************
    elif dist_name == 'nD_Rosenbrock':
        def H_function(state):
            dim = len(state)//2
            q, p = state[0:dim], state[dim:]
            term1 = tf.reduce_sum((100 * (q[1:] - q[:-1]**2)**2 + (1 - q[:-1])**2) / 20)
            term2 = tf.reduce_sum(p**2/2)
            H = term1 + term2
            return H
    elif dist_name == "2D_Nealsfunnel":
        def H_function(state):
            q1, q2, p = state[0], state[1], state[2:]
            term1 = q1**2/(2*3**2) + q2**2/(2*tf.math.exp(q1))
            term2 = tf.reduce_sum(p**2/2)
            H = term1 + term2
            return H
    elif dist_name == "5D_illconditioned_Gaussian":
        def H_function(state):
            dim = len(state)//2
            q, p = state[0:dim], state[dim:]
            var = tf.constant([0.01, 0.1, 1, 10, 100], dtype=tf.float64)
            term1 = tf.reduce_sum(q**2/(2*var))
            term2 = tf.reduce_sum(p**2/2)
            H = term1 + term2
            return H
    elif dist_name == "AllenCahn":
        def H_function(state):
            dim = len(state)//2
            q, p = state[0:dim], state[dim:]
            def V(x):
                return (1-x**2)**2
            dx = 1/25
            term1 = tf.reduce_sum((q[1:] - q[:-1])**2/(2*dx) + dx/2 * (V(q[1:]) + V(q[:-1])))
            term2 = tf.reduce_sum(p**2/2)
            H = term1 + term2
            return H
    elif dist_name == "ellipticpde":
        def H_function(state):
            dim = len(state)//2
            q, p = state[0:dim], state[dim:]
            sensor = utils.get_pdesensor()
            fval,_ = utils.get_pdef()
            k = tf.matmul(tf.linalg.inv(tf.matmul(tf.transpose(sensor), sensor)), tf.matmul(tf.transpose(sensor),tf.reshape(q,shape=(-1,1))))
            x, y = sensor[:,0], sensor[:,1]
            term1 = tf.reduce_sum((k[0,0] * 2 * tf.math.cos(2*x) - q * 4 * tf.math.sin(2*x) + k[1,0] * 2 * tf.cos(2*y) - q * 4 * tf.math.sin(2*y) - fval) ** 2 / 2)
            term2 = tf.reduce_sum(p**2/2)
            H = term1 + term2
            return H
    elif dist_name == "pmglmmA":
        def H_function(state):
            # returns the value of Hamiltonian A of equatioin (16) in Alenlov et al 2021.
            dim = len(state)//2
            theta, u, rho, p = state[0:13], state[13:dim], state[dim:dim+13], state[dim+13:]
            return tf.reduce_sum(rho**2/2) + tf.reduce_sum(u**2/2) + tf.reduce_sum(p**2/2)
    elif dist_name == "pmglmmB":
        def H_function(state):
            # returns the value of Hamiltonian B of equatioin (16) in Alenlov et al 2021.
            dim = len(state)//2
            theta, u, rho, p = state[0:13], state[13:dim], state[dim:dim+13], state[dim+13:]
            part1 = tf.reduce_sum(theta**2/2)
            part2 = -tf.math.log()

    elif dist_name == "pmglmm":
        def H_function(state):
            # returns the value of Hamiltonian H = A + B of equatioin (16) in Alenlov et al 2021.
            dim = len(state)//2
            theta, u, rho, p = state[0:13], state[13:dim], state[dim:dim+13], state[dim+13:]
            H_A = tf.reduce_sum(rho**2/2) + tf.reduce_sum(u**2/2) + tf.reduce_sum(p**2/2)


    else:
        raise ValueError("probability distribution name not recognized")

    return H_function