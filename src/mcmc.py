# Code by Yeda CUI at department of SEEM of The Chinese Unviersity of Hong Kong
# Markov Chain Monte Carlo methods

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import uniform, norm
import utils

class MCMC(ABC):
    def __init__(self, initial_state, num_samples, burnin=0, chains=1):
        """
        Abstract base class for implementing Markov Chain Monte Carlo.

        Args:
            initial_state : 'array' like object.
            num_samples : 'int' Integer number of samples.
            burnin : 'int' Integer number of burn-in steps.
            cahins : 'int' Integer number of independent chains.
        """
        self.initial_state = initial_state
        self.dim = len(initial_state)
        self.num_samples = num_samples
        self.burnin = burnin
        self.chains = chains
        self.samples = None # records samples
        self.accept = None # records the acceptance satus of proposed samples

    def sample(self):
        """
        Genearte samples.
        """
        self.samples = np.zeros((self.chains, self.num_samples, self.dim))
        self.accept = np.zeros((self.chains, self.num_samples))

        # Sampling loop for each chain
        for idx_chain in range(self.chains):
            print(f"Starting chain {idx_chain}/{self.chains}.")
            # Initialize the state
            cur_state = np.array(self.initial_state)
            samples = np.zeros(self.num_samples, self.dim)
            accept = np.zeros(self.num_samples)

            for idx_sample in range(self.num_samples):
                # Generate the proposed state
                new_state = self.get_proposal(cur_state)
                # Calculate the accetance rate alpha
                alpha = self.get_acceptance_rate(cur_state, new_state)
                # Genrate Uniform random variable
                u = uniform().rvs()

                if u < alpha:
                    # accept the new state
                    cur_state = new_state
                    samples[idx_sample,:] = new_state
                    accept[idx_sample] = 1
                else:
                    # reject the new state
                    samples[idx_sample,:] = cur_state
            
            self.samples[idx_chain,:,:] = samples
            self.accept[idx_chain,:] = accept

    @abstractmethod
    def get_proposal(self, cur_state):
        """
        Calculate the proposed state based on the current state.

        Args:
            cur_state : current state vector.
        """
        return
    
    @abstractmethod
    def get_acceptance_rate(self, cur_state, new_state):
        """
        Calculate the acceptance rate.

        Args:
            cur_state : current state vector.
            new_state : proposed state vevtor.
        """
        return

    def get_efficient_sample_size(self):
        ess_list = []
        for chain_idx in range(self.chains):
            # Exclude burn-in samples
            samples_tf = tf.convert_to_tensor(self.samples[chain_idx, self.burn_in:, :], dtype=tf.float32)
            ess = tfp.mcmc.effective_sample_size(samples_tf).numpy()
            ess_list.append(ess)
        return ess_list


class HMC(MCMC):
    def __init__(self, initial_state, num_samples, burnin=0, chains=1, L=10, epsilon=0.25, hamiltonain_model=None, H_function=None):
        """
        Initialize for the Hamiltoanian Monte Carlo.

        Args:
            initial_state : array-like object recording the intial state (not including momentum).
            num_samples : 'int' Integer number of samples.
            burnin : 'int' Integer number of burn-in steps.
            cahins : 'int' Integer number of independent chains.
            L : 'int' or 'float' Time length.
            epsilon : 'float' Integration step size.
            hamiltonian_model : python class with method 'get_gradient' calculating the derivative w.r.t. the current state.
            H_funtion : python callable which takes an arguments like "state" and returns the log-density at this state.
        """
        super().__init__(initial_state, num_samples, burnin=burnin, chains=chains)
        self.initial_state = np.concat([self.initial_state, np.zeros(self.dim)]) # add the momentum vector into the state vector
        self.dim *= 2 # update the state dimension 
        self.L = L
        self.epsilon = epsilon
        self.num_lf = int(L/epsilon) # number of leapfrog steps
        self.hamiltonian_model = hamiltonain_model
        self.H_function = H_function

    def get_proposal(self, cur_state):
        # initialize the momentum vector.
        for i in range(self.dim//2):
            cur_state[self.dim//2+i] = norm(loc=0,scale=1).rvs()
        # define the get_acceleration function which is the -\partial H/ \partial position.
        def get_acceleration(cur_state):
            """
            Calculate the acceleration vector.
            """
            cur_state = tf.reshape(tf.convert_to_tensor(cur_state, dtype=tf.float32), [1,self.dim])
            return tf.reshape(self.hamiltonian_model.get_gradient(cur_state),[-1]).numpy()[self.dim//2:]
        return utils.leapfrog(get_acceleration, cur_state, self.L/self.num_lf, self.num_lf)[-1]
    
    def sample(self):
        super().sample()
        self.samples = self.samples[:,:,0:self.dim//2] # delete the momentum vectors.
    
    def get_acceptance_rate(self, cur_state, new_state):
        H_cur = self.H_function(cur_state)
        H_new = self.H_function(new_state)
        return np.minimum(1,np.exp(H_cur - H_new))