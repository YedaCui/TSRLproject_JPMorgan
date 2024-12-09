# Code by Yeda CUI at department of SEEM of The Chinese Unviersity of Hong Kong
# Markov Chain Monte Carlo methods

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import uniform, norm
import utils


class MCMC(ABC):
    def __init__(self, initial_state, num_samples, burnin=0, chains=1, seed=0,**kwargs):
        """
        Abstract base class for implementing Markov Chain Monte Carlo.

        Args:
            initial_state : 'array' like object.
            num_samples : 'int' Integer number of samples.
            burnin : 'int' Integer number of burn-in steps.
            cahins : 'int' Integer number of independent chains.
        """
        super(MCMC, self).__init__(**kwargs)
        self.initial_state = initial_state
        self.dim = len(initial_state)
        self.num_samples = num_samples
        self.burnin = burnin
        self.chains = chains
        self.samples = None # records samples
        self.accept = None # records the acceptance satus of proposed samples
        self.seed = seed

    def sample(self):
        """
        Genearte samples.
        """
        self.samples = np.zeros((self.chains, self.num_samples, self.dim))
        self.accept = np.zeros((self.chains, self.num_samples))
        np.random.seed(self.seed)

        self.prep()

        # Sampling loop for each chain
        for idx_chain in range(self.chains):
            self.idx_chain = idx_chain
            print(f"Starting chain {idx_chain+1}/{self.chains}.")
            self.prep_chain()
            # Initialize the state
            cur_state = np.array(self.initial_state)
            samples = np.zeros((self.num_samples, self.dim))
            accept = np.zeros(self.num_samples)

            for idx_sample in range(self.num_samples):
                self.idx_sample = idx_sample
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
                if idx_sample % 1000 == 0:
                    print(f"Completed generating samples {idx_sample}/{self.num_samples}.")
            
            self.samples[idx_chain,:,:] = samples
            self.accept[idx_chain,:] = accept
            self.end_chain()

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
    
    def prep(self):
        """
        Extra preperation before sampling.
        """
        pass

    def prep_chain(self):
        """
        Extra preperation before sampling each chain.
        """
        pass

    def end_chain(self):
        """
        Extra commands after sampling each chain.
        """
        pass

    def get_efficient_sample_size(self):
        ess_list = []
        for chain_idx in range(self.chains):
            # Exclude burn-in samples
            samples_tf = tf.convert_to_tensor(self.samples[chain_idx, self.burnin:, :], dtype=tf.float32)
            ess = tfp.mcmc.effective_sample_size(samples_tf).numpy()
            ess_list.append(ess)
        return ess_list


class HMC(MCMC):
    def __init__(self, initial_state, num_samples, burnin=0, chains=1, L=10, epsilon=0.25, hamiltonain_model=None, H_function=None, seed=0, **kwargs):
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
        super(HMC,self).__init__(initial_state, num_samples, burnin=burnin, chains=chains, seed=seed, **kwargs)
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
            if self.hamiltonian_model:
                cur_state = tf.reshape(tf.convert_to_tensor(cur_state, dtype=tf.float32), [1,self.dim])
                return tf.reshape(self.hamiltonian_model.get_gradient(cur_state),[-1]).numpy()[self.dim//2:]
            else:
                timegrad_fn = utils.get_timegrad_fn(self.H_function)
                return timegrad_fn(cur_state)[self.dim//2:]
        return utils.leapfrog(get_acceleration, cur_state, self.L/self.num_lf, self.num_lf)[-1]
    
    def sample(self):
        super().sample()
        self.samples = self.samples[:,:,0:self.dim//2] # delete the momentum vectors.
    
    def get_acceptance_rate(self, cur_state, new_state):
        H_cur = self.H_function(cur_state)
        H_new = self.H_function(new_state)
        return np.minimum(1,np.exp(H_cur - H_new))


class NUTS(MCMC):
    def __init__(self, initial_state, num_samples, burnin=0, chains=1, epsilon=0.25, threshold_lf=1000, threshold_nn=10, online_error_monitor=True, num_lf=5, hamiltonain_model=None, H_function=None, seed=0, **kwargs):
        """
        Initialize for the No-U-Turn Sampler (NUTS) by Hoffman&Gelman 2014 (including the Online Error Monitoring by Dhulipala et al 2023).

        Args:
            initial_state : array-like object recording the intial state (not including momentum).
            num_samples : 'int' Integer number of samples.
            burnin : 'int' Integer number of burn-in steps.
            cahins : 'int' Integer number of independent chains.
            epsilon : 'float' Integration step size.
            threshold_lf : threshold for numerical gradient leapfrog integration error.
            threshold_nn : threshold for neural network integration error. Once the error exceeds the threshold, use num_lf steps numerical gradient leapforp integration.
            online_error_monitor : 'bool' if True, apply the online error monitor and if the error exceeds the thresholdthreshold_nn, use num_lf steps numerical gradient leapforp integration.
            num_lf : 'int' Number of numerical gradient leapforp integration step before using neural network back.
            hamiltonian_model : python class with method 'get_gradient' calculating the derivative w.r.t. the current state.
            H_funtion : python callable which takes an arguments like "state" and returns the log-density at this state.
        """
        super(NUTS,self).__init__(initial_state, num_samples, burnin=burnin, chains=chains, seed=seed, **kwargs)
        self.initial_state = np.concat([self.initial_state, np.zeros(self.dim)]) # add the momentum vector into the state vector
        self.dim *= 2 # update the state dimension 
        self.epsilon = epsilon
        self.threshold_lf = threshold_lf
        self.threshold_nn = threshold_nn
        self.online_error_monitor = online_error_monitor
        self.num_lf = num_lf
        self.hamiltonian_model = hamiltonain_model
        self.H_function = H_function
        self.acceleration_hnn = self.get_acceleration("hnn")
        self.acceleration_grad = self.get_acceleration("grad")

    def get_acceleration(self, method):
        """
        Calculate the acceleration based on different method.

        Args:
            method : 'str' if 'hnn', by neural network. If 'grad', by gradient.
        """
        if method == "hnn":
            if self.hamiltonian_model:
                def acceleration(cur_state):
                    cur_state = tf.reshape(tf.convert_to_tensor(cur_state, dtype=tf.float32), [1,self.dim])
                    return tf.reshape(self.hamiltonian_model.get_gradient(cur_state),[-1]).numpy()[self.dim//2:]
            else:
                return None
        if method == "grad":
            def acceleration(cur_state):
                timegrad_fn = utils.get_timegrad_fn(self.H_function)
                return timegrad_fn(cur_state)[self.dim//2:]
        return acceleration

    def get_proposal(self, cur_state):
        # initialize the momentum vector.
        for i in range(self.dim//2):
            cur_state[self.dim//2+i] = norm(loc=0,scale=1).rvs()
        H_ini = self.H_function(cur_state)
        u = np.random.uniform(0, np.exp(-H_ini))
        new_state = cur_state
        pos_plus, mom_plus, pos_minus, mom_minus = cur_state[0:self.dim//2], cur_state[self.dim//2:], cur_state[0:self.dim//2], cur_state[self.dim//2:]
        j, n, s = 0, 1, 1

        if self.only_grad:
            self.counter_grad += 1
        if self.counter_grad == self.num_lf:
            self.only_grad = 0
            self.counter_grad = 0
        
        while s==1:
            v = int(np.random.choice([-1,1]))
            if v==1:
                _, _, pos_plus, mom_plus, pos_new, mom_new, nprime, sprime, alpha, nalpha, monitor_val, self.only_grad = self.build_tree(pos_plus, mom_plus, u, v, j, H_ini, self.only_grad)
            else:
                pos_minus, mom_minus, _, _, pos_new, mom_new, nprime, sprime, alpha, nalpha, monitor_val, self.only_grad = self.build_tree(pos_minus, mom_minus, u, v, j, H_ini, self.only_grad)
            
            if (sprime == 1) and (np.random.uniform() < min(1, nprime/n)):
                new_state = np.concat([pos_new, mom_new], axis=0)
            
            n += nprime
            s = int(sprime and self.stop_criterion(pos_minus, mom_minus, pos_plus, mom_plus))
            j += 1
            self.monitor_chain[self.idx_sample] = monitor_val
        return new_state
    
    def sample(self):
        super().sample()
        self.samples = self.samples[:,:,0:self.dim//2] # delete the momentum vectors.
    
    def get_acceptance_rate(self, cur_state, new_state):
        return 1

    @staticmethod
    def stop_criterion(pos_minus, mom_minus, pos_plus, mom_plus):
        return (np.dot(pos_plus-pos_minus, mom_minus)>=0) & (np.dot(pos_plus-pos_minus, mom_plus)>=0)
    
    def prep(self):
        self.monitor = np.zeros((self.chains, self.num_samples))
        self.numgrad = np.zeros(self.chains)
    
    def prep_chain(self):
        self.only_grad = 0
        self.counter_grad = 0
        self.monitor_chain = np.zeros(self.num_samples)
        self.numgrad_chain = 0
    
    def end_chain(self):
        self.monitor[self.idx_chain,:] = self.monitor_chain
        self.numgrad[self.idx_chain] = self.numgrad_chain

    def build_tree(self, pos, mom, u, v, j, H_ini, only_grad):
        """
        Build tree for NUTS incuding the online error monitoring by Dhulipala et al 2023.

        Args:
        pos : start postion.
        mom : start momentum.
        u : sampled uniform random variable.
        v : direction {-1, 1}.
        j : deepth of the tree.
        H_ini : Hamiltonian value of initial state.
        only_grad : {1 , 0} if 1, only use graduation for leapforg integration.
        """
        if j == 0:
            state = np.concat([pos,mom],axis=0)
            monitor = None
            if self.acceleration_hnn:
                state_new = utils.leapfrog(self.acceleration_hnn, state, v*self.epsilon, 1)[-1]
                pos_new, mom_new = state_new[0:self.dim//2], state_new[self.dim//2:]
                H_cur = self.H_function(state_new)
                only_grad = only_grad or int(np.log(u) + H_cur > self.threshold_nn)
                monitor = np.log(u) + H_cur
                sprime = int(np.log(u) + H_cur <= self.threshold_nn)
            if self.online_error_monitor and (only_grad or (self.acceleration_hnn is None)):
                self.numgrad_chain += 1
                state_new = utils.leapfrog(self.acceleration_grad, state, v*self.epsilon, 1)[-1]
                pos_new, mom_new = state_new[0:self.dim//2], state_new[self.dim//2:]
                H_cur = self.H_function(state_new)
                sprime = int(np.log(u) + H_cur <= self.threshold_lf)
            nprime = int(u <= np.exp(-H_cur))
            pos_minus, mom_minus, pos_plus, mom_plus, alphaprime, nalphaprime = pos_new, mom_new, pos_new, mom_new, min(1.0, np.exp(H_cur-H_ini)), 1
        else:
            # Recursion: Implicitly build the height j-1 left and right subtrees.
            pos_minus, mom_minus, pos_plus, mom_plus, pos_new, mom_new, nprime, sprime, alphaprime, nalphaprime, monitor, only_grad = self.build_tree(pos, mom, u, v, j-1, H_ini, only_grad)
            if sprime==1:
                if v==1:
                    _, _, pos_plus, mom_plus, pos_new1, mom_new1, nprime2, sprime2, alphaprime2, nalphaprime2, monitor, only_grad = self.build_tree(pos_plus, mom_plus, u, v, j-1, H_ini, only_grad)
                else:
                    pos_minus, mom_minus, _, _, pos_new1, mom_new1, nprime2, sprime2, alphaprime2, nalphaprime2, monitor, only_grad = self.build_tree(pos_minus, mom_minus, u, v, j-1, H_ini, only_grad)
                # Choose a subtree
                if np.random.uniform() < nprime2 / max(nprime2 + nprime, 1):
                    pos_new, mom_new =  pos_new1, mom_new1
                nprime = nprime + nprime2
                sprime = int(sprime and sprime2 and self.stop_criterion(pos_minus, mom_minus, pos_plus, mom_plus))
                alphaprime = alphaprime + alphaprime2
                nalphaprime = nalphaprime + nalphaprime2
        
        return pos_minus, mom_minus, pos_plus, mom_plus, pos_new, mom_new, nprime, sprime, alphaprime, nalphaprime, monitor, only_grad

