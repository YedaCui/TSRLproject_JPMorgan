import pytest
from src.mcmc import *
import tensorflow as tf
import numpy as np
from src.hnn import HNN
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_visible_devices(gpus[0], "GPU")


class MetropolisHastings(MCMC):
    def __init__(self, initial_state, num_samples, burnin=0, chains=1, seed=0, target_density=None):
        """
        Subclass of MCMC implementing the Metropolis-Hastings algorithm.

        Args:
            target_density (callable): A function that computes the target density (up to a normalizing constant).
        """
        super().__init__(initial_state, num_samples, burnin, chains, seed)
        self.target_density = target_density

    def get_proposal(self, cur_state):
        proposal = np.random.normal(loc=cur_state, scale=1.0, size=self.dim)  # Gaussian proposal
        return proposal

    def get_acceptance_rate(self, cur_state, new_state):
        target_cur = self.target_density(cur_state)
        target_new = self.target_density(new_state)
        return min(1.0, target_new / target_cur)

# # Define a simple target density for testing (e.g., a standard normal distribution)
# def target_density(x):
#     return np.exp(-0.5 * np.sum(x**2))

# def test_mcmc():
#     initial_state = np.array([0.0])
#     num_samples = 1000
#     # Instantiate the Metropolis-Hastings sampler
#     sampler = MetropolisHastings(
#         initial_state=initial_state,
#         num_samples=num_samples,
#         burnin=100,
#         chains=1,
#         seed=42,
#         target_density=target_density,
#     )
#     sampler.sample()
#     samples = sampler.samples
#     acceptance_rates = sampler.accept
#     assert samples.shape == (1, num_samples, len(initial_state)), "Samples shape is incorrect"
#     assert acceptance_rates.shape == (1, num_samples), "Acceptance rates shape is incorrect"
#     # Check that samples are not all the same (indicating movement in the chain)
#     assert not np.all(samples[0, :, 0] == samples[0, 0, 0]), "Samples are not moving"
#     # Check that samples are not NaN or Inf
#     assert not np.any(np.isnan(samples)), "Samples contain NaN values!"
#     assert not np.any(np.isinf(samples)), "Samples contain Inf values!"
#     # Check effective sample size (ESS)
#     ess_list = sampler.get_efficient_sample_size()
#     assert all(ess > 0 for chain_ess in ess_list for ess in chain_ess), "ESS should be positive"

# def test_hmc():
#     initial_state = np.array([0.0, 1.0]) 
#     num_samples = 20
#     burnin = 10 
#     chains = 1
#     L = 5
#     dt = 0.1
#     # Define a target distribution
#     def target_distribution(state):
#         return 0.5 * tf.reduce_sum(state ** 2)
#     # Instantiate the HMC sampler
#     sampler = HMC(
#         initial_state=initial_state,
#         num_samples=num_samples,
#         burnin=burnin,
#         chains=chains,
#         L=L,
#         epsilon=dt,
#         hamiltonain_model=None,
#         H_function=target_distribution,
#     )
#     sampler.sample()
#     # Test 1: Samples contain no NaN or Inf values
#     samples = sampler.samples
#     assert not np.any(np.isnan(samples)), "Samples contain NaN values!"
#     assert not np.any(np.isinf(samples)), "Samples contain Inf values!"
#     assert samples.shape == (chains, num_samples, len(initial_state)), "Samples have an incorrect shape!"
#     # Test 2: Check effective sample size (ESS)
#     ess_list = sampler.get_efficient_sample_size()
#     assert all(ess > 0 for chain_ess in ess_list for ess in chain_ess), "ESS should be positive"
#     # Test 3: Momentum variables are removed from the final samples
#     assert samples.shape[2] == len(initial_state), "Momentum variables were not removed!"
#     # Test 4: Check that samples are not all the same (indicating movement in the chain)
#     assert not np.all(samples[0, :, 0] == samples[0, 0, 0]), "Samples are not moving"

# def test_nuts():
#     """
#     Test the NUTS class, including its build_tree and sampling logic.
#     """
#     initial_state = np.array([0.0, 1.0])
#     num_samples = 20 
#     burnin = 10 
#     chains = 1
#     epsilon = 0.1
#     threshold_lf = 10
#     threshold_nn = 10
#     num_lf = 5
#     # Define a target distribution
#     def target_distribution(state):
#         return 0.5 * tf.reduce_sum(state ** 2)
#     # Instantiate the NUTS sampler
#     sampler = NUTS(
#         initial_state=initial_state,
#         num_samples=num_samples,
#         burnin=burnin,
#         chains=chains,
#         epsilon=epsilon,
#         threshold_lf=threshold_lf,
#         threshold_nn=threshold_nn,
#         num_lf=num_lf,
#         hamiltonain_model=HNN(input_dim=4, num_hidden=5, num_layers=2, output_dim=4),
#         H_function=target_distribution,
#     )
#     # Mock acceleration functions
#     sampler.acceleration_hnn = sampler.get_acceleration(method="hnn")
#     sampler.acceleration_grad = sampler.get_acceleration(method="grad")
#     # Test 1: Ensure `build_tree` generates valid outputs
#     pos = np.array([0.0, 1.0])
#     mom = np.array([0.5, -0.5])
#     u = np.random.uniform(0, 1)
#     v = 1  # Direction
#     j = 0  # Tree depth
#     H_ini = target_distribution(np.concatenate([pos, mom]))  # Initial Hamiltonian
#     only_grad = 0
#     tree_output = sampler.build_tree(pos, mom, u, v, j, H_ini, only_grad)
#     # Validate tree output
#     assert len(tree_output) == 12, "build_tree output does not have the correct structure!"
#     pos_minus, mom_minus, pos_plus, mom_plus, pos_new, mom_new, nprime, sprime, alphaprime, nalphaprime, monitor, only_grad = tree_output
#     assert pos_new.shape == pos.shape, "Position shape mismatch in build_tree!"
#     assert mom_new.shape == mom.shape, "Momentum shape mismatch in build_tree!"
#     assert nprime >= 0, "nprime should be non-negative!"
#     assert sprime in [0, 1], "sprime should be 0 or 1!"
#     assert 0 <= alphaprime <= 1, "alphaprime must be between 0 and 1!"
#     # Test 2: Ensure samples are generated without NaN or Inf
#     sampler.sample()
#     samples = sampler.samples
#     assert not np.any(np.isnan(samples)), "Samples contain NaN values!"
#     assert not np.any(np.isinf(samples)), "Samples contain Inf values!"
#     assert samples.shape == (chains, num_samples, len(initial_state)), "Samples have an incorrect shape!"
#     # Test 3: Ensure momentum variables are removed from the final samples
#     assert samples.shape[2] == len(initial_state), "Momentum variables were not removed!"
#     # Test 4: Check that samples are not all the same (indicating movement in the chain)
#     assert not np.all(samples[0, :, 0] == samples[0, 0, 0]), "Samples are not moving"

# def test_pmhmc():
#     """
#     Test the PMHMC class, including its proposal generation, sampling, and acceptance rate logic.
#     """
#     # Define parameters for the test
#     initial_state = np.array([0.0, 1.0, 2.0])  # Initial state for theta, latent variables, and momentum
#     dim_marginal = 2
#     num_samples = 20
#     burnin = 10
#     chains = 1 
#     L = 5
#     epsilon = 0.1
#     H_B=lambda state: 0.5*tf.reduce_sum(state[:len(state)//2]**2)
#     # Instantiate the PMHMC sampler
#     sampler = PMHMC(
#         initial_state=initial_state,
#         dim_marginal=dim_marginal,
#         num_samples=num_samples,
#         burnin=burnin,
#         chains=chains,
#         L=L,
#         epsilon=epsilon,
#         hamiltonain_model=None,
#         H_function=lambda state: 0.5*tf.reduce_sum(state**2) + 0.5*state[2]**2,
#         H_B=H_B,
#     )
#     sampler.pmgrad_fn = utils.getpmgrad_fn(H_B)
#     sampler.pmintegrator = utils.pmintegrator
#     # Test 1: Ensure `get_proposal` generates valid outputs
#     cur_state = np.concatenate([initial_state, np.zeros_like(initial_state)])  # Include momentum
#     proposal = sampler.get_proposal(cur_state)
#     assert len(proposal) == sampler.dim, "Proposal state does not have the correct dimension!"
#     assert not np.any(np.isnan(proposal)), "Proposal state contains NaN values!"
#     assert not np.any(np.isinf(proposal)), "Proposal state contains Inf values!"
#     # Test 2: Ensure `get_acceptance_rate` computes valid probabilities
#     new_state = sampler.get_proposal(cur_state)
#     acceptance_rate = sampler.get_acceptance_rate(cur_state, new_state)
#     assert 0 <= acceptance_rate <= 1, "Acceptance rate is out of bounds!"
#     # Test 3: Ensure samples are generated without NaN or Inf
#     sampler.sample()
#     samples = sampler.samples
#     assert not np.any(np.isnan(samples)), "Samples contain NaN values!"
#     assert not np.any(np.isinf(samples)), "Samples contain Inf values!"
#     assert samples.shape == (chains, num_samples, dim_marginal), "Samples have an incorrect shape!"
#     # Test 4: Ensure latent variables and momentum are removed from the final samples
#     assert samples.shape[2] == dim_marginal, "Latent variables or momentum were not removed from final samples!"
#     # Test 5: Check that samples are not all the same (indicating movement in the chain)
#     assert not np.all(samples[0, :, 0] == samples[0, 0, 0]), "Samples are not moving"
