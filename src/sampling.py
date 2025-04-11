# Code by Yeda CUI at department of SEEM of The Chinese Unviersity of Hong Kong

from . import hnn, mcmc, utils, functions, logdensity
from .configs import CONFIGS
import numpy as np
import tensorflow as tf
import pickle
import sys
sys.path.append("/home/ycui/Documents/TSRLproject_JPMorgan/custom_bins")
import tensorflow_probability.python.mcmc as tfpmcmc

def sampling(config_name, sampler_type, initial_state, num_samples, burnin, chains=1, epsilon=0.05, model_path=None, seed=0, sample_path=None, **kwargs):
    """
    Perform sampling using MCMC methods.

    This function initializes and runs an MCMC sampler (e.g., HMC, NUTS, PMHMC) based on the provided configuration
    and sampler type. If a pre-trained Neural Network model is provided, it will be used to approximate the time gradients
    during the sampling process.

    Args:
        config_name (str): Name of the configuration to use. This is used to fetch parameters from the global `CONFIGS`.
        sampler_type (str): Type of sampler to use. Supported types are "HMC", "NUTS", and "PMHMC".
        initial_state (array-like): Initial state for the sampler (e.g., position and momentum).
        num_samples (int): Number of samples to generate.
        burnin (int): Number of burn-in samples to discard.
        chains (int): Number of chains to run in parallel. Default is 1.
        epsilon (float): Step size for integration.
        model_path (str, optional): Path to a pre-trained neural network model file. If provided, the model will be loaded.
        seed (int): Random seed for reproducibility. Default is 0.
        sample_path (str, optional): Path to a file containing precomputed samples. If provided, the sampler will load these
                                     samples instead of running the MCMC process.

    Returns:
        sampler (object): The MCMC sampler object after sampling.
    """
    config = CONFIGS[config_name]
    if model_path:
        model = hnn.HNN(config["input_dim"], config["num_hidden"], config["num_layers"], config["output_dim"], acti=config["acti"], baseline=config["baseline"], field_type=config["field_type"])
        model.load_weights(model_path)
    else:
        model = None
    unnormalized_log_prob = logdensity.get_log_density_fn(config["dist_name"])

    if sampler_type == "HMC":
        adaptive_sampler = tfpmcmc.SimpleStepSizeAdaptation(
            tfpmcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=unnormalized_log_prob,
                num_leapfrog_steps=3,
                step_size=epsilon,
                model=model),
            num_adaptation_steps=int(burnin * 0.8))
        # sampler = tfpmcmc.HamiltonianMonteCarlo(initial_state=initial_state, num_samples=num_samples, burnin=burnin, chains=chains, epsilon=epsilon, hamiltonain_model=model, H_function=functions.functions(config["dist_name"]),seed=seed, **kwargs)
    # elif sampler_type == "NUTS":
    #     sampler = mcmc.NUTS(initial_state=initial_state, num_samples=num_samples, burnin=burnin, chains=chains, epsilon=epsilon, hamiltonain_model=model, H_function=functions.functions(config["dist_name"]),seed=seed, **kwargs)
    # elif sampler_type == "PMHMC":
    #     sampler = mcmc.PMHMC(initial_state=initial_state, num_samples=num_samples, burnin=burnin, chains=chains, epsilon=epsilon, hamiltonain_model=model, H_function=functions.functions(config["dist_name"]),H_B=functions.functions(config["dist_name_B"]), seed=seed, **kwargs)
    else:
        raise ValueError("The sampler type is not supported.")
    
    @tf.function
    def run_chain():
        # Run the chain (with burn-in).
        samples, is_accepted = tfpmcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=burnin,
            current_state=1.,
            kernel=adaptive_sampler,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

        return samples, is_accepted
    if sample_path is None:
        samples, is_accepted = run_chain()
    else:
        with open(sample_path, "rb") as f:
            samples = pickle.load(f)
        # for key, value in data_samples.items():
        #     setattr(sampler, key, value)
    return samples

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(gpus[2], "GPU")

    samples = sampling(config_name="LHNN_1DGaussianmixture", sampler_type="HMC", initial_state=np.array([0]), num_samples=5000, burnin=1000, chains=1, epsilon=0.05, 
             model_path="/home/ycui/Documents/TSRLproject_JPMorgan/models/LHNN_1DGaussianmixture_sin.weights.h5", L=5)
    sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/lhnn_hmc_LHNN_1DGaussianmixture.pkl"

    # sampler = sampling(config_name="LHNN_1DGaussianmixture", sampler_type="HMC", initial_state=np.array([0]), num_samples=5000, burnin=1000, chains=1, epsilon=0.05, 
    #          model_path=None, L=5)
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/grad_hmc_LHNN_1DGaussianmixture.pkl"

    # sampler = sampling(config_name="LHNN_1DGaussianmixture", sampler_type="NUTS", initial_state=np.array([0]), num_samples=5000, burnin=1000, chains=1, epsilon=0.05, 
    #          model_path="/home/ycui/Documents/TSRLproject_JPMorgan/models/LHNN_1DGaussianmixture_sin.weights.h5", threshold_nn=1000, threshold_lf=1000, online_error_monitor=False) # without online error monitoring.
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/lhnn_nuts_LHNN_1DGaussianmixture.pkl"

    # sampler = sampling(config_name="LHNN_1DGaussianmixture", sampler_type="NUTS", initial_state=np.array([0]), num_samples=5000, burnin=1000, chains=1, epsilon=0.05, 
    #          model_path=None, threshold_nn=np.inf, threshold_lf=1000)
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/grad_nuts_LHNN_1DGaussianmixture.pkl"
    
    # sampler = sampling(config_name="LHNN_3DRosenbrock", sampler_type="NUTS", initial_state=np.array([0,0,0]), num_samples=5000, burnin=1000, chains=1, epsilon=0.025, 
    #          model_path="/home/ycui/Documents/TSRLproject_JPMorgan/models/LHNN_3DRosenbrock_sin.weights.h5", threshold_nn=1000, threshold_lf=1000, online_error_monitor=False) # without online error monitoring.
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/lhnn_nuts_3Drb_notoem.pkl"

    # sampler = sampling(config_name="LHNN_3DRosenbrock", sampler_type="NUTS", initial_state=np.array([0,0,0]), num_samples=35000, burnin=1000, chains=1, epsilon=0.025, 
    #          model_path="/home/ycui/Documents/TSRLproject_JPMorgan/models/LHNN_3DRosenbrock_sin.weights.h5", threshold_nn=10, num_lf=20, threshold_lf=1000) # with online error monitoring.
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/lhnn_nuts_3Drb.pkl"

    # sampler = sampling(config_name="HNN_3DRosenbrock", sampler_type="NUTS", initial_state=np.array([0,0,0]), num_samples=35000, burnin=1000, chains=1, epsilon=0.025, 
    #          model_path="/home/ycui/Documents/TSRLproject_JPMorgan/models/HNN_3DRosenbrock_sin.weights.h5", threshold_nn=10, num_lf=20, threshold_lf=1000) # with online error monitoring.
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/hnn_nuts_3Drb.pkl"

    # sampler = sampling(config_name="LHNN_3DRosenbrock_T250", sampler_type="NUTS", initial_state=np.array([0,0,0]), num_samples=125000, burnin=5000, chains=1, epsilon=0.025, 
    #          model_path="/home/ycui/Documents/TSRLproject_JPMorgan/models/LHNN_3DRosenbrock_T250_sin.weights.h5", threshold_nn=10, num_lf=20, threshold_lf=1000) # with online error monitoring.
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/lhnn_nuts_3Drb_T250.pkl"

    # sampler = sampling(config_name="LHNN_3DRosenbrock", sampler_type="NUTS", initial_state=np.array([0,0,0]), num_samples=125000, burnin=5000, chains=1, epsilon=0.025, 
    #          model_path=None, threshold_lf=1000) # the traditioinal nuts.
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/grad_nuts_3Drb.pkl"

    # sampler = sampling(config_name="LHNN_2DNealsfunnel", sampler_type="NUTS", initial_state=np.array([0,0]), num_samples=25000, burnin=5000, chains=1, epsilon=0.025, 
    #          model_path="/home/ycui/Documents/TSRLproject_JPMorgan/models/LHNN_2DNealsfunnel_sin.weights.h5", threshold_nn=10, num_lf=20, threshold_lf=1000) # with online error monitoring.
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/lhnn_nuts_2Dnf.pkl"

    # sampler = sampling(config_name="LHNN_2DNealsfunnel", sampler_type="NUTS", initial_state=np.array([0,0]), num_samples=25000, burnin=5000, chains=1, epsilon=0.025, 
    #          model_path=None, threshold_nn=10, num_lf=20, threshold_lf=1000) # traditional nuts.
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/grad_nuts_2Dnf.pkl"

    # sampler = sampling(config_name="LHNN_10DRosenbrock", sampler_type="NUTS", initial_state=np.array([0] * 10), num_samples=125000, burnin=5000, chains=1, epsilon=0.025, 
    #          model_path="/home/ycui/Documents/TSRLproject_JPMorgan/models/LHNN_10DRosenbrock_sin.weights.h5", threshold_nn=10, num_lf=20, threshold_lf=1000) # with online error monitoring.
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/lhnn_nuts_10Drb.pkl"

    # sampler = sampling(config_name="LHNN_10DRosenbrock", sampler_type="NUTS", initial_state=np.array([0] * 10), num_samples=125000, burnin=5000, chains=1, epsilon=0.025, 
    #          model_path=None, threshold_nn=10, num_lf=20, threshold_lf=1000) # traditional nuts.
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/grad_nuts_10Drb.pkl"

    # sampler = sampling(config_name="LHNN_5DillconditionedGaussian", sampler_type="NUTS", initial_state=np.array([0]*5), num_samples=25000, burnin=5000, chains=1, epsilon=0.025, 
    #          model_path="/home/ycui/Documents/TSRLproject_JPMorgan/models/LHNN_5DillconditionedGaussian_sin.weights.h5", threshold_nn=10, num_lf=20, threshold_lf=1000) # with online error monitoring.
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/lhnn_nuts_5DillG.pkl"

    # sampler = sampling(config_name="LHNN_5DillconditionedGaussian", sampler_type="NUTS", initial_state=np.array([0]*5), num_samples=25000, burnin=5000, chains=1, epsilon=0.025, 
    #          model_path=None, threshold_nn=10, num_lf=20, threshold_lf=1000) # traditional nuts.
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/grad_nuts_5DillG.pkl"

    # sampler = sampling(config_name="LHNN_AllenCahn", sampler_type="NUTS", initial_state=np.array([0]*25), num_samples=5000, burnin=1000, chains=1, epsilon=0.025, 
    #          model_path="/home/ycui/Documents/TSRLproject_JPMorgan/models/LHNN_AllenCahn_sin.weights.h5", threshold_nn=10, num_lf=20, threshold_lf=1000) 
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/lhnn_nuts_ac.pkl"

    # sampler = sampling(config_name="LHNN_AllenCahn", sampler_type="NUTS", initial_state=np.array([0]*25), num_samples=5000, burnin=1000, chains=1, epsilon=0.025, 
    #          model_path=None, threshold_nn=10, num_lf=20, threshold_lf=1000) # traditional nuts.
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/grad_nuts_ac.pkl"

    # sampler = sampling(config_name="LHNN_ellipticpde", sampler_type="NUTS", initial_state=np.array([0]*50), num_samples=5000, burnin=1000, chains=1, epsilon=0.025, 
    #          model_path="/home/ycui/Documents/TSRLproject_JPMorgan/models/LHNN_ellipticpde_sin.weights.h5", threshold_nn=10, num_lf=20, threshold_lf=1000) 
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/lhnn_nuts_ellipticpde.pkl"

    # sampler = sampling(config_name="LHNN_ellipticpde", sampler_type="NUTS", initial_state=np.array([0]*50), num_samples=5000, burnin=1000, chains=1, epsilon=0.025, 
    #          model_path=None, threshold_nn=10, num_lf=20, threshold_lf=1000) # traditional nuts.
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/grad_nuts_ellipticpde.pkl"

    # sampler = sampling(config_name="LHNN_pmglmm", sampler_type="PMHMC", initial_state=np.concat([utils.get_marginal_initial(),utils.get_latent_u(500*128)]), dim_marginal=13, num_samples=5000, burnin=1000, chains=1, epsilon=0.005, L=5,
    #          model_path=None)
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/grad_hmc_pmglmm_priorwide_all_5000.pkl"

    # sampler = sampling(config_name="LHNN_pmglmm", sampler_type="PMHMC", initial_state=np.concat([utils.get_marginal_initial(),utils.get_latent_u(500*128)]), dim_marginal=13, num_samples=100, burnin=1000, chains=1, epsilon=0.005, L=5,
    #          model_path="/home/ycui/Documents/TSRLproject_JPMorgan/models/LHNN_pmglmmB_sin.weights.h5")
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/lhnn_hmc_pmglmm_100.pkl"

    # # Save the samples
    # data_samples = {
    #     name: getattr(sampler, name, None) for name in ["samples","numgrad", "monitor", "accept"]
    # }
    # with open(sample_path, "wb") as f:
    #     pickle.dump(data_samples, f)