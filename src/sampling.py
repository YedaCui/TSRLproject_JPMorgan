import functions
import hnn
from configs import CONFIGS
import mcmc
import numpy as np
import tensorflow as tf
import pickle


def sampling(config_name, sampler_type, initial_state, num_samples, burnin, chains=1, epsilon=0.05, model_path=None, seed=0, sample_path=None, **kwargs):
    """
    Sampling by MCMC.
    """
    config = CONFIGS[config_name]
    if model_path:
        model = hnn.HNN(config["input_dim"], config["num_hidden"], config["num_layers"], config["output_dim"], acti=config["acti"], baseline=config["baseline"], field_type=config["field_type"])
        model.load_weights(model_path)
    else:
        model = None

    if sampler_type == "HMC":
        sampler = mcmc.HMC(initial_state=initial_state, num_samples=num_samples, burnin=burnin, chains=chains, epsilon=epsilon, hamiltonain_model=model, H_function=functions.functions(config["dist_name"]),seed=seed, **kwargs)
    elif sampler_type == "NUTS":
        sampler = mcmc.NUTS(initial_state=initial_state, num_samples=num_samples, burnin=burnin, chains=chains, epsilon=epsilon, hamiltonain_model=model, H_function=functions.functions(config["dist_name"]),seed=seed, **kwargs)
    if sample_path is None:
        sampler.sample()
    else:
        with open(sample_path, "rb") as f:
            data_samples = pickle.load(f)
        sampler.samples = data_samples["samples"]
        sampler.numgrad = data_samples["numgrad"]
        sampler.monitor = data_samples["monitor"]
        # for name in ["samples","numgrad", "monitor"]:
        #     if hasattr(sampler, name):
        #         setattr(sampler, name, data_samples[name])
    return sampler

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(gpus[1], "GPU")

    # sampler = sampling(config_name="LHNN_1DGaussianmixture", sampler_type="HMC", initial_state=np.array([0]), num_samples=5000, burnin=1000, chains=1, epsilon=0.05, 
    #          model_path="/home/ycui/Documents/TSRLproject_JPMorgan/models/LHNN_1DGaussianmixture_sin.weights.h5", L=5)
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/lhnn_hmc_LHNN_1DGaussianmixture.pkl"

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
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/lhnn_nuts_3Drb_notoem_1000.pkl"

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
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/lhnn_nuts_3Dnf.pkl"

    # sampler = sampling(config_name="LHNN_2DNealsfunnel", sampler_type="NUTS", initial_state=np.array([0,0]), num_samples=25000, burnin=5000, chains=1, epsilon=0.025, 
    #          model_path=None, threshold_nn=10, num_lf=20, threshold_lf=1000) # traditional nuts.
    # sample_path = "/home/ycui/Documents/TSRLproject_JPMorgan/samples/grad_nuts_3Dnf.pkl"

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

    # Save the samples
    data_samples = {
        name: getattr(sampler, name, None) for name in ["samples","numgrad", "monitor"]
    }
    with open(sample_path, "wb") as f:
        pickle.dump(data_samples, f)