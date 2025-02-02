import pytest
from src.sampling import *
import tensorflow as tf
import numpy as np
import os
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_visible_devices(gpus[0], "GPU")

def test_sampling():
    sampler = sampling(config_name="LHNN_ellipticpde", sampler_type="NUTS", initial_state=np.array([0]*50), num_samples=5, burnin=1, chains=1, epsilon=0.025, 
             model_path=None, threshold_nn=10, num_lf=20, threshold_lf=1000) 
    sample_path = "./onlyfortest.pkl"
    # Save the samples
    data_samples = {
        name: getattr(sampler, name, None) for name in ["samples","numgrad", "monitor", "accept"]
    }
    with open(sample_path, "wb") as f:
        pickle.dump(data_samples, f)
    saved_sampler = sampling(config_name="LHNN_ellipticpde", sampler_type="NUTS", initial_state=np.array([0]*50), num_samples=5, burnin=1, chains=1, epsilon=0.025, 
            sample_path = sample_path, threshold_nn=10, num_lf=20, threshold_lf=1000)
    for attr in ["samples","numgrad", "monitor", "accept"]:
        sampler_attr = np.nan_to_num(getattr(sampler, attr, None), nan=0)
        saved_sampler_attr = np.nan_to_num(getattr(saved_sampler, attr, None), nan=0)
        assert np.allclose(sampler_attr, saved_sampler_attr), f"Mismatch in attribute '{attr}': Arrays are not equal."
    # Clean up the saved file
    if os.path.exists(sample_path):
        os.remove(sample_path)

