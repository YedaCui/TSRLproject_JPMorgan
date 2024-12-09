
import utils
import functions
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import hnn
from configs import CONFIGS
import mcmc

config = CONFIGS["LHNN_1DGaussianmixture"]

model = hnn.HNN(config["input_dim"], config["num_hidden"], config["num_layers"], config["output_dim"], acti=config["acti"], baseline=config["baseline"], field_type=config["field_type"])
model.load_weights("/home/ycui/Documents/TSRLproject_JPMorgan/models/LHNN_1DGaussianmixture.weights.h5")

nuts_lhnn = mcmc.NUTS(initial_state=np.array([0]), num_samples=500, burnin=1000, chains=1, epsilon=0.05, hamiltonain_model=model, H_function=functions.functions("1D_Gauss_mix"),seed=config["seed"])
nuts_lhnn.sample()
nuts_grad = mcmc.NUTS(initial_state=np.array([0]), num_samples=500, burnin=1000, chains=1, epsilon=0.05, hamiltonain_model=None, H_function=functions.functions("1D_Gauss_mix"),seed=config["seed"])
nuts_grad.sample()