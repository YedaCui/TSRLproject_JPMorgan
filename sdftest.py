import numpy as np
from src import utils

state = np.load("states.npy")

from src.functions import functions

H_B = functions(dist_name="pmglmmB")
H_B(state)
dim = len(state)//2
theta, u, rho, p = state[0:13], state[13:dim], state[dim:dim+13], state[dim+13:]
pmgrad_fn = utils.getpmgrad_fn(H_B)
pmgrad_fn(theta, u, rho, p)