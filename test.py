from src.functions import functions
from src import utils
H = functions(dist_name="pmglmm")
H_B = functions(dist_name="pmglmmB")
pmgrad_fn = utils.getpmgrad_fn(H_B)
import numpy as np

rho = np.random.normal(size=13)
p = np.random.normal(size=len(utils.get_latent_u(500*128)))
states = utils.pmintegrator(utils.get_marginal_initial(), utils.get_latent_u(500*128), rho, p, pmgrad_fn, 0.01, 5000)