# from src.functions import functions
# from src import utils
# H = functions(dist_name="pmglmm")
# H_B = functions(dist_name="pmglmmB")
# pmgrad_fn = utils.getpmgrad_fn(H_B)
# import numpy as np

# rho = np.random.normal(size=13)
# p = np.random.normal(size=len(utils.get_latent_u(500*128)))
# states = utils.pmintegrator(utils.get_marginal_initial(), utils.get_latent_u(500*128), rho, p, pmgrad_fn, 0.01, 5000)

path = "/home/ycui/Documents/TSRLproject_JPMorgan/data/pmglmmB_wide"
import os
from src import utils


train_file_paths = [os.path.join(path,f) for f in os.listdir(path) if f.startswith("train")]
test_file_paths = [os.path.join(path,f) for f in os.listdir(path) if f.startswith("test")]

utils.write_pickle_to_tfrecord(train_file_paths, path+"_train.tfrecord")
# utils.write_pickle_to_tfrecord(test_file_paths, path+"_test.tfrecord")