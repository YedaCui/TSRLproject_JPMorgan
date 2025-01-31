import pytest
from src.gene_data import *
import tensorflow as tf
import numpy as np
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_visible_devices(gpus[0], "GPU")

def test_get_dataset():
    config = {
        "seed" : 0,
        "load" : False,
        "path" : "onlyfortest.obj",
        "num_samples" : 2,
        "per_train" : 0.95,
        "input_dim" : 2,
        "dist_name" : '1D_Gauss_mix',
        "dt" : 0.05,
        "num_lf" : 4
    }
    get_dataset(**config)
    # Delete the data file after testing
    if os.path.exists("onlyfortest.obj"):
        os.remove("onlyfortest.obj")

def test_get_dataset_pm():
    config = {
            "seed" : 0,
            "pseudo-marginal" : None,
            "path" : "onlyfortest",
            "num_samples" : 5,
            "per_train" : 0.9,
            "dim_u" : 500*128,
            "dist_name" : 'pmglmm',
            "dist_name_B" : 'pmglmmB',
            "dt" : 0.005,
            "num_int" : 2
        }
    get_dataset_pm(**config)
    # Delete the data file after testing
    if os.path.exists("onlyfortest_train.tfrecord"):
        os.remove("onlyfortest_train.tfrecord")
    if os.path.exists("onlyfortest_test.tfrecord"):
        os.remove("onlyfortest_test.tfrecord")

