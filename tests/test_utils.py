from src import utils
import numpy as np

def test_functions():
    dist_name = '1D_Gauss_mix'
    H_function = utils.functions(dist_name)
    state = np.array([0.5, 0.5])
    assert H_function(state) == 0.5