import numpy as np
import utils
import pickle
from functions import functions
import scipy
import os

def get_dataset(seed=0, load=True, path=None, num_samples=10000, per_train=0.8, input_dim=2, dist_name='1D_Gauss_mix', dt=0.25, num_lf=4, **kwargs):
    """
    Get the dataset for traing and testing.

    Args:
        seed : random seed.
        load : 'bool' If True, load data from path.
        path : 'str' or None. If not None, load data from the path.
        num_samples : 'int' Number of total samples.
        per_train : 'float' Percentage of train dataset.
        input_dim : 'int' Integer number of input dimension.
        dist_name : 'str' The distribution name.
        dt : 'float' The integration time step.
        num_lf : 'int' Number of leapfrog step.
    """

    if load:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print("Successfully loaded data")
        return data
    
    data = {'meta': locals()} # record all the augments.
    np.random.seed(seed)
    states, timegrads = [],[]
    initial_state = np.zeros(input_dim)

    timegrad_fn = utils.get_timegrad_fn(functions(dist_name)) # Get the timegrad function based on the distribution name.
    for idx_sample in range(num_samples):
        print(f"Begin to generate the {idx_sample}th sample.")
        for idx_dim in range(input_dim//2, input_dim):
            initial_state[idx_dim] = scipy.stats.norm(0,1).rvs() # initialize the momentum.
        state, timegrad, _ = utils.get_trajectory(timegrad_fn, initial_state, dt, num_lf)
        states.append(state)
        timegrads.append(timegrad)
        initial_state[0:input_dim//2] = state[-1,0:input_dim//2]
    data['states'] = np.concat(states)
    data['timegrads'] = np.concat(timegrads)

    # make a train/test split
    split_idx = int(len(data['states']) * per_train)
    split_data = {}
    for k in ['states', 'timegrads']:
        split_data['train_' + k], split_data['test_' + k] = data[k][:split_idx], data[k][split_idx:]
    data = split_data

    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return data