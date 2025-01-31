import numpy as np
from . import utils
import pickle
from .functions import functions
import tensorflow as tf
import scipy
import os

def get_dataset(seed=0, load=True, path=None, num_samples=10000, per_train=0.8, input_dim=2, dist_name='1D_Gauss_mix', dt=0.25, num_lf=4, **kwargs):
    """
    Get the dataset for traing and testing.

    Args:
        seed : random seed.
        load : 'bool' If True, load data from path.
        path : 'str' The path to save or load the data.
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
    data={}
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

def get_dataset_pm(seed=0, path=None, num_samples=10000, per_train=0.8, initial_marginal=utils.get_marginal_initial(), dim_u=128, dist_name_B='pmglmmB', dt=0.25, num_int=4, **kwargs):
    """
    Get the dataset for traing and testing.

    Args:
        seed : random seed.
        path : 'str' The path to save the data.
        num_samples : 'int' Number of total samples.
        per_train : 'float' Percentage of train dataset.
        initial_marginal : 'np.array' The initial marginal state.
        dim_u : 'int' The dimension of the latent varibles.
        dist_name_B : 'str' The distribution name of Hamiltoian B.
        dt : 'float' The integration time step.
        num_int : 'int' Number of integrator step.
    """
    np.random.seed(seed)
    dim_marginal = len(initial_marginal)
    input_dim = 2*(dim_marginal+dim_u)
    initial_state = np.zeros(input_dim)
    initial_state[:dim_marginal] = initial_marginal
    initial_state[dim_marginal:dim_marginal+dim_u] = utils.get_latent_u(dim_u)

    def serialize_example(states, timegrads):
        """
        Create a serialized example containing states and timegrads.

        Args:
            states (np.array): The states array.
            timegrads (np.array): The time gradients array.

        Returns:
            Serialized TensorFlow Example.
        """
        feature = {
            "states": tf.train.Feature(float_list=tf.train.FloatList(value=states)),
            "timegrads": tf.train.Feature(float_list=tf.train.FloatList(value=timegrads)),
        }
        # Create a Features message using tf.train.Example
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    pmgrad_fn = utils.getpmgrad_fn(functions(dist_name_B)) # Get the timegrad function based on the distribution name.
    with tf.io.TFRecordWriter(path+"_train.tfrecord") as train_writer, tf.io.TFRecordWriter(path+"_test.tfrecord") as test_writer:
        for idx_sample in range(num_samples):
            print(f"Begin to generate the {idx_sample}th sample.")
            initial_state[input_dim//2:] = np.random.normal(size=input_dim//2) # initialize the momentum.
            state, state_for_grad, timegrad  = utils.pmintegrator(initial_state[0:dim_marginal],
                                                                initial_state[dim_marginal:dim_marginal+dim_u],
                                                                initial_state[dim_marginal+dim_u:2*dim_marginal+dim_u],
                                                                initial_state[2*dim_marginal+dim_u:],
                                                                pmgrad_fn, dt, num_int, require_grads=True)
            # Training/testing split index
            split_idx = int(len(timegrad) * per_train)
            for idx in range(split_idx):
                train_serialized = serialize_example(state_for_grad[idx], timegrad[idx])
                train_writer.write(train_serialized)
            for idx in range(split_idx, len(state_for_grad)):
                test_serialized = serialize_example(state_for_grad[idx], timegrad[idx])
                test_writer.write(test_serialized)
            initial_state[0:input_dim//2] = state[-1,0:input_dim//2] # update the initial_state
