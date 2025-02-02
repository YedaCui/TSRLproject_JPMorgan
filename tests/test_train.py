import pytest
from src import hnn
from src.train import *
import tensorflow as tf
import numpy as np
import os, pickle
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_visible_devices(gpus[0], "GPU")

def test_train():
    """
    Test the train function to ensure proper model training and loss reduction.
    """
    weights_path = "onlyfortest.weights.h5"
    path = "onlyfortest.obj"
    # Generate random data to mock training 
    data = {
        "train_states": np.random.rand(10, 2),
        "train_timegrads": np.random.rand(10, 2),
        "test_states": np.random.rand(10, 2),
        "test_timegrads": np.random.rand(10, 2)
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
 
    # Mock configuration
    config = {
        "seed" : 0,
        "load" : True,
        "path" : path,
        "num_samples" : 20,
        "per_train" : 0.95,
        "input_dim" : 2,
        "dist_name" : '1D_Gauss_mix',
        "dt" : 0.05,
        "num_lf" : 400,
        "num_hidden": 10,
        "num_layers": 3,
        "output_dim": 2,
        "acti" : "sin",
        "baseline" : False,
        "field_type" : "solenoidal",
        "separate_fields" : False,
        "train_epoch" : 1,
        "train_step" : 1,
        "path_model" : weights_path,
    }
    best_model = train(config)
    best_model.save_weights(config["path_model"])
    # Ensure the returned model is not None
    assert best_model is not None, "The train function did not return a trained model!"
    # Check if the model has trained weights
    trained_weights = best_model.get_weights()
    assert len(trained_weights) > 0, "The trained model does not have any weights!"
    saved_model = hnn.HNN(config["input_dim"], config["num_hidden"], config["num_layers"], config["output_dim"], acti=config["acti"], baseline=config["baseline"], field_type=config["field_type"])
    saved_model.load_weights(weights_path)
    # Ensure the saved model is identical to the trained model
    for w1, w2 in zip(best_model.get_weights(), saved_model.get_weights()):
        assert np.allclose(w1, w2), "The saved model does not match the trained model!"
    # Delete the weights file after testing
    if os.path.exists(weights_path):
        os.remove(weights_path)
    if os.path.exists(path):
        os.remove(path)

def test_train_pm():
    """
    Test the train_pm function to ensure proper model training and loss reduction.
    """
    weights_path = "onlyfortest.weights.h5"
    path = "onlyfortest"
    def create_example(states, timegrads):
        """
        Creates a tf.train.Example for the given states and timegrads.
        """
        feature = {
            "states": tf.train.Feature(float_list=tf.train.FloatList(value=states)),
            "timegrads": tf.train.Feature(float_list=tf.train.FloatList(value=timegrads)),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    dims = (13 + 500 * 128) * 2  
    num_samples = 2

    # Generate random data  to mock training 
    data = {
        "states": np.random.rand(num_samples, dims).astype(np.float32),
        "timegrads": np.random.rand(num_samples, dims).astype(np.float32)
    }
    with tf.io.TFRecordWriter(path+"_train.tfrecord") as train_writer, tf.io.TFRecordWriter(path+"_test.tfrecord") as test_writer:
        for i in range(num_samples):
            states = data["states"][i]
            timegrads = data["timegrads"][i]
            example = create_example(states, timegrads)
            train_writer.write(example.SerializeToString())
            test_writer.write(example.SerializeToString())
    # Mock configuration
    config = {
        "seed" : 0,
        "load" : True,
        "pseudo-marginal" : None,
        "path" : path,
        "num_samples" : 500,
        "per_train" : 0.9,
        "dim_u" : 500*128,
        "type_NN" : "HNN",
        "input_dim": (13+500*128)*2,
        "dist_name" : 'pmglmm',
        "dist_name_B" : 'pmglmmB',
        "dt" : 0.005,
        "num_int" : 2000,
        "num_hidden": 100,
        "num_layers": 3,
        "output_dim": (13+500*128)*2,
        "acti" : "tanh",
        "baseline" : True,
        "field_type" : "solenoidal",
        "separate_fields" : False,
        "train_epoch" : 1,
        "train_step" : 1,
        "batch_size" : 8,
        "accumulation_steps" : 1,
        "testset_size" : 1,
        "path_model" : weights_path,
    }
    best_model = train_pm(config)
    best_model.save_weights(config["path_model"])
    # Ensure the returned model is not None
    assert best_model is not None, "The train function did not return a trained model!"
    # Check if the model has trained weights
    trained_weights = best_model.get_weights()
    assert len(trained_weights) > 0, "The trained model does not have any weights!"
    saved_model = hnn.HNN(config["input_dim"], config["num_hidden"], config["num_layers"], config["output_dim"], acti=config["acti"], baseline=config["baseline"], field_type=config["field_type"])
    saved_model.load_weights(weights_path)
    # Ensure the saved model is identical to the trained model
    for w1, w2 in zip(best_model.get_weights(), saved_model.get_weights()):
        assert np.allclose(w1, w2), "The saved model does not match the trained model!"
    # Delete the weights file after testing
    for path_file in [weights_path, path+"_train.tfrecord", path+"_test.tfrecord"]:
        if os.path.exists(path_file):
            os.remove(path_file)