import numpy as np
import tensorflow as tf
from . import gene_data, hnn
from .configs import CONFIGS
import random

def train(config):
    """
    This function trains a neural network HNN to approximate the Hamiltonian gradients.

    Args:
        config : "dict" Configuration dictionary containing training parameters.
    
    Returns:
        bestmodel: The model with the lowest test loss during training.
    """

    np.random.seed(config["seed"])
    tf.random.set_seed(config["seed"])
    random.seed(config["seed"])
    
    data = gene_data.get_dataset(**config)
    # arrange data
    train_states = tf.convert_to_tensor( data['train_states'], dtype=tf.float32)
    test_states = tf.convert_to_tensor(data['test_states'], dtype=tf.float32)
    train_timegrads = tf.convert_to_tensor(data['train_timegrads'], dtype=tf.float32)
    test_timegrads = tf.convert_to_tensor(data['test_timegrads'], dtype=tf.float32)

    model = hnn.HNN(config["input_dim"], config["num_hidden"], config["num_layers"], config["output_dim"], acti=config["acti"], baseline=config["baseline"], field_type=config["field_type"])
    
    loss_obj = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()
    
    @tf.function
    def train_step(x,y):
        with tf.GradientTape() as tape:
            y_pred = model.get_gradient(x)
            loss = loss_obj(y,y_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    bestloss, bestmodel = np.inf, None
    for idx_epoch in range(config["train_epoch"]):
        print(f"Begin the {idx_epoch}th training epoch:")
        for idx_step in range(config["train_step"]):
            idxs_shuffle = tf.random.shuffle(tf.range(train_states.shape[0]))
            shuffled_states, shuffled_timegrads = tf.gather(train_states, idxs_shuffle), tf.gather(train_timegrads, idxs_shuffle)
            train_step(shuffled_states, shuffled_timegrads)
        
        train_timegrads_hat = model.get_gradient(train_states)
        train_dist = (train_timegrads - train_timegrads_hat)**2 
        test_timegrads_hat = model.get_gradient(test_states)
        test_dist = (test_timegrads - test_timegrads_hat)**2
        print('Train loss {:.4e} +/- {:.4e}\n test loss {:.4e} +/- {:.4e}'
            .format(tf.reduce_mean(train_dist).numpy(), tf.math.reduce_std(train_dist).numpy()/np.sqrt(train_dist.shape[0]),
                    tf.reduce_mean(test_dist).numpy(),  tf.math.reduce_std(test_dist).numpy()/np.sqrt(test_dist.shape[0])))
        if tf.reduce_mean(test_dist).numpy() < bestloss:
            bestloss = tf.reduce_mean(test_dist).numpy()
            bestmodel = tf.keras.models.clone_model(model)
            bestmodel.set_weights(model.get_weights())
    return bestmodel

def train_pm(config):
    """
    The training process for Pseudo-Marginal Hamiltonian Monte Carlo (PMHMC).
    
    This function trains a neural network (HNN or PMHNN) to approximate the Hamiltonian
    gradients. Due to the high-dimensional nature of the input data, the entire dataset
    cannot fit into memory or be processed in a single step. Instead, TFRecord files are used
    for efficient data streaming and batching.

    Args:
        config : "dict" Configuration dictionary containing training parameters.
    
    Returns:
        bestmodel: The model with the lowest test loss during training.
    """

    np.random.seed(config["seed"])
    tf.random.set_seed(config["seed"])
    random.seed(config["seed"])

    def parse_tfrecord(proto):
        feature_description = {
            "states": tf.io.FixedLenFeature(config["input_dim"], tf.float32),
            "timegrads": tf.io.FixedLenFeature(config["input_dim"], tf.float32),
        }
        return tf.io.parse_single_example(proto, feature_description)

    def load_tfrecord(filename, batch_size, repeat=True):
        """
        Loads and prepares a TFRecord dataset.

        Args:
            filename (str): Path to the TFRecord file.
            batch_size (int): Number of examples per batch.
            repeat (bool): Whether to loop over the dataset indefinitely (for training).
        
        Returns:
            tf.data.Dataset: Prepared dataset.
        """
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(parse_tfrecord)
        if repeat:
            dataset = dataset.repeat()  
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    train_dataset = load_tfrecord(config["path"]+"_train.tfrecord", batch_size=config["batch_size"], repeat=True)
    test_dataset = load_tfrecord(config["path"]+"_test.tfrecord", batch_size=config["batch_size"], repeat=False)

    if "type_NN" not in config.keys() or config["type_NN"] == "HNN":
        model = hnn.HNN(config["input_dim"], config["num_hidden"], config["num_layers"], config["output_dim"], acti=config["acti"], baseline=config["baseline"], field_type=config["field_type"])
    elif config["type_NN"] == "PMHNN":
        model = hnn.PMHNN(config["inputdim_NN"], config["num_hidden"], config["num_layers"], acti=config["acti"])
    else:
        raise ValueError(f"Invalid 'type_NN' value: {config.get('type_NN', None)}")

    # loss_obj = tf.keras.losses.MeanSquaredError()
    def loss_obj(y, y_pred):
        loss = (y-y_pred)**2
        loss = loss / (tf.math.reduce_std(loss, axis=0) + 1e-8)
        return tf.reduce_mean(loss)
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(x, y, accumulated_gradients=None):
        with tf.GradientTape() as tape:
            y_pred = model.get_gradient(x)
            loss = loss_obj(y, y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        if accumulated_gradients is not None:
            # Accumulate gradients for gradient accumulation
            gradients = [g + ag for g, ag in zip(gradients, accumulated_gradients)]
        return gradients, loss
    
    bestloss, bestmodel = np.inf, None
    for idx_epoch in range(config["train_epoch"]):
        print(f"Begin the {idx_epoch}th training epoch:")
        train_dist, test_dist = [], []
        accumulated_gradients = None
        for step, batch in enumerate(train_dataset.take(config["train_step"])):
            states, timegrads = batch["states"], batch["timegrads"]
            accumulated_gradients, loss = train_step(states, timegrads, accumulated_gradients)
            train_dist.append(loss)
            # Apply accumulated gradients after a specified number of steps
            if (step + 1) % config["accumulation_steps"] == 0:
                accumulated_gradients = [g / config["accumulation_steps"] for g in accumulated_gradients]   
                optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
                accumulated_gradients = None # Reset accumulated gradients
        # Evaluate on the test dataset
        for batch in test_dataset.take(config["testset_size"]):
            states, timegrads = batch["states"], batch["timegrads"]
            test_timegrads_hat = model.get_gradient(states)
            test_dist.append(tf.reduce_mean((timegrads-test_timegrads_hat)**2))
            del test_timegrads_hat

        print('Train loss {:.4e} +/- {:.4e}\n test loss {:.4e} +/- {:.4e}'
            .format(np.mean(np.array(train_dist)), np.std(np.array(train_dist))/len(train_dist)**0.5,
                    np.mean(np.array(test_dist)), np.std(np.array(test_dist))/len(train_dist)**0.5))
        if np.mean(np.array(test_dist)) < bestloss:
            bestloss = np.mean(np.array(test_dist))
            bestmodel = tf.keras.models.clone_model(model)
            bestmodel.set_weights(model.get_weights())
    return bestmodel

if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(gpus[2], "GPU")

    # config = CONFIGS["LHNN_1DGaussianmixture"]
    # config = CONFIGS["LHNN_3DRosenbrock"]
    # config = CONFIGS["LHNN_3DRosenbrock_T100"]
    # config = CONFIGS["LHNN_10DRosenbrock"]
    # config = CONFIGS["LHNN_2DNealsfunnel"]
    config = CONFIGS["PMHNN_pmglmm"]
    
    # config = CONFIGS["LHNN_ellipticpde"]
    # gene_data.get_dataset(**config)
    # print("Finished generating the dataset.")
    if "pseudo-marginal" in config.keys():
        model = train_pm(config)
    else:
        model = train(config)
    model.save_weights(config["path_model"])