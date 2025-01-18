import numpy as np
import tensorflow as tf
import gene_data
from configs import CONFIGS
import hnn
import random

def train(config):
    """
    The training process.
    """

    np.random.seed(config["seed"])
    tf.random.set_seed(config["seed"])
    random.seed(config["seed"])
    
    if "pseudo-marginal" in config:
        data = gene_data.get_dataset_pm(**config)
    else:
        data = gene_data.get_dataset(**config)
    # arrange data
    train_states = tf.convert_to_tensor( data['train_states'], dtype=tf.float32)
    test_states = tf.convert_to_tensor(data['test_states'], dtype=tf.float32)
    train_timegrads = tf.convert_to_tensor(data['train_timegrads'], dtype=tf.float32)
    test_timegrads = tf.convert_to_tensor(data['test_timegrads'], dtype=tf.float32)

    model = hnn.HNN(config["input_dim"], config["num_hidden"], config["num_layers"], config["output_dim"], acti=config["acti"], baseline=config["baseline"], field_type=config["field_type"])
    
    loss_obj = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
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

if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(gpus[1], "GPU")

    # config = CONFIGS["LHNN_1DGaussianmixture"] # load the config which records all experiment parameters.
    config = CONFIGS["LHNN_3DRosenbrock"]
    # config = CONFIGS["LHNN_3DRosenbrock_T100"]
    # config = CONFIGS["LHNN_10DRosenbrock"]
    # config = CONFIGS["LHNN_2DNealsfunnel"]
    
    # config = CONFIGS["LHNN_ellipticpde"]
    # data = gene_data.get_dataset(**config)
    # print("Finished generating the dataset.")
    model = train(config)
    model.save_weights(config["path_model"])