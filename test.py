import numpy as np
import tensorflow as tf
from src import hnn
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_visible_devices(gpus[2], "GPU")

def training_test(config):
    if "type_NN" not in config.keys() or config["type_NN"] == "HNN":
        model = hnn.HNN(config["input_dim"], config["num_hidden"], config["num_layers"], config["output_dim"], acti=config["acti"], baseline=config["baseline"], field_type=config["field_type"])
    elif config["type_NN"] == "PMHNN":
        model = hnn.PMHNN(config["inputdim_NN"], config["num_hidden"], config["num_layers"], acti=config["acti"], dim_marginal=0, coef=np.array([[0]]*(config["input_dim"]//2)), obs=np.array([[0]]*(config["input_dim"]//2)))
    else:
        raise ValueError(f"Invalid 'type_NN' value: {config.get('type_NN', None)}")
    
    data = tf.random.normal(shape=(config["num_samples"], config["input_dim"]), dtype=tf.float32)
    grad = tf.concat([data[:, len(data[0]//2):], -data[:,:len(data[0]//2)]], axis=-1)

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
            train_step(data, grad)
        
        train_timegrads_hat = model.get_gradient(data)
        train_dist = (grad - train_timegrads_hat)**2
        if tf.reduce_mean(train_dist).numpy() < bestloss:
            bestloss = tf.reduce_mean(train_dist).numpy()
        print('Train loss {:.4e} +/- {:.4e}'
            .format(tf.reduce_mean(train_dist).numpy(), tf.math.reduce_std(train_dist).numpy()/np.sqrt(train_dist.shape[0])))
    return bestloss
    
# config = {
#         "pseudo-marginal" : True,
#         "type_NN" : "HNN",
#         "input_dim": 10000,
#         "num_hidden": 100,
#         "num_layers": 3,
#         "output_dim": 10000,
#         "acti" : "sin",
#         "baseline" : True,
#         "field_type" : "solenoidal",
#         "separate_fields" : False,
#         "train_epoch" : 100,
#         "train_step" : 100,
#         "num_samples" : 16,
#     }

config = {
        "pseudo-marginal" : True,
        "per_train" : 0.9,
        "dim_u" : 100,
        "type_NN" : "PMHNN",
        "input_dim": 100,
        "num_int" : 100,
        "inputdim_NN" : 3,
        "num_hidden": 100,
        "num_layers": 3,
        "acti" : "tanh",
        "train_epoch" : 100,
        "train_step" : 100,
        "num_samples" : 16,
        }

training_test(config)