import tensorflow as tf
import utils

class HNN(tf.keras.Model):
    """
    The class HNN represents the (latent) Hamiltonian Neural Network by Greydanus et al 2019 and Dhulipala et al 2022.
    """
    def __init__(self, input_dim, num_hidden, num_layers, output_dim, acti="relu", baseline=False, field_type='solenoidal',**kwargs):
        """
        Args:
            input_dim : 'int' Integer of the input dimesion.
            num_hidden : 'int' the number of neurons.
            num_layers : 'int' the number of hidden layers.
            output_dim : 'int' the output dimension.
            nn : tf.keras.Model subclass whose call function takes an argument "state" with shape (num_sample, num_dimenson) 
            and returns tensor with shape (num_sample, 2) (i.e. HNN by Greydanus et al 2019) 
            or (num_sample, input_dim) (i.e. Latent HNN by Dhulipala et al 2022).
            baseline : 'bool' If True, the class inplements the vanilla DNN whose outputs are used as gradient directly.
            field_type : 'str' It could be either 'solenoidal' or 'conservative'.
        """
        super(HNN, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.hidden_layers = [tf.keras.layers.Dense(num_hidden, activation=None) for idx_layer in range(num_layers)]
        self.lastlayer = tf.keras.layers.Dense(output_dim, activation=None, use_bias=False)
        self.acti = utils.choose_acti(acti)
        self.baseline = baseline
        self.field_type = field_type

        # Apply orthogonal initialization
        for idx_layer, layer in enumerate(self.hidden_layers + [self.lastlayer]):
            # Build the layer to initialize weights
            if idx_layer == 0:
                layer.build((None, input_dim))
            else:
                layer.build((None, num_hidden))
            w_shape = layer.kernel.shape
            orthogonal_init = tf.keras.initializers.Orthogonal()
            layer.kernel.assign(orthogonal_init(w_shape))
            # Initialize bias to zeros (if the layer uses bias)
            if layer.bias is not None:
                layer.bias.assign(tf.zeros_like(layer.bias))

    def call(self, state):
        """
        Args:
            state : 'tensor' Combination of position and momentum.
        """
        for layer in self.hidden_layers:
            state = self.acti(layer(state))
        y = self.lastlayer(state)
        if self.baseline:
            return y # In this case, the output must have shape (\cdot, self.input_dim) since it represents the gradient w.r.t. state.
        
        dim = y.shape[1]
        F1, F2 = y[:,:dim//2], y[:,dim//2:]
        return F1, F2

    def get_gradient(self, state, separate_fields=False):
        """
        Calculate the gradient w.r.t. the state by the neural network.

        Args:
            state : 'tensor' Combination of position and momentum.
            separate_fields : 'bool' If True, return the gradients of different fields separetely
        """
        if self.baseline:
            return self.call(state)
        
        conservative_field, solenoidal_field = tf.zeros_like(state), tf.zeros_like(state)

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(state)
            F1, F2 = self.call(state)
            F1, F2 = tf.reduce_sum(F1), tf.reduce_sum(F2)
        
        if self.field_type == "conservative":
            dF1 = tape.gradient(F1, state)
            conservative_field = dF1
        if self.field_type == "solenoidal":
            dF2 = tape.gradient(F2, state)
            # solenoidal_field = tf.matmul(dF2, tf.transpose(self.M))
            n = dF2.shape[-1]
            solenoidal_field = tf.concat([dF2[:,n//2:], -dF2[:,:n//2]], axis=-1)
        del tape

        if separate_fields:
            return [conservative_field, solenoidal_field]
        else:
            return  conservative_field + solenoidal_field

        
    def permutation_tensor(self, n):
        """
        Generate the matrix [[0, I], [-I, 0]].
        """
        M = tf.eye(n, dtype=tf.float32)
        n_half = n // 2
        M_upper = M[:n_half, :]
        M_lower = M[n_half:, :]
        M = tf.concat([M_lower, -M_upper], axis=0)
        return M
    
    def get_config(self):
        config = super(HNN, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "num_hidden" : self.num_hidden,
            "num_layers" : self.num_layers,
            "output_dim" : self.output_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class PMHNN(tf.keras.Model):
    """
    The class PMHNN introduced in the report of the repo.
    """
    def __init__(self, input_dim, num_hidden, num_layers, acti="relu",**kwargs):
        """
        Args:
            input_dim : 'int' Integer of the input dimesion of the PMHNN.
            num_hidden : 'int' the number of neurons.
            num_layers : 'int' the number of hidden layers.
        """
        super(PMHNN, self).__init__(**kwargs)
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.hidden_layers = [tf.keras.layers.Dense(num_hidden, activation=None) for idx_layer in range(num_layers)]
        self.lastlayer = tf.keras.layers.Dense(1, activation=None, use_bias=False)
        self.acti = utils.choose_acti(acti)
        self.Z_coe = tf.convert_to_tensor(utils.getZ(), dtype=tf.float32)
        self.Z = tf.reshape(self.Z_coe,shape=(1, 500, 6*8))
        self.obs=tf.convert_to_tensor(utils.getglmmdata(), dtype=tf.float32)
        self.obs = tf.reshape(self.obs,shape=(1, 500, 6))

        # Apply orthogonal initialization
        for idx_layer, layer in enumerate(self.hidden_layers + [self.lastlayer]):
            # Build the layer to initialize weights
            if idx_layer == 0:
                layer.build((None, input_dim))
            else:
                layer.build((None, num_hidden))
            w_shape = layer.kernel.shape
            orthogonal_init = tf.keras.initializers.Orthogonal()
            layer.kernel.assign(orthogonal_init(w_shape))
            # Initialize bias to zeros (if the layer uses bias)
            if layer.bias is not None:
                layer.bias.assign(tf.zeros_like(layer.bias))

    def call(self, state):
        """
        Args:
            state : 'tensor' Combination of position and momentum.
        """
        n = state.shape[-1]
        theta, u = state[:,:13], state[:,13:n//2]
        u = tf.reshape(u,shape=(len(u), 500, 128))
        theta = tf.reshape(theta,shape=(len(theta), 1, 13))
        data_shape = [max(dims) for dims in zip(self.obs.shape, self.Z.shape, theta.shape, u.shape)]
        
        x = tf.concat([tf.broadcast_to(self.obs, shape=data_shape[:-1]+[self.obs.shape[-1]]),
                       tf.broadcast_to(self.Z, shape=data_shape[:-1]+[self.Z.shape[-1]]),
                       tf.broadcast_to(theta, shape=data_shape[:-1]+[theta.shape[-1]]),
                       tf.broadcast_to(u, shape=data_shape[:-1]+[u.shape[-1]])], axis=-1)
        x = tf.reshape(x, shape=[-1, x.shape[-1]])
        for layer in self.hidden_layers:
            x = self.acti(layer(x))
        y = self.lastlayer(x)
        y = tf.reshape(y, shape=data_shape[:-1])
        y = tf.reduce_sum(y, axis=-1, keepdims=True)
        y = y + tf.reduce_sum(tf.squeeze(theta,1)**2/2, axis=-1, keepdims=True)
        return y

    def get_gradient(self, state):
        """
        Calculate the gradient w.r.t. the state by the neural network.

        Args:
            state : 'tensor' Combination of position and momentum.
        """
        
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(state)
            F = self.call(state)
            dF = tape.gradient(F, state)
            n = dF.shape[-1]
            grads = tf.concat([dF[:,n//2:], -dF[:,:n//2]], axis=-1)
        return grads
    
    def get_config(self):
        config = super(PMHNN, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "num_hidden" : self.num_hidden,
            "num_layers" : self.num_layers,
            "output_dim" : self.output_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)