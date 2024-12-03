import tensorflow as tf
import utils


# class HNN(tf.keras.Model):
#     """
#     The class HNN represents the (latent) Hamiltonian Neural Network by Greydanus et al 2019 and Dhulipala et al 2022.
#     """
#     def __init__(self, input_dim, nn, baseline=False, field_type='solenoidal',**kwargs):
#         """
#         Args:
#             input_dim : 'int' Integer of the input dimesion.
#             nn : tf.keras.Model subclass whose call function takes an argument "state" with shape (num_sample, num_dimenson) 
#             and returns tensor with shape (num_sample, 2) (i.e. HNN by Greydanus et al 2019) 
#             or (num_sample, input_dim) (i.e. Latent HNN by Dhulipala et al 2022).
#             baseline : 'bool' If True, the class inplements the vanilla DNN whose outputs are used as gradient directly.
#             field_type : 'str' It could be either 'solenoidal' or 'conservative'.
#         """
#         super(HNN, self).__init__(**kwargs)
#         self.input_dim = input_dim
#         self.nn = nn
#         self.baseline = baseline
#         self.field_type = field_type
#         self.M = self.permutation_tensor(input_dim)

#     def call(self, state):
#         """
#         Args:
#             state : 'tensor' Combination of position and momentum.
#         """
#         if self.baseline:
#             return self.nn(state) # In this case, the output must have shape (\cdot, self.input_dim) since it represents the gradient w.r.t. state.
        
#         y = self.nn(state)
#         dim = y.shape[1]
#         F1, F2 = y[:,:dim//2], y[:,dim//2:]
#         return F1, F2

#     def get_gradient(self, state, separate_fields=False):
#         """
#         Calculate the gradient w.r.t. the state by the neural network.

#         Args:
#             state : 'tensor' Combination of position and momentum.
#             separate_fields : 'bool' If True, return the gradients of different fields separetely
#         """
#         if self.baseline:
#             return self.nn(state)
        
#         conservative_field, solenoidal_field = tf.zeros_like(state), tf.zeros_like(state)

#         with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
#             tape.watch(state)
#             F1, F2 = self.call(state)
#             F1, F2 = tf.reduce_sum(F1), tf.reduce_sum(F2)
        
#         if self.field_type == "conservative":
#             dF1 = tape.gradient(F1, state)
#             conservative_field = dF1
#         if self.field_type == "solenoidal":
#             dF2 = tape.gradient(F2, state)
#             solenoidal_field = tf.matmul(dF2, tf.transpose(self.M))
#         del tape

#         if separate_fields:
#             return [conservative_field, solenoidal_field]
#         else:
#             return  conservative_field + solenoidal_field

        
#     def permutation_tensor(self, n):
#         """
#         Generate the matrix [[0, I], [-I, 0]].
#         """
#         M = tf.eye(n, dtype=tf.float32)
#         n_half = n // 2
#         M_upper = M[:n_half, :]
#         M_lower = M[n_half:, :]
#         M = tf.concat([M_lower, -M_upper], axis=0)
#         return M
    
#     def get_config(self):
#         config = super(HNN, self).get_config()
#         config.update({
#             "input_dim": self.input_dim,
#             "nn": self.nn.get_config()
#         })
#         return config

#     @classmethod
#     def from_config(cls, config):
#         nn_config = config.pop("nn")
#         nn = MLP.from_config(nn_config)
#         return cls(nn=nn,**config)



# class MLP(tf.keras.Model):
#     def __init__(self, input_dim, num_hidden, num_layers, output_dim, acti="relu", **kwargs):
#         """
#         Generate the vanilla Neural Network.

#         Args:
#             num_hidden : 'int' the number of neurons.
#             num_layers : 'int' the number of hidden layers.
#             output_dim : 'int' the output dimension.
#         """
#         super(MLP, self).__init__(**kwargs)
#         self.input_dim = input_dim
#         self.num_hidden = num_hidden
#         self.num_layers = num_layers
#         self.output_dim = output_dim
#         self.hidden_layers = [tf.keras.layers.Dense(num_hidden, activation=None) for idx_layer in range(num_layers)]
#         self.lastlayer = tf.keras.layers.Dense(output_dim, activation=None, use_bias=False)
#         self.acti = utils.choose_acti(acti)

#         # Apply orthogonal initialization
#         for idx_layer, layer in enumerate(self.hidden_layers + [self.lastlayer]):
#             # Build the layer to initialize weights
#             if idx_layer == 0:
#                 layer.build((None, input_dim))
#             else:
#                 layer.build((None, num_hidden))
#             w_shape = layer.kernel.shape
#             orthogonal_init = tf.keras.initializers.Orthogonal()
#             layer.kernel.assign(orthogonal_init(w_shape))
#             # Initialize bias to zeros (if the layer uses bias)
#             if layer.bias is not None:
#                 layer.bias.assign(tf.zeros_like(layer.bias))
        
#     def call(self,x):
#         for layer in self.hidden_layers:
#             x = self.acti(layer(x))
#         return self.lastlayer(x)
    
#     def get_config(self):
#         config = super(MLP, self).get_config()
#         config.update({
#             "input_dim" : self.input_dim,
#             "num_hidden" : self.num_hidden,
#             "num_layers" : self.num_layers,
#             "output_dim" : self.output_dim
#         })
#         return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)


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
        self.M = self.permutation_tensor(input_dim)

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
            solenoidal_field = tf.matmul(dF2, tf.transpose(self.M))
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