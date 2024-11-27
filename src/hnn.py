import tensorflow as tf
import utils


class HNN(tf.keras.Model):
    """
    The class HNN represents the (latent) Hamiltonian Neural Network by Greydanus et al 2019 and Dhulipala et al 2022.
    """
    def __init__(self, input_dim, nn, baseline=False, field_type='solenoidal'):
        """
        Args:
            input_dim : 'int' Integer of the input dimesion.
            nn : tf.keras.Model subclass whose call function takes an argument "state" with shape (num_sample, num_dimenson) 
            and returns tensor with shape (num_sample, 2) (i.e. HNN by Greydanus et al 2019) 
            or (num_sample, input_dim) (i.e. Latent HNN by Dhulipala et al 2022).
            baseline : 'bool' If True, the class inplements the vanilla DNN whose outputs are used as gradient directly.
            field_type : 'str' It could be either 'solenoidal' or 'conservative'.
        """
        super().__init__()
        self.input_dim = input_dim
        self.nn = nn
        self.baseline = baseline
        self.field_type = field_type
        self.M = self.permutation_tensor(input_dim)

    def call(self, state):
        """
        Args:
            state : 'tensor' Combination of position and momentum.
        """
        if self.baseline:
            return self.nn(state) # In this case, the output must have shape (\cdot, self.input_dim) since it represents the gradient w.r.t. state.
        
        y = self.nn(state)
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
            return self.nn(state)
        
        conservative_field, solenoidal_field = tf.zeros_like(state), tf.zeros_like(state)

        with tf.GradientTape(persistent=True) as tape:
            tf.watch(state)
            F1, F2 = self.call(state)
            F1, F2 = tf.reduce_sum(F1), tf.reduce_sum(F2)
        
        if self.field_type == "conservative":
            dF1 = tf.gradient(F1, state)
            conservative_field = dF1
        if self.field_type == "solenoidal":
            dF2 = tf.gradient(F2, state)
            solenoidal_field = tf.matmul(dF2, self.M)
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


class MLP(tf.keras.Model):
    def __init__(self, input_dim, num_hidden, num_layers, output_dim, acti="relu"):
        """
        Generate the vanilla Neural Network.

        Args:
            num_hidden : 'int' the number of neurons.
            num_layers : 'int' the number of hidden layers.
            output_dim : 'int' the output dimension.
        """
        super().__init__()
        self.layers = [tf.keras.layers.Dense(num_hidden, activation=None) for idx_layer in range(num_layers)]
        self.lastlayer = tf.keras.layers.Dense(output_dim, activation=None, use_bias=False)
        self.acti = utils.choose_acti(acti)

        # Apply orthogonal initialization
        for idx_layer, layer in enumerate(self.layers + [self.lastlayer]):
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
        
    def call(self,x):
        for layer in self.layers:
            x = self.acti(layer(x))
        return self.lastlayer(x)
        