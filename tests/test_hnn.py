import pytest
from src.hnn import *
import tensorflow as tf
import numpy as np
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_visible_devices(gpus[0], "GPU")

def test_hnn():
    """
    Unified test for the HNN class. 
    Tests initialization, forward pass, gradient computation, and serialization.
    """
    input_dim = 4
    num_hidden = 16
    num_layers = 3
    output_dim = 4
    # ** Test Initialization **
    model = HNN(input_dim, num_hidden, num_layers, output_dim)
    # Check if the model is initialized correctly
    assert len(model.hidden_layers) == num_layers, "Incorrect number of hidden layers."
    assert isinstance(model.lastlayer, tf.keras.layers.Dense), "Last layer is not Dense."
    assert model.lastlayer.units == output_dim, "Last layer output dimension is incorrect."
    assert model.acti == tf.nn.relu, "Activation function is not ReLU."
    # Check if the layers are initialized with orthogonal weights
    for layer in model.hidden_layers + [model.lastlayer]:
        if layer.kernel is not None:
            orthogonal_initializer = tf.keras.initializers.Orthogonal()
            expected_shape = layer.kernel.shape
            expected_weights = orthogonal_initializer(expected_shape)
            assert layer.kernel.shape == expected_shape, "Kernel shape is incorrect."
    # Test Forward Pass - Baseline=True
    baseline_model = HNN(input_dim, num_hidden, num_layers, output_dim, baseline=True)
    state = tf.random.normal((5, input_dim))
    baseline_output = baseline_model(state)
    assert baseline_output.shape == (5, input_dim), "Output shape is incorrect for baseline=True."
    # Test Forward Pass - Baseline=False
    non_baseline_model = HNN(input_dim, num_hidden, num_layers, output_dim, baseline=False)
    F1, F2 = non_baseline_model(state)
    assert F1.shape == (5, output_dim // 2), "F1 output shape is incorrect."
    assert F2.shape == (5, output_dim // 2), "F2 output shape is incorrect."
    # Test Gradient Computation - Conservative Field
    conservative_model = HNN(input_dim, num_hidden, num_layers, output_dim, field_type="conservative", baseline=False)
    gradients = conservative_model.get_gradient(state)
    assert gradients.shape == state.shape, "Gradient shape is incorrect for conservative field."
    # Test Gradient Computation - Separate Fields
    solenoidal_model = HNN(input_dim, num_hidden, num_layers, output_dim, field_type="solenoidal", baseline=False)
    conservative_field, solenoidal_field = solenoidal_model.get_gradient(state, separate_fields=True)
    assert conservative_field.shape == state.shape, "Conservative field shape is incorrect."
    assert solenoidal_field.shape == state.shape, "Solenoidal field shape is incorrect."

def test_pmhnn():
    """
    Unified test for the PMHNN class.
    Tests initialization, forward pass, gradient computation, and serialization.
    """
    input_dim = 14
    num_hidden = 16
    num_layers = 3
    dim_marginal = 5
    # Generate mock data for coefficients and observations
    num_observations = 5
    coef = np.random.normal(size=(num_observations, 5)) 
    obs = np.random.normal(size=(num_observations, 3))
    # ** Test Initialization **
    model = PMHNN(input_dim, num_hidden, num_layers, dim_marginal=dim_marginal, coef=coef, obs=obs)
    # Check model attributes
    assert model.input_dim == input_dim, "Input dimension mismatch."
    assert model.num_hidden == num_hidden, "Number of hidden units mismatch."
    assert model.num_layers == num_layers, "Number of layers mismatch."
    assert model.dim_marginal == dim_marginal, "Dimension of marginal variables mismatch."
    assert model.T == num_observations, "Number of observations mismatch."
    # Check layer initialization
    assert len(model.hidden_layers) == num_layers, "Incorrect number of hidden layers."
    assert model.lastlayer.units == 1, "Last layer output dimension is incorrect."
    # ** Test Forward Pass **
    batch_size = 4
    state = tf.random.normal((batch_size, 20))  # Example state input (batch_size, input_dim)
    output = model(state)
    # Check output shape
    assert output.shape == (batch_size, 1), "Output shape is incorrect."
    # ** Test Gradient Computation **
    gradients = model.get_gradient(state)
    # Check gradient shape
    assert gradients.shape == state.shape, "Gradient shape is incorrect."