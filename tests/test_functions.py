import pytest
from src.functions import *
import tensorflow as tf
import numpy as np
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_visible_devices(gpus[0], "GPU")

@pytest.mark.skip(reason="This is a helpr function")
def testdistribution(dist_name, state, expected_result, atol=1e-6):
    """
    Test a distribution's H_function by comparing output to expected result.

    Args:
        dist_name (str): The name of the distribution.
        state (tf.Tensor): The state input to the H_function.
        expected_result (float): The expected result for the H_function.
        atol (float): Absolute tolerance for the comparison.
    """
    H_function = functions(dist_name)
    result = H_function(state)
    assert isinstance(result, tf.Tensor), f"{dist_name} did not return a TensorFlow tensor."
    assert result.shape == (), f"{dist_name} did not return a scalar tensor."
    assert tf.abs(result - expected_result) < atol, f"{dist_name} failed: {result.numpy()} != {expected_result}"

def test_1D_Gauss_mix():
    state = tf.convert_to_tensor([0.0, 0.0], dtype=tf.float64)
    mu1, mu2, sigma = 1.0, -1.0, 0.35
    expected_result = -np.log(0.5 * np.exp(-(mu1)**2 / (2 * sigma**2)) + 0.5 * np.exp(-(mu2)**2 / (2 * sigma**2)))
    testdistribution("1D_Gauss_mix", state, expected_result)

def test_nD_Rosenbrock():
    state = tf.convert_to_tensor([1.0, 1.0, 1.0, 1.0], dtype=tf.float64)
    q, p = np.array([1.0, 1.0]), np.array([1.0, 1.0])
    term1 = np.sum((100 * (q[1:] - q[:-1]**2)**2 + (1 - q[:-1])**2) / 20)
    term2 = np.sum(np.array(p)**2 / 2)
    expected_result = term1 + term2
    testdistribution("nD_Rosenbrock", state, expected_result)

def test_2D_Nealsfunnel():
    state = tf.convert_to_tensor([1.0, 2.0, 1.0, 1.0], dtype=tf.float64)
    q1, q2, p = 1.0, 2.0, [1.0, 1.0]
    term1 = q1**2 / (2 * 3**2) + q2**2 / (2 * np.exp(q1))
    term2 = np.sum(np.array(p)**2 / 2)
    expected_result = term1 + term2
    testdistribution("2D_Nealsfunnel", state, expected_result)

def test_5D_illconditioned_Gaussian():
    var = np.array([0.01, 0.1, 1, 10, 100], dtype=np.float64)
    state = tf.convert_to_tensor([1.0] * 10, dtype=tf.float64)
    q, p = np.array([1.0] * 5), np.array([1.0] * 5)
    term1 = np.sum(q**2 / (2 * var))
    term2 = np.sum(p**2 / 2)
    expected_result = term1 + term2
    testdistribution("5D_illconditioned_Gaussian", state, expected_result)

def test_AllenCahn():
    dx = 1 / 25
    state = tf.convert_to_tensor(np.random.rand(50), dtype=tf.float64)
    q, p = state[:25].numpy(), state[25:].numpy()
    V = lambda x: (1 - x**2)**2
    term1 = np.sum((q[1:] - q[:-1])**2 / (2 * dx) + dx / 2 * (V(q[1:]) + V(q[:-1])))
    term2 = np.sum(p**2 / 2)
    expected_result = term1 + term2
    testdistribution("AllenCahn", state, expected_result)

def test_ellipticpde():
    state = tf.convert_to_tensor(np.random.rand(100), dtype=tf.float64)
    H_function = functions("ellipticpde")
    result = H_function(state)
    assert isinstance(result, tf.Tensor)
    assert result.shape == ()

def test_pmglmmA():
    state = tf.zeros(100, dtype=tf.float64)
    H_function = functions("pmglmmA")
    expected_results = 0
    testdistribution("pmglmmA", state, expected_results)

def test_pmglmmB():
    state = tf.convert_to_tensor(np.random.rand((13+500*128)*2), dtype=tf.float64)
    H_function = functions("pmglmmB")
    result = H_function(state)
    assert isinstance(result, tf.Tensor)
    assert result.shape == ()

def test_pmglmm():
    state = tf.convert_to_tensor(np.random.rand((13+500*128)*2), dtype=tf.float64)
    H_function = functions("pmglmm")
    result = H_function(state)
    assert isinstance(result, tf.Tensor)
    assert result.shape == ()