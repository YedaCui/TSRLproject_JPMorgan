# Code by Yeda CUI at department of SEEM of The Chinese Unviersity of Hong Kong
# Testing the functions in src.utils.py

import pytest
from src.utils import *
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_visible_devices(gpus[0], "GPU")


# ---------------------------------------------------------------------------
#                           Test leapfrog function
# ---------------------------------------------------------------------------
def mock_acceleration(state):
    """
    A mock acceleration function that returns constant acceleration.
    For simplicity, assume potential function is 0.5 * posi^2
    """
    position = state[:len(state) // 2]
    return -position

def test_leapfrog():
    initial_state = np.array([1.0, 0.0, 0.0, 1.0])  # Initial position and momentum
    dt = 0.1  # Time step
    num_lf = 10  # Number of leapfrog steps
    states = leapfrog(mock_acceleration, initial_state, dt, num_lf)
    # Check that the output shape is correct
    assert states.shape == (num_lf + 1, len(initial_state))
    # Ensure the first state matches the initial state
    assert np.allclose(states[0], initial_state)

    # Validate the energy conservation
    # Total energy (kinetic + potential) should remain approximately constant
    energies = []
    for state in states:
        position = state[:len(state) // 2]
        momentum = state[len(state) // 2:]
        potential_energy = 0.5 * np.sum(position ** 2)  
        kinetic_energy = 0.5 * np.sum(momentum ** 2)
        total_energy = potential_energy + kinetic_energy
        energies.append(total_energy)
    # Check that the total energy variation is within a small tolerance
    assert np.allclose(energies, energies[0], atol=1e-8)

# ---------------------------------------------------------------------------
#                           Test choose_acti function
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("activation, expected", [
    ("sin", tf.math.sin),
    ("tanh", tf.nn.tanh),
    ("relu", tf.nn.relu),
    ("invalid", None)
])
def test_choose_acti(activation, expected):
    assert choose_acti(activation) == expected

# ---------------------------------------------------------------------------
#                           Test get_timegrad_fn function
# ---------------------------------------------------------------------------
# Mock Hamiltonian function for testing
def mock_hamiltonian(state):
    """
    A simple quadratic Hamiltonian for testing:
        H = 0.5 * (posi^2 + momen^2)
    """
    position = state[:len(state) // 2]
    momentum = state[len(state) // 2:]
    return 0.5 * tf.reduce_sum(position**2 + momentum**2)

def test_timegrad_shape():
    timegrad_fn = get_timegrad_fn(mock_hamiltonian)
    state = np.array([1.0, 2.0, 3.0, 4.0])
    timegrad = timegrad_fn(state)
    print(timegrad)
    assert timegrad.shape == state.shape  # Time gradient should have the same shape as the state

def test_timegrad_values():
    timegrad_fn = get_timegrad_fn(mock_hamiltonian)
    state = np.array([1.0, 0.0, 0.0, 1.0])
    timegrad = timegrad_fn(state)
    # Expected result:
    # d(position)/dt = ∂H/∂momentum = [0.0, 1.0]
    # d(momentum)/dt = -∂H/∂position = [-1.0, 0.0]
    expected = np.array([0.0, 1.0, -1.0, 0.0])
    assert np.allclose(timegrad, expected, atol=1e-8)

# ---------------------------------------------------------------------------
#                           Test get_trajectory function
# ---------------------------------------------------------------------------
def test_get_trajectory():
    # Define test parameters
    initial_state = np.array([1.0, 0.0])  # Initial position (q=1) and momentum (p=0)
    dt = 0.01
    num_lf = 100
    require_grads = True
    # Time vector for evaluation
    t_eval = np.linspace(0, num_lf * dt, num_lf + 1)
    # Expected results by the analytical solution
    q0, p0 = initial_state[0], initial_state[1]
    expected_positions = q0 * np.cos(t_eval) + p0 * np.sin(t_eval)
    expected_momentum = p0 * np.cos(t_eval) - q0 * np.sin(t_eval)  
    # Call the function being tested
    states, timegrad, t_eval = get_trajectory(get_timegrad_fn(mock_hamiltonian), initial_state, dt, num_lf, require_grads)
    # Extract positions and momentum from the resulting states
    positions = states[:, 0]
    momentum = states[:, 1]
    # Assert that positions and momentum are close to the expected values
    assert np.allclose(positions, expected_positions, atol=1e-8), "Positions do not match expected values."
    assert np.allclose(momentum, expected_momentum, atol=1e-8), "Momentum, do not match expected values."
    # Assert time gradients (optional, depends on how you interpret timegrad_fn)
    for idx, grad in enumerate(timegrad):
        assert np.allclose(grad[:1], momentum[idx], atol=1e-8), "dq/dt (timegrad) does not match expected value."
        assert np.allclose(grad[1:], -positions[idx], atol=1e-8), "dp/dt (timegrad) does not match expected value."
    # Assert time evaluations
    assert len(t_eval) == num_lf + 1
    assert np.allclose(t_eval, np.linspace(0, num_lf * dt, num_lf + 1))

# ---------------------------------------------------------------------------
#                           Test get_pdesensor function
# ---------------------------------------------------------------------------
def test_get_pdesensor():
    num = 50
    seed = 0
    sensor = get_pdesensor(num=num, seed=seed)
    assert sensor.shape == (num, 2)  
    assert np.all(sensor >= 0) and np.all(sensor <= 3) 

# ---------------------------------------------------------------------------
#                           Test get_pdef function
# ---------------------------------------------------------------------------
def test_get_pdef():
    fval, (x, y, fg) = get_pdef(seed=0)
    assert len(fval) == 50 
    assert fg.shape == (301, 301)

# ---------------------------------------------------------------------------
#                           Test getZ function
# ---------------------------------------------------------------------------
def test_getZ():
    T, n, p = 500, 6, 8
    Z = getZ(seed=0, T=T, n=n, p=p)
    assert Z.shape == (T, n, p)

# ---------------------------------------------------------------------------
#                           Test getglmmdata function
# ---------------------------------------------------------------------------
def test_getglmmdata():
    seed = 42
    T, n = 500, 6
    Z = np.ones((T, n, 8))  # Use a simple design matrix for testing
    # Generate data with the same seed
    Y1 = getglmmdata(seed=seed, Z=Z)
    Y2 = getglmmdata(seed=seed, Z=Z)
    # Assert reproducibility
    assert np.array_equal(Y1, Y2), "Generated data should be identical for the same seed."
    # Test output dimensions
    assert Y1.shape == (T,n)
    # Test binary output
    assert np.all(np.isin(Y1, [0, 1])), "Output Y should only contain binary values (0 or 1)."

# ---------------------------------------------------------------------------
#                           Test H_A_sol function
# ---------------------------------------------------------------------------
def test_H_A_sol():
    # Test 1: Basic functionality (known input and output)
    theta, u, rho, p = 1.0, 2.0, 0.5, -1.0
    dt = np.pi / 2  # 90 degrees
    theta1, u1, rho1, p1 = H_A_sol(theta, u, rho, p, dt)

    # Expected solution
    expected_theta1 = theta + dt * rho
    expected_u1 = p * np.sin(dt) + u * np.cos(dt)
    expected_rho1 = rho
    expected_p1 = p * np.cos(dt) - u * np.sin(dt)

    assert np.isclose(theta1, expected_theta1), "Theta evolution is incorrect."
    assert np.isclose(u1, expected_u1), "u evolution is incorrect."
    assert np.isclose(rho1, expected_rho1), "rho evolution is incorrect."
    assert np.isclose(p1, expected_p1), "p evolution is incorrect."

    # Initial conditions
    theta, u, rho, p = 1.0, 2.0, 0.5, -1.0
    dt = 0.1
    num_steps = 100 
    # Compute initial Hamiltonian
    H_A = lambda u, rho, p: 0.5 * (u**2 + rho**2 + p**2)
    H0 = H_A(u,rho,p)

    # Evolve the system and check energy conservation
    for _ in range(num_steps):
        theta, u, rho, p = H_A_sol(theta, u, rho, p, dt)
        H1 = H_A(u, rho, p)
        assert np.isclose(H0, H1, atol=1e-8), f"Hamiltonian is not conserved: H0={H0}, H1={H1}"

# ---------------------------------------------------------------------------
#                          Test getpmgrad_fn function
# ---------------------------------------------------------------------------
def test_getpmgrad_fn():
    # A simple quadratic Hamiltonian for testing
    def H(state):
        return 0.5 * tf.reduce_sum(state**2)
    # Create the gradient function
    pmgrad_fn = getpmgrad_fn(H)
    theta = tf.constant([1.0, 2.0])
    u = tf.constant([3.0, 4.0])
    rho = tf.constant([5.0, 6.0])
    p = tf.constant([7.0, 8.0])
    grad_rho, grad_p = pmgrad_fn(theta, u, rho, p)
    # Expected gradients (manually derived)
    expected_grad_rho = -theta
    expected_grad_p = -u
    assert np.allclose(grad_rho, expected_grad_rho), f"Expected grad_rho={expected_grad_rho}, but got {grad_rho}."
    assert np.allclose(grad_p, expected_grad_p), f"Expected grad_p={expected_grad_p}, but got {grad_p}."

# ---------------------------------------------------------------------------
#                          Test H_B_sol function
# ---------------------------------------------------------------------------
def test_H_B_sol():
    # A simple case with H_B(theta, u, rho ,p) = 0.5 * (theta^2 + u^2).
    def H_B(state):
        return 0.5 * tf.reduce_sum(state[:len(state)//2]**2)
    theta = np.array([1.0]) 
    u = np.array([2.0])
    rho = np.array([3.0])
    p = np.array([4.0])
    dt = 0.01
    num_steps = 100 
    # Compute initial Hamiltonian
    H0 = H_B(np.concat([theta,u,rho,p]))
    # Evolve the system and check energy conservation
    for _ in range(num_steps):
        theta, u, rho, p = H_B_sol(theta, u, rho, p, getpmgrad_fn(H_B), dt)
        H1 = H_B(np.concat([theta,u,rho,p]))
        assert np.isclose(H0, H1, atol=1e-8), f"Hamiltonian is not conserved: H0={H0}, H1={H1}"


# ---------------------------------------------------------------------------
#                          Test pmintegrator function
# ---------------------------------------------------------------------------
def test_pmintegrator():
    # A simple case with H_A(theta, u, rho, p) = 0.5 * (u^2 + rho^2 + p^2).
    def H_A(state):
        return 0.5 * tf.reduce_sum(state[1:]**2)
    # A simple case with H_B(theta, u, rho, p) = 0.5 * (theta^2 + u^2).
    def H_B(state):
        return 0.5 * tf.reduce_sum(state[:len(state)//2]**2)
    # Initial conditions
    theta = np.array([1.0])
    u = np.array([2.0])
    rho = np.array([3.0])
    p = np.array([4.0])
    dt = 0.005
    num_int = 100  # Number of integration steps
    # Compute initial Hamiltonian
    H0 = H_A(np.concat([theta,u,rho,p])) + H_B(np.concat([theta,u,rho,p]))
    states = pmintegrator(theta, u, rho, p, getpmgrad_fn(H_B), dt, num_int)
    # Check energy conservation
    for state in states:
        H = H_A(state) + H_B(state)
        assert np.isclose(H0, H, atol=1e-8), f"Hamiltonian is not conserved: H0={H0}, H={H}"