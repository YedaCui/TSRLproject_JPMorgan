# TSRLproject_JPMorgan

This is a project for the **AI & Data Science Associate Internship Program** of the Machine Learning Center of Excellence.

This repository implements the **Latent Hamiltonian Neural Network (LHNN)** introduced in *Dhulipala et al., 2023* and the **Pseudo Marginal Hamiltonaian Monte Carlo (PMHMC)** describled in *Alenlov et al, 2021*. Additionally, this repository explores the application of  **LHNN** to enhance **PMHMC**. The project is built using **Python 3** and **TensorFlow 2.18.0**. 

All resource files and code are organized in the `src/` folder.

---

## **Project Structure**

- **`src/functions/`**: Contains all Hamiltonian functions.
- **`src/mcmc.py`**: Implements Hamiltonian Monte Carlo (HMC), No-U-Turn Sampler (NUTS), and Pseudo-Marginal HMC.
- **`src/hnn.py`**: Defines the Hamiltonian Neural Network (HNN) classes.
- **`src/gene_data.py`**: Contains functions to generate datasets.
- **`src/train.py`**: Handles the training process for the LHNN.
- **`src/configs.py`**: Stores configurations for different experiments.

All test files corresponding to the `src/` scripts are located in the `tests/` folder.

---

## Usage

### 1. Training the LHNN
To train the Latent Hamiltonian Neural Network, run:

```bash
python -m src/train.py
```

### 2. Generating Datasets
To generate a synthetic dataset for training and evaluation, use:

```bash
python -m src/gene_data.py
```

### 3. Testing
All classes and functions in the **`src/`** folder have corresponding test files in the **`tests/`** folder. To run the tests, execute the following command from the root directory:
```bash
pytest
```

---

## References

This implementation is based on the paper:
- Dhulipala et al., 2023: Latent Hamiltonian Neural Networks for Dynamical Systems.
- Greydanus et al., 2019: Hamiltonian Neural Networks.
- Alenlov et al, 2021: Pseudo-Marginal Hamiltonian Monte Carlo.

---

## License

This repository is licensed under the MIT License.