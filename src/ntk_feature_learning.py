import os
import cupy as cp 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

def generate_xor_data(n, d):
    X = cp.random.choice([1, -1], size=(n, d)) / cp.sqrt(d)
    y = cp.sign(X[:, 0] * X[:, 1]).astype(cp.float32)
    return X, y

def neuron_activation(x, z, R_bar=15):
    d = len(z)
    a = cp.dot(z, x[:d]) + x[d]
    b = x[d+1]
    return R_bar * (cp.tanh(a) + 2 * cp.tanh(b)) / 3.0

def d_logistic_loss(f, y):
    return -y / (1.0 + cp.exp(y * f))

# --- NTK Specific Prediction and Evaluation ---

def ntk_predict(a, neurons, z, R_bar=15):
    """
    NTK network prediction: f(z) = mean(a * h_{neurons}(z))
    'a' are the trainable output weights (cupy array, shape N)
    'neurons' is a list of fixed neuron parameter vectors (cupy arrays, shape d+2)
    'z' is the input (cupy array, shape d)
    """
    N = len(neurons)
    # Calculate activations for fixed neurons for input z
    # Ensure list comprehension result is converted to cupy array
    outputs = cp.array([neuron_activation(n, z, R_bar) for n in neurons])
    # Weighted sum (element-wise product) and mean
    return cp.mean(a * outputs)

def ntk_evaluate_accuracy(a, neurons, X, y, R_bar=15):
    """Evaluate NTK accuracy."""
    n_test = X.shape[0]
    correct = 0
    for i in range(n_test):
        pred_val = ntk_predict(a, neurons, X[i], R_bar)
        if cp.sign(pred_val) == y[i]:
            correct += 1
    return float(correct) / n_test

# --- Simulation Function for NTK with Storing ---

def run_ntk_simulation_and_store(d=20, n_train=500, n_test=200,
                                 num_neurons=1000, eta=0.01, T_total=200, # Use T_total
                                 R_bar=15,
                                 save_dir='ntk_results',
                                 store_interval=20): # How often to evaluate and store accuracy
    """Runs NTK simulation (fixed features, trains output weights) and stores results."""

    os.makedirs(save_dir, exist_ok=True)
    print(f"NTK Results will be saved in: {save_dir}")
    start_time = time.time()

    X_train, y_train = generate_xor_data(n_train, d)
    X_test, y_test = generate_xor_data(n_test, d)

    # Initialize fixed neurons (scale with 1 / \sqrt{d+2} should be more stable)
    neurons = [cp.random.randn(d+2, dtype=cp.float32) / cp.sqrt(d+2) for _ in range(num_neurons)]
    # Initialize trainable output weights (on GPU)
    a = cp.ones(num_neurons, dtype=cp.float32) # Start with equal weights

    n = n_train
    N = num_neurons

    accuracy_history = []
    output_weights_history = {} # Store output weights 'a' over time

    # --- Save Initial Fixed Neurons (only once) ---
    print("Saving initial fixed neuron parameters...")
    initial_neuron_matrix_cp = cp.stack(neurons)
    np.save(os.path.join(save_dir, 'ntk_initial_neurons.npy'), cp.asnumpy(initial_neuron_matrix_cp))
    # Store initial weights 'a'
    output_weights_history[0] = cp.copy(a)

    # Precompute neuron activations for training data (using fixed neurons)
    # This avoids recalculating them in every gradient step
    # Shape: (n_train, num_neurons)
    print("Precomputing fixed neuron activations for training data...")
    train_activations = cp.zeros((n, N), dtype=cp.float32)
    for i in range(n):
        train_activations[i, :] = cp.array([neuron_activation(neu, X_train[i], R_bar) for neu in neurons])
    print("Precomputation finished.")

    # --- Main NTK Training Loop ---
    for t in range(T_total):
        # Calculate predictions using current weights 'a' and precomputed activations
        # f_preds[i] = mean(a * train_activations[i, :])
        # Vectorized calculation: mean(a * train_activations, axis=1)
        f_preds = cp.mean(a * train_activations, axis=1) # Efficient calculation

        # Compute loss derivative for each sample based on current predictions (recall the loss here doesn't really matter in NTK)
        loss_derivs = cp.array([d_logistic_loss(f_preds[i], y_train[i]) for i in range(n)])

        # Compute gradient ONLY for output weights 'a'
        # grad_a[j] = mean_over_data [ dL/df_i * df_i/da_j ]
        # df_i/da_j = (1/N) * h_{x_j0}(z_i) = (1/N) * train_activations[i, j]
        # grad_a[j] = mean_over_data [ loss_derivs[i] * (1/N) * train_activations[i, j] ]
        grad_a_contributions = loss_derivs[:, cp.newaxis] * train_activations # Shape (n, N)
        grad_a = cp.mean(grad_a_contributions, axis=0) / N # Average over data, include 1/N from prediction func

        # Update output weights 'a'
        a = a - eta * grad_a

        # Store state and evaluate periodically
        if (t + 1) % store_interval == 0 or t == T_total - 1:
            iter_actual = t + 1
            print(f"--- Iteration {iter_actual}/{T_total} ---")

            avg_loss = cp.mean(cp.log1p(cp.exp(-y_train * f_preds)))
            print(f"Avg logistic loss at iter {t+1}: {avg_loss:.4f}")

            # Save output weights 'a'
            print(f"Storing output weights 'a' at iteration {iter_actual}...")
            output_weights_history[iter_actual] = cp.copy(a)
            np.save(os.path.join(save_dir, f'ntk_output_weights_iter_{iter_actual}.npy'), cp.asnumpy(a))

            # Evaluate accuracy
            acc = ntk_evaluate_accuracy(a, neurons, X_test, y_test, R_bar)
            print(f"Test accuracy at iteration {iter_actual}: {acc:.4f}")
            accuracy_history.append({'iteration': iter_actual, 'accuracy': acc})
        # Removed less frequent print statement for clarity

    # --- Save Final Accuracy History ---
    acc_data = np.array([(item['iteration'], item['accuracy']) for item in accuracy_history],
                        dtype=[('iteration', 'i4'), ('accuracy', 'f4')])
    np.save(os.path.join(save_dir, 'ntk_accuracy_history.npy'), acc_data)

    end_time = time.time()
    print(f"\nNTK Simulation finished in {end_time - start_time:.2f} seconds.")
    print("Saved accuracy history and output weights.")

    # Return initial neurons (for potential PCA plot) and accuracy history
    return initial_neuron_matrix_cp, accuracy_history


# --- Simulation Setup ---
cp.random.seed(42)
SAVE_DIR_NTK = "/home/ddz5/Desktop/sds659/results/ntk_results" # Define the directory name for NTK results

# Simulation Parameters
params_ntk = {
    'd': 20,
    'n_train': 500,
    'n_test': 200,
    'num_neurons': 1000, # Corresponds to num_particles in MF/MFLD
    'eta': 0.05,       # Learning rate for output weights might need tuning
    'T_total': 1500,     # Total iterations
    'R_bar': 15,
    'save_dir': SAVE_DIR_NTK,
    'store_interval': 100 # Store state less frequently
}

# Run NTK simulation and store results
initial_neurons_cp, accuracy_history_ntk = run_ntk_simulation_and_store(**params_ntk)

# --- Plotting and Saving Plots ---

print("\n--- Generating and Saving NTK Plots ---")
d_value = params_ntk['d'] # Get d value for plotting labels

# Plot 1: Test Accuracy Evolution
plt.figure(figsize=(8, 6))
iters_ntk = [item['iteration'] for item in accuracy_history_ntk]
accs_ntk = [item['accuracy'] for item in accuracy_history_ntk]
plt.plot(iters_ntk, accs_ntk, marker='o')
plt.xlabel('Training Iteration')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy over Training Iterations (NTK)')
plt.grid(True)
plt.ylim(bottom=0.4)
plt.savefig(os.path.join(SAVE_DIR_NTK, 'ntk_accuracy_vs_iteration.png'))
plt.show()

# Plot 2: PCA Visualization of *Initial Fixed* Neurons
print("Processing PCA for initial fixed neurons...")
plt.figure(figsize=(10, 8))
pca_ntk = PCA(n_components=2)
scaler_ntk = StandardScaler()

# Use the initial neurons matrix saved earlier
initial_neurons_np = cp.asnumpy(initial_neurons_cp) # Transfer if not already CPU
neurons_scaled = scaler_ntk.fit_transform(initial_neurons_np)
neurons_pca = pca_ntk.fit_transform(neurons_scaled)

plt.scatter(neurons_pca[:, 0], neurons_pca[:, 1], alpha=0.6, s=10)
plt.title('PCA of Initial Fixed Neuron Parameters (NTK Features)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR_NTK, 'ntk_pca_initial_neurons.png'))
plt.show()

print(f"\nNTK plots saved to {SAVE_DIR_NTK}")
print(f"Initial NTK neurons and final output weights saved as .npy files in {SAVE_DIR_NTK}")