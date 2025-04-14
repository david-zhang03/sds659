import os
import cupy as cp
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

def generate_xor_data(n, d):
    # Generate data from {±1/sqrt(d)}^d and labels as XOR of first 2 features.
    X = cp.random.choice([1, -1], size=(n, d)) / cp.sqrt(d)
    y = cp.sign(X[:, 0] * X[:, 1]).astype(cp.float32)
    return X, y

def neuron_activation(x, z, R_bar=15):
    # x is in R^(d+2): x[:d] are weights, x[d] is a bias for tanh(z^T x[:d]), and x[d+1] is a further bias.
    d = len(z)
    a = cp.dot(z, x[:d]) + x[d]
    b = x[d+1]
    return R_bar * (cp.tanh(a) + 2 * cp.tanh(b)) / 3.0

def dtanh(x):
    return 1.0 - cp.tanh(x)**2

def neuron_grad(x, z, R_bar=15):
    # Compute gradients of h_x with respect to the parameters x.
    d = len(z)
    a = cp.dot(z, x[:d]) + x[d]
    b = x[d+1]
    grad_x1 = (R_bar/3.0) * dtanh(a) * z
    grad_bias = cp.array([R_bar/3.0 * dtanh(a)])
    grad_bias2 = cp.array([2*R_bar/3.0 * dtanh(b)])
    return cp.concatenate([grad_x1, grad_bias, grad_bias2])

def logistic_loss(f, y):
    val = -y * f
    return cp.maximum(0, val) + cp.log(1 + cp.exp(-cp.abs(val)))

def d_logistic_loss(f, y):
    return -y / (1 + cp.exp(y * f))

def predict(particles, z, R_bar=15):
    # Ensure list comprehension result is converted to cupy array
    outputs = cp.array([neuron_activation(x, z, R_bar) for x in particles])
    return cp.mean(outputs)

def evaluate_accuracy(particles, X, y, R_bar=15):
    n_test = X.shape[0]
    correct = 0
    # Loop for potentially lower memory usage on GPU
    for i in range(n_test):
         pred_val = predict(particles, X[i], R_bar)
         # Ensure comparison happens correctly (GPU/CPU)
         if cp.sign(pred_val) == y[i]:
             correct += 1
    # Ensure result is a standard float
    return float(correct) / n_test

# --- Modified update_particles for Regular Mean Field (Gradient Descent) ---
def update_particles_mf(particles, X, y, eta, lambda_1, R_bar=15):
    """
    Mean Field (Gradient Descent) update step for particles.
    Equivalent to MFLD with noise variance lam=0.
    args:
        lambda_1: denotes strength of L2 reg.
    """
    n, d = X.shape
    N = len(particles)
    # Compute network output for each data point
    f_preds = cp.zeros(n, dtype=cp.float32)
    for i in range(n):
        f_preds[i] = cp.mean(cp.array([neuron_activation(x, X[i], R_bar) for x in particles]))
    # Compute derivative of the loss for each data point
    loss_derivs = cp.array([d_logistic_loss(f_preds[i], y[i]) for i in range(n)])

    new_particles = []
    for x in particles:
        grad_loss = cp.zeros_like(x)
        # Accumulate gradient contributions from all training data
        for i in range(n):
            grad_loss += loss_derivs[i] * neuron_grad(x, X[i], R_bar)
        grad_loss = grad_loss / n # Corrected typo
        grad_reg = 2 * lambda_1 * x
        grad = grad_loss + grad_reg
        
        # no noise
        x_new = x - eta * grad
        new_particles.append(x_new)
    return new_particles

# --- Modified run_simulation_and_store for Regular Mean Field ---
def run_simulation_mf_and_store(d=20, n_train=500, n_test=200,
                                num_particles=1000, eta=0.05, T_total=1200,
                                lambda_1=0.1, R_bar=15,
                                save_dir='mf_results',
                                store_interval=200): # How often to store particle state
    """Runs regular mean field (gradient descent) simulation and stores results."""

    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved in: {save_dir}")
    start_time = time.time()

    X_train, y_train = generate_xor_data(n_train, d)
    X_test, y_test = generate_xor_data(n_test, d)
    particles = [cp.random.randn(d+2, dtype=cp.float32) for _ in range(num_particles)]

    particle_history = {} # Dictionary to store particle states in memory for plotting
    accuracy_history = []

    print("Storing initial particle state (Iter 0)...")
    initial_particle_matrix_cp = cp.stack(particles)
    np.save(os.path.join(save_dir, 'particles_iter_0.npy'), cp.asnumpy(initial_particle_matrix_cp))
    particle_history[0] = [p.copy() for p in particles] # Store initial state for plotting

    # --- Main Simulation Loop (Gradient Descent) ---
    for t in range(T_total):
        particles = update_particles_mf(particles, X_train, y_train, eta, lambda_1, R_bar)

        # Store state and evaluate periodically
        if (t + 1) % store_interval == 0 or t == T_total - 1:
            iter_actual = t + 1
            print(f"--- Iteration {iter_actual}/{T_total} ---")

            # Save Particles
            print(f"Storing particle state at iteration {iter_actual}...")
            current_particle_matrix_cp = cp.stack(particles)
            np.save(os.path.join(save_dir, f'particles_iter_{iter_actual}.npy'), cp.asnumpy(current_particle_matrix_cp))
            particle_history[iter_actual] = [p.copy() for p in particles] # Store for plotting

            # Evaluate accuracy
            acc = evaluate_accuracy(particles, X_test, y_test, R_bar)
            print(f"Test accuracy at iteration {iter_actual}: {acc:.4f}")
            accuracy_history.append({'iteration': iter_actual, 'accuracy': acc})
        elif (t+1) % 50 == 0: # Print progress less often otherwise
             print(f"  Iteration {t+1} / {T_total}")


    # --- Save Final Accuracy History ---
    # Save as a structured numpy array or simple text file
    acc_data = np.array([(item['iteration'], item['accuracy']) for item in accuracy_history],
                        dtype=[('iteration', 'i4'), ('accuracy', 'f4')])
    np.save(os.path.join(save_dir, 'accuracy_history.npy'), acc_data)

    end_time = time.time()
    print(f"\nSimulation finished in {end_time - start_time:.2f} seconds.")
    print("Saved accuracy history.")

    return particle_history, accuracy_history

# --- Simulation Setup ---
cp.random.seed(42)
SAVE_DIR_MF = "/home/ddz5/Desktop/sds659/results/mf_results" # Define the directory name for Mean Field results

# Simulation Parameters
params = {
    'd': 20,
    'n_train': 500,
    'n_test': 200,
    'num_particles': 1000, 
    'eta': 0.05,
    'T_total': 1500, # Total iterations (e.g., 6 rounds * 250 steps/round)
    'lambda_1': 0.1,
    'R_bar': 15,
    'save_dir': SAVE_DIR_MF,
    'store_interval': 100 # Store state every 100 iterations
}

# Run simulation and store results
particle_history, accuracy_history = run_simulation_mf_and_store(**params)

# --- Plotting and Saving Plots ---

print("\n--- Generating and Saving Plots ---")
d_value = params['d'] # Get d value for plotting labels

# Plot 1: Test Accuracy Evolution
plt.figure(figsize=(8, 6))
iters = [item['iteration'] for item in accuracy_history]
accs = [item['accuracy'] for item in accuracy_history]
plt.plot(iters, accs, marker='o')
plt.xlabel('Training Iteration')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy over Training Iterations (Mean Field)')
plt.grid(True)
plt.ylim(bottom=0.4) # Adjust y-axis start if needed
plt.savefig(os.path.join(SAVE_DIR_MF, 'mf_accuracy_vs_iteration.png'))
plt.show()

# Plot 2: PCA Visualization
plt.figure(figsize=(10, 8))
num_snapshots = len(particle_history)
colors = plt.cm.viridis(np.linspace(0, 1, num_snapshots))
pca = PCA(n_components=2)
scaler = StandardScaler()

stored_iters = sorted(particle_history.keys())
for i, iter_num in enumerate(stored_iters):
    particles_cp_list = particle_history[iter_num]
    print(f"Processing PCA for iteration {iter_num}")
    particle_matrix_cp = cp.stack(particles_cp_list)
    particle_matrix_np = cp.asnumpy(particle_matrix_cp)
    particle_matrix_scaled = scaler.fit_transform(particle_matrix_np)
    particles_pca = pca.fit_transform(particle_matrix_scaled)
    plt.scatter(particles_pca[:, 0], particles_pca[:, 1],
                alpha=0.6, label=f'Iter {iter_num}', color=colors[i], s=10)

plt.title('PCA of Mean Field Particle Parameters Over Training')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR_MF, 'mf_pca_particles_over_iterations.png'))
plt.show()

# Plot 3: Weight Component Histograms
iters_to_compare = [0, max(particle_history.keys())] # Compare initial and final stored state
num_bins = 50
fig, axes = plt.subplots(1, len(iters_to_compare), figsize=(12, 5), sharey=True)
fig.suptitle('Distribution of Relevant vs. Irrelevant Weight Components (Mean Field)')

for i, iter_num in enumerate(iters_to_compare):
    print(f"Processing weight distributions for iteration {iter_num}")
    ax = axes[i]
    particles_cp_list = particle_history[iter_num]
    particle_matrix_cp = cp.stack(particles_cp_list)
    weights_cp = particle_matrix_cp[:, :d_value]
    relevant_weights_cp = weights_cp[:, :2].flatten()
    irrelevant_weights_cp = weights_cp[:, 2:].flatten()
    relevant_weights_np = cp.asnumpy(relevant_weights_cp)
    irrelevant_weights_np = cp.asnumpy(irrelevant_weights_cp)

    bin_min = np.min([relevant_weights_np.min(), irrelevant_weights_np.min()])
    bin_max = np.max([relevant_weights_np.max(), irrelevant_weights_np.max()])
    bins = np.linspace(bin_min, bin_max, num_bins)

    ax.hist(relevant_weights_np, bins=bins, alpha=0.7, label='Relevant Weights (Dims 1-2)', density=True)
    ax.hist(irrelevant_weights_np, bins=bins, alpha=0.7, label=f'Irrelevant Weights (Dims 3-{d_value})', density=True)
    ax.set_title(f'Iteration {iter_num}')
    ax.set_xlabel('Weight Value')
    if i == 0:
        ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(SAVE_DIR_MF, 'mf_weight_distribution_comparison.png'))
plt.show()

# Plot 4: Relevant Weight Scatter Plot
plt.figure(figsize=(10, 8))
for i, iter_num in enumerate(stored_iters):
    print(f"Processing scatter plot for iteration {iter_num}")
    particles_cp_list = particle_history[iter_num]
    particle_matrix_cp = cp.stack(particles_cp_list)
    weights_cp = particle_matrix_cp[:, :d_value]
    relevant_weights_cp = weights_cp[:, :2]
    relevant_weights_np = cp.asnumpy(relevant_weights_cp)
    plt.scatter(relevant_weights_np[:, 0], relevant_weights_np[:, 1],
                alpha=0.3, label=f'Iter {iter_num}', color=colors[i], s=10)

plt.title('Scatter Plot of First Two Weight Components Over Training (Mean Field)')
plt.xlabel('Weight Component 1')
plt.ylabel('Weight Component 2')
plt.grid(True)
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.savefig(os.path.join(SAVE_DIR_MF, 'mf_relevant_weights_scatter.png'))
plt.show()

print(f"\nAll plots saved to {SAVE_DIR_MF}")
print(f"Particle states saved as .npy files in {SAVE_DIR_MF}")