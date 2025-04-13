import os

import numpy as np
import cupy as cp

import matplotlib.pyplot as plt

def generate_xor_data(n, d):
    # Generate data from {±1/sqrt(d)}^d and labels as XOR of first 2 features.
    X = cp.random.choice([1, -1], size=(n, d)) / cp.sqrt(d)
    y = cp.sign(X[:, 0] * X[:, 1])
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
    grad_x1 = (R_bar/3.0) * dtanh(a) * z          # gradient w.r.t. x1
    grad_bias = cp.array([R_bar/3.0 * dtanh(a)])      # gradient w.r.t. bias x[d]
    grad_bias2 = cp.array([2*R_bar/3.0 * dtanh(b)])   # gradient w.r.t. bias x[d+1]
    return cp.concatenate([grad_x1, grad_bias, grad_bias2])

def logistic_loss(f, y):
    return cp.log(1 + cp.exp(-y * f))

def d_logistic_loss(f, y):
    return - y / (1 + cp.exp(y * f))

def predict(particles, z, R_bar=15):
    outputs = cp.array([neuron_activation(x, z, R_bar) for x in particles])
    return cp.mean(outputs)

def evaluate_accuracy(particles, X, y, R_bar=15):
    preds = cp.array([cp.sign(predict(particles, x, R_bar)) for x in X])
    return cp.mean(preds == y)

def update_particles(particles, X, y, eta, lam, lambda_1, R_bar=15):
    """
    Note that the update step follows from eq (4) of (Suzuki et al., 2023)
    Our L2 regularization update is implicity included in \nabla \frac{\delta F(\mu_i)}}{\delta \mu} of eq (4)
    args:
        lam: denotes the variance of the noise updates in Langevin dynamics (i.e. \sqrt{2 \lambda \eta} W, W \sim Brownian)
        lambda_1: denotes strength of L2 reg.
    """
    n, d = X.shape
    N = len(particles)
    # Compute network output for each data point
    f_preds = cp.zeros(n)
    for i in range(n):
        f_preds[i] = cp.mean(cp.array([neuron_activation(x, X[i], R_bar) for x in particles]))
    # Compute derivative of the loss for each data point
    loss_derivs = cp.array([d_logistic_loss(f_preds[i], y[i]) for i in range(n)])
    
    new_particles = []
    for x in particles:
        grad_loss = cp.zeros_like(x)
        # Accumulate gradient contributions from all training data
        for i in range(n):
            grad_loss += (loss_derivs[i] * y[i]) * neuron_grad(x, X[i], R_bar)
        grad_loss = grad_loss / n
        grad_reg = 2 * lambda_1 * x
        grad = grad_loss + grad_reg
        noise = cp.random.randn(*x.shape)
        x_new = x - eta * grad + cp.sqrt(2 * lam * eta) * noise
        new_particles.append(x_new)
    return new_particles

def run_simulation(d=20, n_train=500, n_test=200, 
                   num_particles=1000, eta=0.05, T_per_round=200, 
                   num_rounds=6, lam_init=0.1, R_bar=15):
    # Generate training and test data
    X_train, y_train = generate_xor_data(n_train, d)
    X_test, y_test = generate_xor_data(n_test, d)
    # Initialize particles randomly
    particles = [cp.random.randn(d+2) for _ in range(num_particles)]
    
    accuracy_history = []
    lam_history = []
    
    lam_current = lam_init
    lambda_1_value = 0.1 # Suzuki et al., 2023
    for round in range(num_rounds):
        print(f"Annealing Round {round+1}, lambda = {lam_current:.5f}")
        for t in range(T_per_round):
            particles = update_particles(particles, X_train, y_train, eta, lam_current, lambda_1_value, R_bar)
        acc = evaluate_accuracy(particles, X_test, y_test, R_bar)
        print(f"Test accuracy after round {round+1}: {acc:.3f}")
        accuracy_history.append(acc)
        lam_history.append(lam_current)
        lam_current *= 0.5  # Anneal: reduce lambda by factor of 2.
    return accuracy_history, lam_history

def run_simulation_and_store(d=20, n_train=500, n_test=200,
                             num_particles=1000, eta=0.05, T_per_round=200,
                             num_rounds=6, lam_init=0.1, lambda_1=0.1, R_bar=15,
                             save_dir="/home/ddz5/Desktop/sds659/results/mfld_results"):

    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved in: {save_dir}")

    X_train, y_train = generate_xor_data(n_train, d)
    X_test, y_test = generate_xor_data(n_test, d)
    particles = [cp.random.randn(d+2, dtype=cp.float32) for _ in range(num_particles)]

    particle_history = {} # Dictionary to store particle states
    accuracy_history = []
    lam_history = []

    print("Storing initial particle state...")
    initial_particle_matrix_cp = cp.stack(particles)
    np.save(os.path.join(save_dir, 'particles_round_0.npy'), cp.asnumpy(initial_particle_matrix_cp))
    # for plotting
    particle_history[0] = [p.copy() for p in particles]

    lam_current = lam_init
    for round_num in range(num_rounds):
        round_actual = round_num + 1
        print(f"Annealing Round {round_actual}, lambda = {lam_current:.5f}")
        for t in range(T_per_round):
            particles = update_particles(particles, X_train, y_train, eta, lam_current, lambda_1, R_bar)
            if (t+1) % 25 == 0:
                 print(f"  Round {round_actual}, Timestep: {t+1} / {T_per_round}")

        print(f"Storing particle state after round {round_actual}...")
        current_particle_matrix_cp = cp.stack(particles)
        np.save(os.path.join(save_dir, f'particles_round_{round_actual}.npy'), cp.asnumpy(current_particle_matrix_cp))
        # Store in memory for plotting
        particle_history[round_actual] = [p.copy() for p in particles]

        acc = evaluate_accuracy(particles, X_test, y_test, R_bar)
        # Result should be float now, no need for cp.asnumpy
        print(f"Test accuracy after round {round_actual}: {acc:.4f}")
        accuracy_history.append(acc)
        lam_history.append(lam_current)

        lam_current *= 0.5

    # --- Save Accuracy and Lambda History ---
    np.save(os.path.join(save_dir, 'accuracy_history.npy'), np.array(accuracy_history))
    np.save(os.path.join(save_dir, 'lambda_history.npy'), np.array(lam_history))
    print("Saved accuracy and lambda history.")

    return particle_history, accuracy_history, lam_history

cp.random.seed(42)
SAVE_DIR = "/home/ddz5/Desktop/sds659/results/mfld_results_large"

# Run simulation and store results
particle_history, accuracy_history, lam_history = run_simulation_and_store(
    d=20, n_train=500, n_test=200,
    num_particles=1000,
    eta=0.05, T_per_round=50,
    num_rounds=6, lam_init=0.1,
    lambda_1=0.1,
    R_bar=15,
    save_dir=SAVE_DIR # Pass the directory
)

print("\n--- Generating and Saving Plots ---")

# Plot 1: Test Accuracy Evolution
plt.figure(figsize=(8, 6))
# Use numpy arange for plotting if accuracy_history is a standard list
plt.plot(np.arange(1, len(accuracy_history) + 1), accuracy_history, marker='o')
plt.xlabel('Annealing Round')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy over Annealing Rounds')
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, 'accuracy_vs_round.png')) 
plt.show()

# Plot 2: Lambda Evolution
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(lam_history) + 1), lam_history, marker='o', color='r')
plt.xlabel('Annealing Round')
plt.ylabel('Lambda (Noise Scale)')
plt.title('Lambda Parameter over Annealing Rounds')
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, 'lambda_vs_round.png'))
plt.show()

# Plot 3: PCA Visualization
plt.figure(figsize=(10, 8))
num_rounds_to_plot = len(particle_history)
colors = plt.cm.viridis(np.linspace(0, 1, num_rounds_to_plot))
pca = PCA(n_components=2)
scaler = StandardScaler()

for round_num, particles_cp_list in particle_history.items():
    print(f"Processing PCA for round {round_num}")
    particle_matrix_cp = cp.stack(particles_cp_list)
    particle_matrix_np = cp.asnumpy(particle_matrix_cp)
    particle_matrix_scaled = scaler.fit_transform(particle_matrix_np)
    # Fit PCA and transform data for the current round
    particles_pca = pca.fit_transform(particle_matrix_scaled) # Fit and transform per round
    plt.scatter(particles_pca[:, 0], particles_pca[:, 1],
                alpha=0.6, label=f'Round {round_num}', color=colors[round_num], s=10)

plt.title('PCA of MFLD Particle Parameters Over Annealing Rounds')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, 'pca_particles_over_rounds.png')) # Save the plot
plt.show() # Display plot

# Plot 4: Weight Component Histograms
rounds_to_compare = [0, len(particle_history) - 1]
d_value = 20 # Make sure this matches the d used in simulation
num_bins = 50
fig, axes = plt.subplots(1, len(rounds_to_compare), figsize=(12, 5), sharey=True)
fig.suptitle('Distribution of Relevant vs. Irrelevant Weight Components')

for i, round_num in enumerate(rounds_to_compare):
    print(f"Processing weight distributions for round {round_num}")
    ax = axes[i]
    particles_cp_list = particle_history[round_num]
    particle_matrix_cp = cp.stack(particles_cp_list)
    weights_cp = particle_matrix_cp[:, :d_value]
    relevant_weights_cp = weights_cp[:, :2].flatten()
    irrelevant_weights_cp = weights_cp[:, 2:].flatten()
    relevant_weights_np = cp.asnumpy(relevant_weights_cp)
    irrelevant_weights_np = cp.asnumpy(irrelevant_weights_cp)

    # Use numpy for histogram calculation if needed
    bin_min = np.min([relevant_weights_np.min(), irrelevant_weights_np.min()])
    bin_max = np.max([relevant_weights_np.max(), irrelevant_weights_np.max()])
    bins = np.linspace(bin_min, bin_max, num_bins)

    ax.hist(relevant_weights_np, bins=bins, alpha=0.7, label='Relevant Weights (Dims 1-2)', density=True)
    ax.hist(irrelevant_weights_np, bins=bins, alpha=0.7, label=f'Irrelevant Weights (Dims 3-{d_value})', density=True)
    ax.set_title(f'Round {round_num}')
    ax.set_xlabel('Weight Value')
    if i == 0:
        ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(SAVE_DIR, 'weight_distribution_comparison.png'))
plt.show()

# Plot 5: Relevant Weight Scatter Plot
plt.figure(figsize=(10, 8))
for round_num, particles_cp_list in particle_history.items():
    print(f"Processing scatter plot for round {round_num}")
    particle_matrix_cp = cp.stack(particles_cp_list)
    weights_cp = particle_matrix_cp[:, :d_value]
    relevant_weights_cp = weights_cp[:, :2]
    relevant_weights_np = cp.asnumpy(relevant_weights_cp)
    plt.scatter(relevant_weights_np[:, 0], relevant_weights_np[:, 1],
                alpha=0.3, label=f'Round {round_num}', color=colors[round_num], s=10)

plt.title('Scatter Plot of First Two Weight Components Over Annealing Rounds')
plt.xlabel('Weight Component 1')
plt.ylabel('Weight Component 2')
plt.grid(True)
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.savefig(os.path.join(SAVE_DIR, 'relevant_weights_scatter.png'))
plt.show()

print(f"\nAll plots saved to {SAVE_DIR}")
print(f"Particle states saved as .npy files in {SAVE_DIR}")