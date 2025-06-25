import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# set style
plt.style.use('default')
sns.set_palette("husl")

# parameters
n_outlets = 49  # total number of outlets
cluster_sizes = np.arange(1, 51)  # cluster sizes from 1 to 50

# calculate surprisal weights
surprisal_weights = -np.log2(cluster_sizes / n_outlets)

# create the visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# plot 1: surprisal vs cluster size
ax1.plot(cluster_sizes, surprisal_weights, 'o-', linewidth=2, markersize=4, color='#2E86AB')
ax1.set_xlabel('Cluster Size (||c||)', fontweight='bold')
ax1.set_ylabel('Surprisal Weight (bits)', fontweight='bold')
ax1.set_title('Surprisal Weight vs Cluster Size\n$s = -\\log_2(||c||/n)$, $n=49$', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 50)

# highlight some key points
key_sizes = [2, 5, 10, 25, 49]
key_weights = -np.log2(np.array(key_sizes) / n_outlets)
ax1.scatter(key_sizes, key_weights, color='#C73E1D', s=60, zorder=5)

# add annotations for key points
for size, weight in zip(key_sizes, key_weights):
    ax1.annotate(f'||c||={size}\ns={weight:.1f}', 
                xy=(size, weight), 
                xytext=(10, 10), 
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                fontsize=8)

# plot 2: probability vs cluster size (for context)
probabilities = cluster_sizes / n_outlets
ax2.plot(cluster_sizes, probabilities, 'o-', linewidth=2, markersize=4, color='#F18F01')
ax2.set_xlabel('Cluster Size (||c||)', fontweight='bold')
ax2.set_ylabel('Probability (||c||/n)', fontweight='bold')
ax2.set_title('Cluster Probability vs Size\n$P = ||c||/n$, $n=49$', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 50)
ax2.set_ylim(0, 1.1)

# highlight the same key points
key_probs = np.array(key_sizes) / n_outlets
ax2.scatter(key_sizes, key_probs, color='#C73E1D', s=60, zorder=5)

# add annotations
for size, prob in zip(key_sizes, key_probs):
    ax2.annotate(f'||c||={size}\nP={prob:.2f}', 
                xy=(size, prob), 
                xytext=(10, 10), 
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                fontsize=8)

plt.tight_layout()
plt.savefig('results/surprisal_vs_cluster_size.png', dpi=300, bbox_inches='tight')
plt.show()

# print some key values
print("Surprisal weights for key cluster sizes:")
print("Cluster Size | Probability | Surprisal (bits)")
print("-" * 40)
for size in [2, 5, 10, 15, 20, 25, 30, 40, 49]:
    prob = size / n_outlets
    weight = -np.log2(prob)
    print(f"{size:11d} | {prob:11.3f} | {weight:13.2f}") 