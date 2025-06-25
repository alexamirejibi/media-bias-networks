import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# set style
plt.style.use('default')
sns.set_palette("husl")

# parameters
n_outlets = 49  # total number of outlets
cluster_sizes = np.arange(1, 51)  # cluster sizes from 1 to 50

# calculate weights for both methods
probabilities = cluster_sizes / n_outlets
surprisal_weights = -np.log2(probabilities)
inverse_prob_weights = 1 / probabilities

# create the comparison visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# plot 1: surprisal vs cluster size
ax1.plot(cluster_sizes, surprisal_weights, 'o-', linewidth=2, markersize=3, color='#2E86AB', label='Surprisal')
ax1.set_xlabel('Cluster Size (||c||)', fontweight='bold')
ax1.set_ylabel('Surprisal Weight (bits)', fontweight='bold')
ax1.set_title('Surprisal Weighting\n$s = -\\log_2(||c||/n)$', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 50)

# highlight key points for surprisal
key_sizes = [2, 5, 10, 25, 49]
key_surprisal = -np.log2(np.array(key_sizes) / n_outlets)
ax1.scatter(key_sizes, key_surprisal, color='#C73E1D', s=40, zorder=5)

# plot 2: inverse probability vs cluster size
ax2.plot(cluster_sizes, inverse_prob_weights, 'o-', linewidth=2, markersize=3, color='#F18F01', label='Inverse Prob')
ax2.set_xlabel('Cluster Size (||c||)', fontweight='bold')
ax2.set_ylabel('Inverse Probability Weight', fontweight='bold')
ax2.set_title('Inverse Probability Weighting\n$w = n/||c||$', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 50)
ax2.set_ylim(0, 50)  # limit y-axis for better visibility

# highlight key points for inverse probability
key_inverse = n_outlets / np.array(key_sizes)
ax2.scatter(key_sizes, key_inverse, color='#C73E1D', s=40, zorder=5)

# plot 3: direct comparison (log scale for inverse prob)
ax3.plot(cluster_sizes, surprisal_weights, 'o-', linewidth=2, markersize=3, color='#2E86AB', label='Surprisal')
ax3.plot(cluster_sizes, inverse_prob_weights, 'o-', linewidth=2, markersize=3, color='#F18F01', label='Inverse Prob')
ax3.set_xlabel('Cluster Size (||c||)', fontweight='bold')
ax3.set_ylabel('Weight', fontweight='bold')
ax3.set_title('Direct Comparison\n(Linear Scale)', fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_xlim(0, 50)
ax3.set_ylim(0, 15)  # focus on reasonable range

# plot 4: log scale comparison
ax4.semilogy(cluster_sizes, surprisal_weights, 'o-', linewidth=2, markersize=3, color='#2E86AB', label='Surprisal')
ax4.semilogy(cluster_sizes, inverse_prob_weights, 'o-', linewidth=2, markersize=3, color='#F18F01', label='Inverse Prob')
ax4.set_xlabel('Cluster Size (||c||)', fontweight='bold')
ax4.set_ylabel('Weight (log scale)', fontweight='bold')
ax4.set_title('Direct Comparison\n(Log Scale)', fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.set_xlim(0, 50)

plt.tight_layout()
plt.savefig('results/surprisal_vs_inverse_probability.png', dpi=300, bbox_inches='tight')
plt.show()

# print comparison table
print("Comparison of Surprisal vs Inverse Probability Weighting:")
print("Cluster Size | Probability | Surprisal (bits) | Inverse Prob | Ratio (Inv/Sur)")
print("-" * 75)
for size in [2, 5, 10, 15, 20, 25, 30, 40, 49]:
    prob = size / n_outlets
    surprisal = -np.log2(prob)
    inverse = 1 / prob
    ratio = inverse / surprisal
    print(f"{size:11d} | {prob:11.3f} | {surprisal:13.2f} | {inverse:11.2f} | {ratio:11.2f}")

# demonstrate the extreme weight problem with inverse probability
print(f"\nExtreme Weight Analysis:")
print(f"For smallest meaningful cluster (size=2):")
print(f"  Surprisal weight: {-np.log2(2/49):.2f} bits")
print(f"  Inverse prob weight: {49/2:.1f}")
print(f"  Ratio: {(49/2)/(-np.log2(2/49)):.1f}x larger!")

print(f"\nFor moderate cluster (size=10):")
print(f"  Surprisal weight: {-np.log2(10/49):.2f} bits") 
print(f"  Inverse prob weight: {49/10:.1f}")
print(f"  Ratio: {(49/10)/(-np.log2(10/49)):.1f}x larger") 