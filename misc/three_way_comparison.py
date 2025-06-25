import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# set style
plt.style.use('default')
sns.set_palette("husl")

# parameters
n_outlets = 49  # total number of outlets
cluster_sizes = np.arange(1, 51)  # cluster sizes from 1 to 50

# calculate weights for all three methods
probabilities = cluster_sizes / n_outlets
surprisal_weights = -np.log2(probabilities)
inverse_prob_weights = 1 / probabilities
complement_weights = 1 - probabilities

# create the three-way comparison visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# plot 1: all three methods on linear scale
ax1.plot(cluster_sizes, surprisal_weights, 'o-', linewidth=2, markersize=3, color='#2E86AB', label='Surprisal')
ax1.plot(cluster_sizes, complement_weights, 's-', linewidth=2, markersize=3, color='#A23B72', label='Complement (1-p)')
ax1.plot(cluster_sizes, inverse_prob_weights, '^-', linewidth=2, markersize=3, color='#F18F01', label='Inverse Prob')
ax1.set_xlabel('Cluster Size (||c||)', fontweight='bold')
ax1.set_ylabel('Weight', fontweight='bold')
ax1.set_title('Comparison of Weighting Schemes', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(0, 50)
ax1.set_ylim(0, 12)  # focus on reasonable range

# plot 2: log scale comparison
ax2.semilogy(cluster_sizes, surprisal_weights, 'o-', linewidth=2, markersize=3, color='#2E86AB', label='Surprisal')
ax2.semilogy(cluster_sizes, complement_weights, 's-', linewidth=2, markersize=3, color='#A23B72', label='Complement (1-p)')
ax2.semilogy(cluster_sizes, inverse_prob_weights, '^-', linewidth=2, markersize=3, color='#F18F01', label='Inverse Prob')
ax2.set_xlabel('Cluster Size (||c||)', fontweight='bold')
ax2.set_ylabel('Weight (log scale)', fontweight='bold')
ax2.set_title('Weighting Schemes (Log Scale)', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xlim(0, 50)

# plot 3: focus on small clusters (where differences matter most)
small_sizes = cluster_sizes[:20]  # first 20 cluster sizes
ax3.plot(small_sizes, surprisal_weights[:20], 'o-', linewidth=2, markersize=4, color='#2E86AB', label='Surprisal')
ax3.plot(small_sizes, complement_weights[:20], 's-', linewidth=2, markersize=4, color='#A23B72', label='Complement (1-p)')
ax3.plot(small_sizes, inverse_prob_weights[:20], '^-', linewidth=2, markersize=4, color='#F18F01', label='Inverse Prob')
ax3.set_xlabel('Cluster Size (||c||)', fontweight='bold')
ax3.set_ylabel('Weight', fontweight='bold')
ax3.set_title('Small Cluster Discrimination', fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_xlim(0, 20)
ax3.set_ylim(0, 8)

# plot 4: normalized comparison (all scaled to 0-1 range)
norm_surprisal = surprisal_weights / surprisal_weights.max()
norm_complement = complement_weights / complement_weights.max()
norm_inverse = inverse_prob_weights / inverse_prob_weights.max()

ax4.plot(cluster_sizes, norm_surprisal, 'o-', linewidth=2, markersize=3, color='#2E86AB', label='Surprisal (norm)')
ax4.plot(cluster_sizes, norm_complement, 's-', linewidth=2, markersize=3, color='#A23B72', label='Complement (norm)')
ax4.plot(cluster_sizes, norm_inverse, '^-', linewidth=2, markersize=3, color='#F18F01', label='Inverse Prob (norm)')
ax4.set_xlabel('Cluster Size (||c||)', fontweight='bold')
ax4.set_ylabel('Normalized Weight (0-1)', fontweight='bold')
ax4.set_title('Normalized Weighting Functions', fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.set_xlim(0, 50)
ax4.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('results/three_way_weighting_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# print detailed comparison table
print("Three-Way Comparison: Surprisal vs Complement vs Inverse Probability")
print("Cluster Size | Probability | Surprisal | Complement | Inverse | Sur/Comp | Inv/Sur")
print("-" * 85)
for size in [2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 49]:
    prob = size / n_outlets
    surprisal = -np.log2(prob)
    complement = 1 - prob
    inverse = 1 / prob
    sur_comp_ratio = surprisal / complement if complement > 0 else np.inf
    inv_sur_ratio = inverse / surprisal if surprisal > 0 else np.inf
    print(f"{size:11d} | {prob:11.3f} | {surprisal:9.2f} | {complement:10.3f} | {inverse:7.1f} | {sur_comp_ratio:8.2f} | {inv_sur_ratio:7.2f}")

# analyze dynamic range and discrimination
print(f"\nDynamic Range Analysis:")
print(f"Method          | Min Weight | Max Weight | Range  | Ratio (Max/Min)")
print(f"----------------|------------|------------|--------|---------------")
print(f"Surprisal       | {surprisal_weights[48]:.3f}      | {surprisal_weights[0]:.3f}      | {surprisal_weights[0]-surprisal_weights[48]:.3f}  | {surprisal_weights[0]/max(surprisal_weights[48], 0.001):.1f}")
print(f"Complement      | {complement_weights[48]:.3f}      | {complement_weights[0]:.3f}      | {complement_weights[0]-complement_weights[48]:.3f}  | {complement_weights[0]/max(complement_weights[48], 0.001):.1f}")
print(f"Inverse Prob    | {inverse_prob_weights[48]:.3f}      | {inverse_prob_weights[0]:.1f}     | {inverse_prob_weights[0]-inverse_prob_weights[48]:.1f} | {inverse_prob_weights[0]/inverse_prob_weights[48]:.1f}")

# analyze discrimination in critical small cluster region
print(f"\nDiscrimination in Small Cluster Region (sizes 2-5):")
small_surprisal = surprisal_weights[1:5]  # sizes 2-5 (indices 1-4)
small_complement = complement_weights[1:5]
small_inverse = inverse_prob_weights[1:5]

print(f"Surprisal range (2-5):    {small_surprisal[0]:.2f} to {small_surprisal[3]:.2f} (spread: {small_surprisal[0]-small_surprisal[3]:.2f})")
print(f"Complement range (2-5):   {small_complement[0]:.3f} to {small_complement[3]:.3f} (spread: {small_complement[0]-small_complement[3]:.3f})")
print(f"Inverse range (2-5):      {small_inverse[0]:.1f} to {small_inverse[3]:.1f} (spread: {small_inverse[0]-small_inverse[3]:.1f})")

print(f"\nConclusion:")
print(f"- Complement has the smallest dynamic range and poorest discrimination")
print(f"- Surprisal provides good discrimination with moderate, stable weights")
print(f"- Inverse probability has extreme weights that could dominate analysis") 