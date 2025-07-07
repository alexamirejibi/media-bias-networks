# Unsupervised modeling of media bias groups with community detection

Please see the full_analysis.ipynb file for the complete experiment.

## Code guide (src)
- ★ experiment.py: main experiment framework
- ★ methods.py: contains all community detection network modelling methods; uses a plugin system
- data.py: data management utils
- plugins.py: actual plugin management code
- networks.py: main code for building adjacencies and running community detection methods from methods.py (using the plugin system)
- partition_metrics.py: parition metrics (NMI, ARI, confusion matrix calculation)
- viz.py: misc visualization code

Analysis files (src/analysis):
- core.py: main orchestration file (ResultsAnalyzer class is maintained for backwards-compatibility with pre-refactor version; this class was previously used for all analysis before being broken down into the different files seen now in the src/analysis directory)
- ★ statistics.py: main statistical significance testing code
- stability.py: stability metrics
- temporal.py: between-sample temporal analysis (temporal stability etc.)
- clustering.py: hierarchical clustering, community detection, and consensus building analysis; however, the final statistically validated clustering construction happens in statistics.py