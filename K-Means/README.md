# K-Means Clustering Implementations

This directory contains various implementations of K-Means clustering algorithm optimized for text data using different Python libraries and techniques.

## Features

- Text data clustering using TF-IDF vectorization
- Multiple K-Means implementations
- 2D visualization using PCA
- Word cloud generation for each cluster
- Performance benchmarking
- Interactive visualizations

## Implementations

1. **Pure Python K-Means**
   - Basic implementation using Python loops
   - Uses itertools.product for assignment
   - Reference implementation
   - Good for understanding the algorithm

2. **NumPy Vectorized K-Means**
   - Vectorized operations using NumPy
   - Faster than pure Python
   - Memory efficient
   - Uses broadcasting for distance calculations

3. **Numba JIT K-Means**
   - JIT-compiled implementation
   - Optimized for CPU performance
   - Fast execution
   - Custom distance calculations

4. **scikit-learn K-Means**
   - Industry-standard implementation
   - Most optimized version
   - Production-ready
   - Includes additional features

## Visualization Features

1. **Runtime Comparison**
   - Bar chart comparing execution times
   - Visual performance comparison
   - Easy to interpret results

2. **PCA Scatter Plot**
   - 2D projection of clusters
   - Color-coded clusters
   - Interactive legend
   - Clear cluster separation visualization

3. **Word Clouds**
   - Per-cluster word clouds
   - Visual representation of cluster topics
   - Easy to understand cluster contents
   - Customizable appearance

## Usage

```python
# Configuration
DATA_PATH = "your_dataset.csv"
SAMPLE_N = 50_000  # Number of samples
MAX_FEAT = 1_000   # Maximum TF-IDF features
MAX_ITERS = 10     # Maximum iterations
CLUSTERS = 10      # Number of clusters

# Run the clustering
main()
```

## Performance Considerations

- For small datasets: Use any implementation
- For medium datasets: Use NumPy or Numba
- For large datasets: Use scikit-learn
- For educational purposes: Use Pure Python

## Dependencies

- pandas
- numpy
- numba
- scikit-learn
- matplotlib
- wordcloud
- PCA from scikit-learn
- TF-IDF vectorizer from scikit-learn 