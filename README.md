# Advanced Python! Processing Strings with Big Data

This repository contains a collection of optimized data processing implementations in Python, focusing on performance, memory efficiency, and scalability. The toolkit was developed to handle large-scale string datasets, particularly focusing on Amazon product titles, with over 1.4 million entries. Each module provides multiple implementations of common data operations using different Python libraries and optimization techniques.

## Directory Structure

- `Filtering/` - String filtering implementations with dynamic UI and multiple backend options
- `Grouping/` - Data grouping and memory optimization techniques
- `Searching/` - String searching implementations with case-insensitive support
- `K-Means/` - Text clustering with visualization and performance analysis
- `Sorting/` - Multiple sorting algorithm implementations with various optimizations
- `LSH_Clustering/` - Product title similarity detection and clustering using MinHash and LSH

## Features

### Filtering
- Multiple string filtering implementations (Cython, Pandas, NumPy, Dask)
- Performance-optimized versions (Cython ~0.15s, Pandas ~0.23s, NumPy ~0.25s)
- Memory-efficient processing with categorical encoding
- Dynamic filtering UI with sliders and dropdowns
- Support for numeric ranges, categorical selections, and keyword searches
- Implementation Details:
  - Cython provides the best performance by compiling filtering logic into C
  - Pandas and NumPy offer vectorized operations for efficient processing
  - Dask handles larger-than-memory datasets through partitioning
  - Numba is not suitable for string filtering due to lack of UTF-8 support

### Grouping
- Efficient data grouping operations
- Memory optimization techniques (downcasting, categorical conversion)
- Performance benchmarking
- Visualization of results
- Memory reduction strategies (up to 53% reduction)
- Implementation Details:
  - Downcasting numerics using pd.to_numeric(..., downcast=...)
  - Converting low-cardinality object columns to category dtype
  - String optimization through astype("category")
  - Memory reduction from 580MB to 270MB through optimization

### Searching
- Fast string searching implementations
- Case-insensitive search with na-safe handling
- Memory-efficient processing
- Performance-optimized versions (Cython ~0.4s, NumPy ~1.0s, Pandas ~1.2s)
- Support for flexible user queries
- Implementation Details:
  - Cython provides ~3× speedup over Pandas through loop compilation
  - NumPy offers better performance than Pandas but with array overhead
  - Dask's lazy computation and task graph overhead affects performance
  - Numba JIT compilation not effective for object/string logic

### K-Means
- Text data clustering with TF-IDF vectorization
- Multiple K-Means implementations (Python, NumPy, Numba, scikit-learn)
- Interactive visualizations with PCA
- Word cloud generation
- Performance comparison (scikit-learn 875.6× speedup over baseline)
- Implementation Details:
  - TF-IDF vectorization with 1,000 term vocabulary
  - Stop word removal for meaningful content focus
  - Pure Python baseline for educational insight
  - NumPy vectorized operations for faster distance calculations
  - Numba JIT-compiled version for performance optimization
  - Scikit-learn as the benchmark implementation

### Sorting
- Multiple sorting algorithm implementations (Bubble, Quick, Merge, Heap, Selection, Tim)
- Optimized versions using Numba and Cython
- Performance comparison tools
- Visualization of sorting runtimes
- Support for both ascending and descending order
- Benchmark results for various implementations
- Implementation Details:
  - Quick Sort: Python (0.666s), NumPy (0.008s), Cython (0.941s), Numba (0.033s)
  - Merge Sort: Python (0.862s), NumPy (0.009s), Cython (0.912s), Numba (0.014s)
  - Heap Sort: Python (1.712s), NumPy (0.013s), Cython (1.099s), Numba (0.037s)
  - Timsort: Python (0.235s), NumPy (0.009s), Cython (0.037s), Numba (0.082s)
  - Selection Sort: Terminated in Python, NumPy (1.740s), Terminated in Cython, Numba (5.285s)

### LSH Clustering
- MinHash + LSH implementation with parallel processing (~245s for 22M pairs)
- Cython-optimized similarity detection
- TF-IDF + KMeans clustering
- PCA visualizations
- WordCloud generation
- Interactive menu for easy access to all features
- Support for Jaccard similarity approximation
- Implementation Details:
  - MinHash for probabilistic hashing and fixed-length signatures
  - LSH for clustering similar signatures into buckets
  - Parallelized MinHash signature generation across 12 CPU cores
  - TF-IDF vectorization with 5,000 term vocabulary
  - KMeans clustering with k=10 clusters
  - PCA for dimensionality reduction and visualization
  - Word count analysis for cluster interpretation

## Getting Started

1. Clone the repository:
```bash
git clone git@github.com:Akshi22/adv_python_final.git
cd adv_python_final
```

2. Set up a virtual environment:
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Explore each module's README for specific usage instructions.

## Development Environment

### Recommended IDE Setup
- VS Code with Python extension
- Jupyter Notebook for interactive development
- IPython for interactive Python shell

### Virtual Environment Management
To manage your virtual environment:

```bash
# Deactivate the virtual environment when done
deactivate

# To remove the virtual environment
# On Windows:
rmdir /s /q venv
# On macOS/Linux:
rm -rf venv
```

## Dependencies

Common dependencies across modules:
- pandas
- numpy
- numba
- cython
- matplotlib
- scikit-learn
- datasketch (for LSH)
- seaborn

Additional module-specific dependencies are listed in each module's README.

## Performance Considerations

Each module provides multiple implementations optimized for different scenarios:
- Small datasets: Use Numba or Cython implementations
- Large datasets: Use Dask implementations or LSH pre-filtering
- Quick prototyping: Use Pandas implementations
- Maximum performance: Use Cython implementations
- String-heavy tasks: Prefer Cython over Numba
- Memory optimization: Use categorical encoding and downcasting
- Clustering: Use LSH pre-filtering for large datasets (>1M entries)
- Sorting: Use NumPy for native implementations, Numba for custom algorithms
- Filtering: Use Cython for maximum performance, Pandas for quick development
- Searching: Use Cython for tight loop performance, avoid Dask for small tasks

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the Python community for the excellent libraries used in this project
- Special thanks to the contributors of pandas, numpy, numba, and cython
- Project developed by Evan Beck, Samarth Agarwal, Aanand Krishnakumar, Akshitha Kumbam, Varshitha Reddy Medarametla, and Rithujaa Rajendrakumar
