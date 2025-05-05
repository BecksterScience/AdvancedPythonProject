# Advanced Python Data Processing Toolkit

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

### Grouping
- Efficient data grouping operations
- Memory optimization techniques (downcasting, categorical conversion)
- Performance benchmarking
- Visualization of results
- Memory reduction strategies (up to 53% reduction)

### Searching
- Fast string searching implementations
- Case-insensitive search with na-safe handling
- Memory-efficient processing
- Performance-optimized versions (Cython ~0.4s, NumPy ~1.0s, Pandas ~1.2s)
- Support for flexible user queries

### K-Means
- Text data clustering with TF-IDF vectorization
- Multiple K-Means implementations (Python, NumPy, Numba, scikit-learn)
- Interactive visualizations with PCA
- Word cloud generation
- Performance comparison (scikit-learn 875.6× speedup over baseline)

### Sorting
- Multiple sorting algorithm implementations (Bubble, Quick, Merge, Heap, Selection, Tim)
- Optimized versions using Numba and Cython
- Performance comparison tools
- Visualization of sorting runtimes
- Support for both ascending and descending order
- Benchmark results for various implementations

### LSH Clustering
- MinHash + LSH implementation with parallel processing (~245s for 22M pairs)
- Cython-optimized similarity detection
- TF-IDF + KMeans clustering
- PCA visualizations
- WordCloud generation
- Interactive menu for easy access to all features
- Support for Jaccard similarity approximation

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the Python community for the excellent libraries used in this project
- Special thanks to the contributors of pandas, numpy, numba, and cython
- Project developed by Evan Beck, Samarth Agarwal, Aanand Krishnakumar, Akshitha Kumbam, Varshitha Reddy Medarametla, and Rithujaa Rajendrakumar