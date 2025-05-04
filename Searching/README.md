# String Searching Implementations

This directory contains various implementations of string searching operations optimized for performance using different Python libraries and techniques.

## Implementations

1. **Baseline Pandas Searching**
   - Pure Pandas implementation
   - Uses str.contains() method
   - Case-insensitive search
   - Reference implementation

2. **Optimized Pandas Searching**
   - Enhanced Pandas implementation
   - Pre-converts column to string type
   - More efficient than baseline

3. **NumPy Searching**
   - Uses NumPy's vectorized operations
   - Converts data to NumPy arrays
   - Efficient for large datasets
   - Uses np.char.find for fast searching

4. **Dask Searching**
   - Parallel processing implementation
   - Splits data into partitions
   - Ideal for very large datasets
   - Memory-efficient

5. **Numba Searching**
   - Uses Numba JIT compilation
   - Optimized for CPU performance
   - Fast execution for medium datasets
   - Custom search kernel

6. **Cython Searching**
   - Compiled Cython implementation
   - Native C performance
   - Fastest implementation
   - Custom string search algorithm

## Usage

```python
from searching import test_searching_all_versions

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# Test all implementations
test_searching_all_versions(df, "column_name", "search_term")
```

## Performance Considerations

- For small to medium datasets: Use Numba or Cython implementations
- For large datasets: Use Dask implementation
- For quick prototyping: Use Pandas implementations
- For maximum performance: Use Cython implementation

## Features

- Case-insensitive search
- Handles missing values (NA)
- Memory-efficient implementations
- Parallel processing support
- Performance benchmarking included

## Dependencies

- pandas
- numpy
- dask
- numba
- cython 