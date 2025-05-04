# Filtering Implementations

This directory contains various implementations of string filtering operations optimized for performance using different Python libraries and techniques.

## Implementations

1. **Baseline Pandas Filtering**
   - Pure Pandas implementation
   - Basic string length filtering
   - Serves as a reference point for performance comparison

2. **Optimized Pandas Filtering**
   - Enhanced Pandas implementation
   - Pre-converts column to string type
   - More efficient than baseline

3. **NumPy Filtering**
   - Uses NumPy's vectorized operations
   - Converts data to NumPy arrays for faster processing
   - Efficient for large datasets

4. **Dask Filtering**
   - Parallel processing implementation
   - Splits data into partitions
   - Ideal for very large datasets that don't fit in memory

5. **Numba Filtering**
   - Uses Numba JIT compilation
   - Optimized for CPU performance
   - Fast execution for medium-sized datasets

6. **Cython Filtering**
   - Compiled Cython implementation
   - Native C performance
   - Fastest implementation for most cases

## Usage

```python
from filtering import test_filtering_all_versions

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# Test all implementations
test_filtering_all_versions(df, "column_name", threshold=15)
```

## Performance Considerations

- For small to medium datasets: Use Numba or Cython implementations
- For large datasets: Use Dask implementation
- For quick prototyping: Use Pandas implementations
- For maximum performance: Use Cython implementation

## Dependencies

- pandas
- numpy
- dask
- numba
- cython 