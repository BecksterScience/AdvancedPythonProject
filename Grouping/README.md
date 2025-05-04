# Grouping and Memory Optimization

This directory contains implementations of data grouping operations and memory optimization techniques using various Python libraries and approaches.

## Grouping Implementations

1. **Baseline Pandas Grouping**
   - Standard Pandas groupby operation
   - Basic implementation for reference

2. **Optimized Pandas Grouping**
   - Uses value_counts() for faster grouping
   - More efficient than baseline

3. **Dask Grouping**
   - Parallel processing implementation
   - Handles large datasets efficiently
   - Memory-friendly for big data

4. **Numba Grouping**
   - JIT-compiled implementation
   - Optimized for CPU performance
   - Fast execution for medium datasets

5. **Cython Grouping**
   - Compiled Cython implementation
   - Native C performance
   - Fastest implementation

## Memory Optimization Techniques

1. **Baseline Pandas Memory**
   - No memory optimization
   - Reference point for comparison

2. **Optimized Pandas Memory**
   - Downcasts numeric types
   - Reduces memory usage significantly
   - Maintains data integrity

3. **NumPy Memory Optimization**
   - Uses NumPy's efficient data types
   - Optimizes numeric columns
   - Good balance of performance and memory usage

4. **Dask Memory Optimization**
   - Distributed memory management
   - Handles out-of-memory datasets
   - Best for very large datasets

5. **Numba Memory Optimization**
   - JIT-compiled memory operations
   - Efficient in-memory processing
   - Good for medium datasets

6. **Cython Memory Optimization**
   - Compiled memory operations
   - Maximum performance
   - Best for critical memory operations

## Usage

```python
from grouping import test_grouping_all_versions, test_memory_reduction_all_versions

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# Test grouping implementations
test_grouping_all_versions(df, "column_name")

# Test memory optimization
test_memory_reduction_all_versions(df)
```

## Performance Considerations

- For small to medium datasets: Use Numba or Cython implementations
- For large datasets: Use Dask implementation
- For memory optimization: Use Optimized Pandas or Cython
- For quick prototyping: Use Pandas implementations

## Dependencies

- pandas
- numpy
- dask
- numba
- cython
- matplotlib (for visualization) 