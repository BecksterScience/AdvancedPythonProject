# Sorting Module

This module provides various implementations of sorting algorithms optimized for different use cases and performance requirements. It includes both pure Python implementations and optimized versions using Numba and Cython.

## Available Sorting Algorithms

- Bubble Sort
- Quick Sort
- Merge Sort
- Heap Sort
- Selection Sort
- Tim Sort

## Implementations

Each sorting algorithm is implemented in multiple ways to provide different performance characteristics:

1. **Python Implementation**: Basic implementation using pure Python
2. **NumPy Implementation**: Vectorized implementation using NumPy
3. **Numba Implementation**: JIT-compiled implementation using Numba
4. **Cython Implementation**: Compiled implementation using Cython

## Files

- `Sorting.py`: Main module containing all sorting implementations
- `cython_bubble_sort.pyx`: Cython implementation of bubble sort
- `cython_quicksort.pyx`: Cython implementation of quicksort
- `cython_sorting.pyx`: Cython implementations of merge sort and heap sort
- `setup.py`: Setup file for Cython compilation
- `setup_cython_sorting.py`: Additional setup file for Cython compilation
- `Jacc_Sim.py`: Jaccard similarity implementation

## Usage

### Basic Usage

```python
from Sorting import Sorting

# Create a pandas DataFrame
df = pd.DataFrame({'values': [5, 2, 8, 1, 9]})

# Sort using different implementations
sorted_df, time_taken = Sorting.bubble_sort(df, 'values')
sorted_df, time_taken = Sorting.quicksort_python(df, 'values')
sorted_df, time_taken = Sorting.merge_sort_numpy(df, 'values')
```

### Performance Comparison

The module includes functions to compare the performance of different sorting implementations:

```python
# Compare sorting algorithms
runtime_data = Sorting.compare_sorting_algorithms(df, 'values')
Sorting.visualize_sorting_runtimes(runtime_data)
```

## Performance Considerations

- For small datasets: Use Python or NumPy implementations
- For medium datasets: Use Numba implementations
- For large datasets: Use Cython implementations
- For general-purpose sorting: Use Tim Sort (Python's built-in sort)

## Dependencies

- pandas
- numpy
- numba
- cython
- matplotlib
- seaborn

## Setup

To use the Cython implementations, you need to compile them first:

```bash
python setup.py build_ext --inplace
python setup_cython_sorting.py build_ext --inplace
```

## Notes

- All sorting functions support both ascending and descending order
- The module includes timeout protection for long-running sorts
- Performance visualizations are available for comparing different implementations 