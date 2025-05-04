# 📦 Filtering Full Clean Code (All Versions)

import pandas as pd
import numpy as np
import dask.dataframe as dd
from numba import jit
import time

from filter_cython import filter_strings_cython  # compiled Cython file

# 📋 Timing helper
def time_function(func, *args):
    start = time.time()
    result = func(*args)
    duration = time.time() - start
    return result, duration

# 📋 1. Baseline Filtering — Pure basic Pandas
def filter_baseline_pandas(df, column, threshold):
    return df[df[column].str.len() > threshold]

# 📋 2. Optimized Filtering — Faster Pandas
def filter_data_pandas(df, column, threshold):
    df[column] = df[column].astype(str)
    mask = df[column].str.len() > threshold
    return df[mask]

# 📋 3. NumPy Filtering
def filter_data_numpy(df, column, threshold):
    arr = df[column].astype(str).values
    arr = arr.astype('U')
    lengths = np.char.str_len(arr)
    mask = lengths > threshold
    return df[mask]

# 📋 4. Dask Filtering
def filter_data_dask(df, column, threshold):
    ddf = dd.from_pandas(df, npartitions=4)
    ddf[column] = ddf[column].astype(str)
    filtered = ddf[ddf[column].str.len() > threshold].compute()
    return filtered

# 📋 5. Numba Filtering
@jit(nopython=True)
def numba_filter_kernel(arr, threshold):
    result = []
    for i in range(len(arr)):
        if len(arr[i]) > threshold:
            result.append(True)
        else:
            result.append(False)
    return np.array(result)

def filter_data_numba(df, column, threshold):
    arr = df[column].astype(str).values
    arr = arr.astype('U')
    mask = numba_filter_kernel(arr, threshold)
    return df[mask]

# 📋 6. Cython Filtering
def filter_data_cython(df, column, threshold):
    values = df[column].astype(str).tolist()
    filtered = filter_strings_cython(values, threshold)
    mask = df[column].isin(filtered)
    return df[mask]

# 📋 Testing all filtering methods
def test_filtering_all_versions(df, column, threshold):
    print("\n🧹 FILTERING TESTS (Separate Versions)")

    _, t1 = time_function(filter_baseline_pandas, df.copy(), column, threshold)
    print(f"Filter_Baseline_Pandas: {t1:.4f} sec")

    _, t2 = time_function(filter_data_pandas, df.copy(), column, threshold)
    print(f"Filter_Data_Pandas: {t2:.4f} sec")

    _, t3 = time_function(filter_data_numpy, df.copy(), column, threshold)
    print(f"Filter_Data_Numpy: {t3:.4f} sec")

    _, t4 = time_function(filter_data_dask, df.copy(), column, threshold)
    print(f"Filter_Data_Dask: {t4:.4f} sec")

    _, t5 = time_function(filter_data_numba, df.copy(), column, threshold)
    print(f"Filter_Data_Numba: {t5:.4f} sec")

    _, t6 = time_function(filter_data_cython, df.copy(), column, threshold)
    print(f"Filter_Data_Cython: {t6:.4f} sec")

# 📋 Main Driver
def main():
    path = "amazon_products.csv"  # your dataset path
    df = pd.read_csv(path)
    focus_col = 'title' if 'title' in df.columns else df.columns[0]
    threshold = 15

    test_filtering_all_versions(df, focus_col, threshold)

if __name__ == "__main__":
    main()
