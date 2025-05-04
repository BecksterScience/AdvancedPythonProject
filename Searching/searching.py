# 📦 Searching Full Clean Code (Baseline + Data versions)

import pandas as pd
import numpy as np
import dask.dataframe as dd
from numba import jit
import time

from search_cython import search_strings_cython  # Cython compiled

# Timing helper
def time_function(func, *args):
    start = time.time()
    result = func(*args)
    duration = time.time() - start
    return result, duration

# 📋 1. Searching Baseline - Pure Pandas
def search_baseline_pandas(df, column, keyword):
    return df[df[column].str.contains(keyword, case=False, na=False)]

# 📋 2. Optimized Searching - Faster Pandas
def search_data_pandas(df, column, keyword):
    df[column] = df[column].astype(str)
    return df[df[column].str.contains(keyword, case=False, na=False)]

# 📋 3. NumPy Searching
def search_data_numpy(df, column, keyword):
    arr = df[column].astype(str).values
    arr = arr.astype('U')
    mask = np.char.find(arr, keyword) >= 0
    return df[mask]

# 📋 4. Dask Searching
def search_data_dask(df, column, keyword):
    ddf = dd.from_pandas(df, npartitions=4)
    ddf[column] = ddf[column].astype(str)
    return ddf[ddf[column].str.contains(keyword, case=False, na=False)].compute()

# 📋 5. Numba Searching
@jit(nopython=True)
def numba_search_kernel(arr, keyword):
    result = []
    for i in range(len(arr)):
        if keyword in arr[i]:
            result.append(True)
        else:
            result.append(False)
    return np.array(result)

def search_data_numba(df, column, keyword):
    arr = df[column].astype(str).values
    arr = arr.astype('U')
    mask = numba_search_kernel(arr, keyword)
    return df[mask]

# 📋 6. Cython Searching
def search_data_cython(df, column, keyword):
    values = df[column].astype(str).tolist()
    filtered = search_strings_cython(values, keyword)
    mask = df[column].isin(filtered)
    return df[mask]

# 📋 Testing all searching versions
def test_searching_all_versions(df, column, keyword):
    print("\n🔍 SEARCHING TESTS (Separate Versions)")

    _, t1 = time_function(search_baseline_pandas, df.copy(), column, keyword)
    print(f"Search_Baseline_Pandas: {t1:.4f} sec")

    _, t2 = time_function(search_data_pandas, df.copy(), column, keyword)
    print(f"Search_Data_Pandas: {t2:.4f} sec")

    _, t3 = time_function(search_data_numpy, df.copy(), column, keyword)
    print(f"Search_Data_Numpy: {t3:.4f} sec")

    _, t4 = time_function(search_data_dask, df.copy(), column, keyword)
    print(f"Search_Data_Dask: {t4:.4f} sec")

    _, t5 = time_function(search_data_numba, df.copy(), column, keyword)
    print(f"Search_Data_Numba: {t5:.4f} sec")

    _, t6 = time_function(search_data_cython, df.copy(), column, keyword)
    print(f"Search_Data_Cython: {t6:.4f} sec")

def main():
    path = "amazon_products.csv"
    df = pd.read_csv(path)
    focus_col = 'title' if 'title' in df.columns else df.columns[0]
    keyword = "laptop"

    test_searching_all_versions(df, focus_col, keyword)

if __name__ == "__main__":
    main()
