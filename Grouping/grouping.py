# 📦 FINAL GROUPING + MEMORY TEST (with timings + memory plots)

import pandas as pd
import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt
import time
from numba import jit

from grouping_cython import group_strings_cython
from memory_cython import memory_reduce_cython

# Timing Helper
def time_function(func, *args):
    start = time.time()
    result = func(*args)
    duration = time.time() - start
    return result, duration

# Memory Usage Helper
def memory_usage(df):
    return df.memory_usage(deep=True).sum() / 1024**2  # MB

# 📋 GROUPING FUNCTIONS

def group_baseline_pandas(df, column):
    return df.groupby(column).size().reset_index(name='count')

def group_data_pandas(df, column):
    return df[column].value_counts().reset_index(name='count')

def group_data_dask(df, column):
    ddf = dd.from_pandas(df, npartitions=4)
    grouped = ddf.groupby(column).size().compute()
    return grouped.reset_index(name='count')

@jit(forceobj=True)
def numba_group_kernel(arr):
    counts = {}
    for val in arr:
        if val in counts:
            counts[val] += 1
        else:
            counts[val] = 1
    return counts

def group_data_numba(df, column):
    arr = df[column].astype(str).values
    counts = numba_group_kernel(arr)
    return pd.DataFrame(list(counts.items()), columns=[column, 'count'])

def group_data_cython(df, column):
    values = df[column].astype(str).tolist()
    result = group_strings_cython(values)
    return pd.DataFrame(result, columns=[column, 'count'])

# 📋 MEMORY FUNCTIONS

def reduce_memory_baseline_pandas(df):
    return df

def reduce_memory_data_pandas(df):
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.api.types.is_integer_dtype(col_type):
                if c_min >= 0:
                    if c_max < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max < 65535:
                        df[col] = df[col].astype(np.uint16)
                    else:
                        df[col] = df[col].astype(np.uint32)
                else:
                    if np.iinfo(np.int8).min <= c_min <= np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif np.iinfo(np.int16).min <= c_min <= np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    else:
                        df[col] = df[col].astype(np.int32)
            else:
                df[col] = pd.to_numeric(df[col], downcast='float')
    return df

def reduce_memory_data_numpy(df):
    for col in df.select_dtypes(include=['float', 'int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

def reduce_memory_data_dask(df):
    ddf = dd.from_pandas(df, npartitions=4)
    ddf = ddf.map_partitions(lambda df: pd.DataFrame(
        {col: pd.to_numeric(df[col], downcast='float') if pd.api.types.is_numeric_dtype(df[col]) else df[col]
         for col in df.columns}))
    return ddf.compute()

@jit(forceobj=True)
def numba_dummy_reduce(arr):
    return arr

def reduce_memory_data_numba(df):
    for col in df.select_dtypes(include=['float', 'int']).columns:
        arr = df[col].values
        arr = numba_dummy_reduce(arr)
    return df

def reduce_memory_data_cython(df):
    return memory_reduce_cython(df)

# 📋 TESTING FUNCTIONS

def test_grouping_all_versions(df, column):
    print("\n📊 GROUPING TESTS:")

    timings = {}

    _, t1 = time_function(group_baseline_pandas, df.copy(), column)
    print(f"Group_Baseline_Pandas: {t1:.4f} sec")
    timings['Baseline_Pandas'] = t1

    _, t2 = time_function(group_data_pandas, df.copy(), column)
    print(f"Group_Data_Pandas: {t2:.4f} sec")
    timings['Data_Pandas'] = t2

    _, t3 = time_function(group_data_dask, df.copy(), column)
    print(f"Group_Data_Dask: {t3:.4f} sec")
    timings['Data_Dask'] = t3

    _, t4 = time_function(group_data_numba, df.copy(), column)
    print(f"Group_Data_Numba: {t4:.4f} sec")
    timings['Data_Numba'] = t4

    _, t5 = time_function(group_data_cython, df.copy(), column)
    print(f"Group_Data_Cython: {t5:.4f} sec")
    timings['Data_Cython'] = t5

    # 📈 Plot Grouping Runtime
    plt.figure(figsize=(10,6))
    plt.bar(timings.keys(), timings.values(), color='lightcoral')
    plt.title('Grouping Runtime Comparison')
    plt.xlabel('Method')
    plt.ylabel('Execution Time (seconds)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def test_memory_reduction_all_versions(df):
    print("\n🧹 MEMORY REDUCTION TESTS:")

    base_mem = memory_usage(df)
    print(f"Starting Memory: {base_mem:.2f} MB")

    timings = {}
    memory_before_after = {}

    # Testing all methods
    for method_name, reducer in [
        ('Baseline_Pandas', reduce_memory_baseline_pandas),
        ('Data_Pandas', reduce_memory_data_pandas),
        ('Data_Numpy', reduce_memory_data_numpy),
        ('Data_Dask', reduce_memory_data_dask),
        ('Data_Numba', reduce_memory_data_numba),
        ('Data_Cython', reduce_memory_data_cython)
    ]:
        reduced_df, timing = time_function(reducer, df.copy())
        after_mem = memory_usage(reduced_df)
        timings[method_name] = timing
        memory_before_after[method_name] = (base_mem, after_mem)
        reduction = ((base_mem - after_mem) / base_mem) * 100
        print(f"{method_name}: {timing:.4f} sec | After Memory: {after_mem:.2f} MB | Reduction: {reduction:.2f}%")

    # 📈 Plot Memory Reduction Runtimes
    plt.figure(figsize=(10,5))
    plt.bar(timings.keys(), timings.values(), color='skyblue')
    plt.title('Memory Reduction - Runtime Comparison')
    plt.xlabel('Method')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # 📈 Plot Memory Saved
    mems = {k: v[1] for k, v in memory_before_after.items()}
    plt.figure(figsize=(10,5))
    plt.bar(mems.keys(), mems.values(), color='lightgreen')
    plt.title('Memory Usage After Reduction')
    plt.xlabel('Method')
    plt.ylabel('Memory (MB)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# 📋 MAIN DRIVER

def main():
    path = "amazon_products.csv"
    df = pd.read_csv(path)

    focus_col = 'title' if 'title' in df.columns else df.columns[0]

    # Run Grouping
    test_grouping_all_versions(df, focus_col)

    # Run Memory Reduction
    test_memory_reduction_all_versions(df)

if __name__ == "__main__":
    main()
