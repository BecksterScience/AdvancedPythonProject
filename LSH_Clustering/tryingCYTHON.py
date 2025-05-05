import pandas as pd
import numpy as np
import re
import multiprocessing as mp
import time
from datasketch import MinHash
from fastlsh import FastLSH  # <<< your Cython version of LSH!!

# ------------------- Helper to compute MinHash -------------------
def compute_minhash(idx, words, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in words:
        m.update(word.encode('utf8'))
    return (idx, m)

# ------------------- Main FastLSH Similarity Search -------------------
def get_all_similar_titles_parallel(titles_df, threshold=0.5, num_perm=128, bands=32, num_workers=None):
    title_sets = titles_df['clean_title'].apply(lambda x: set(str(x).split()))
    title_sets = list(title_sets.items())

    if num_workers is None:
        num_workers = mp.cpu_count()

    ctx = mp.get_context('fork')  # ✅ Recommended for Mac
    with ctx.Pool(processes=num_workers) as pool:
        results = pool.starmap(compute_minhash, [(idx, words, num_perm) for idx, words in title_sets])

    title_minhashes = dict(results)
    lsh = FastLSH(num_perm=num_perm, bands=bands)

    for idx, m in title_minhashes.items():
        lsh.insert(idx, m)

    seen_pairs = set()
    all_pairs = []
    for idx, m in title_minhashes.items():
        candidates = lsh.query(m)
        for other_idx in candidates:
            other_idx = int(other_idx)
            if idx == other_idx:
                continue
            pair = tuple(sorted((idx, other_idx)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            sim = m.jaccard(title_minhashes[other_idx])
            all_pairs.append((sim, pair))

    return sorted(all_pairs, reverse=True)

# ------------------- Main function (for CLI or external call) -------------------
def main():
    titles = pd.read_csv('amazon_products.csv')
    titles['clean_title'] = titles['title'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))

    start_time = time.time()

    all_similar_pairs = get_all_similar_titles_parallel(
        titles[['clean_title']],
        threshold=0.5,
        num_perm=128,
        bands=128,  # ✅ Cython LSH uses this
        num_workers=None
    )

    print(f"Found {len(all_similar_pairs)} pairs.")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

# ------------------- If run directly -------------------
if __name__ == '__main__':
    main()
