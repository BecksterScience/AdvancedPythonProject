import pandas as pd
from datasketch import MinHash, MinHashLSH
import multiprocessing as mp
import re
import time
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# -----------------------
# 1. Compute MinHash faster
# -----------------------

def compute_minhash(idx, words, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in words:
        m.update(word.encode('utf8'))
    return (idx, m)

# -----------------------
# 2. Main Similarity Pipeline
# -----------------------

def get_all_similar_titles_parallel(titles_df, threshold=0.5, num_perm=128, num_workers=None):
    title_sets = titles_df['clean_title'].apply(lambda x: set(x.split()))
    title_sets = list(title_sets.items())

    if num_workers is None:
        num_workers = mp.cpu_count()

    ctx = mp.get_context('fork')  # Mac fix
    with ctx.Pool(processes=num_workers) as pool:
        results = pool.starmap(compute_minhash, [(idx, words, num_perm) for idx, words in title_sets])

    title_minhashes = dict(results)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for idx, m in title_minhashes.items():
        lsh.insert(str(idx), m)

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

# -----------------------
# 3. Preprocessing
# -----------------------

def minimal_clean(text):
    text = re.sub(r'[^\w\s]', '', str(text))
    text = text.lower()
    return text

# -----------------------
# 4. Wrapped in main()
# -----------------------

def main():
    titles = pd.read_csv('amazon_products.csv')
    titles['clean_title'] = titles['title'].apply(minimal_clean)

    start_time = time.time()

    all_similar_pairs = get_all_similar_titles_parallel(
        titles[['clean_title']],
        threshold=0.5,
        num_perm=128,
        num_workers=None
    )

    print(f"\n✅ Found {len(all_similar_pairs)} similar pairs.")
    print(f"⏱️  Total time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
