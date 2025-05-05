import pandas as pd
import numpy as np
from datasketch import MinHash
import re
import multiprocessing as mp
import time
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------------------
# Fast MinHashing
# ---------------------------

def compute_minhash_signature(idx, words, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in words:
        m.update(word.encode('utf8'))
    return (idx, m)

# ---------------------------
# Custom LSH table
# ---------------------------

class FastLSH:
    def __init__(self, num_perm=128, bands=32):
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.tables = [{} for _ in range(self.bands)]

    def insert(self, idx, signature):
        sig = np.array(signature.hashvalues)
        for band_idx in range(self.bands):
            start = band_idx * self.rows
            end = start + self.rows
            band = tuple(sig[start:end])
            if band not in self.tables[band_idx]:
                self.tables[band_idx][band] = []
            self.tables[band_idx][band].append(idx)

    def query(self, signature):
        sig = np.array(signature.hashvalues)
        candidates = set()
        for band_idx in range(self.bands):
            start = band_idx * self.rows
            end = start + self.rows
            band = tuple(sig[start:end])
            if band in self.tables[band_idx]:
                candidates.update(self.tables[band_idx][band])
        return list(candidates)

# ---------------------------
# Helpers
# ---------------------------

def clean_title(text):
    text = re.sub(r'[^\w\s]', '', str(text)).lower()
    stopwords = set(ENGLISH_STOP_WORDS)
    words = text.split()
    words_cleaned = [word for word in words if word not in stopwords]
    return ' '.join(words_cleaned)

def get_all_similar_titles_fast(titles_df, threshold=0.5, num_perm=128, bands=32, num_workers=None):
    title_sets = titles_df['clean_title'].apply(lambda x: set(x.split()))
    title_sets = list(title_sets.items())

    if num_workers is None:
        num_workers = mp.cpu_count()

    ctx = mp.get_context('fork')
    with ctx.Pool(processes=num_workers) as pool:
        signatures = pool.starmap(
            compute_minhash_signature,
            [(idx, words, num_perm) for idx, words in title_sets]
        )

    lsh = FastLSH(num_perm=num_perm, bands=bands)

    for idx, sig in signatures:
        lsh.insert(idx, sig)

    idx_to_sig = dict(signatures)

    seen_pairs = set()
    all_pairs = []
    for idx, sig in idx_to_sig.items():
        candidates = lsh.query(sig)
        for other_idx in candidates:
            if idx == other_idx:
                continue
            pair = tuple(sorted((idx, other_idx)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            sim = sig.jaccard(idx_to_sig[other_idx])
            if sim >= threshold:
                all_pairs.append((sim, pair))

    return sorted(all_pairs, reverse=True)

# ---------------------------
# Main Wrapper
# ---------------------------

def main():
    titles = pd.read_csv('amazon_products.csv')
    titles['original_title'] = titles['title']
    titles['clean_title'] = titles['title'].apply(clean_title)

    start_time = time.time()

    all_similar_pairs = get_all_similar_titles_fast(
        titles[['clean_title']],
        threshold=0.5,
        num_perm=128,
        bands=32,
        num_workers=None
    )

    print(f"\n✅ Found {len(all_similar_pairs)} similar pairs.")
    print(f"⏱️  Total time: {time.time() - start_time:.2f} seconds")

    for sim, (idx1, idx2) in all_similar_pairs[:5]:
        print(f"Similarity {sim:.2f}: {titles.iloc[idx1]['original_title']} --- {titles.iloc[idx2]['original_title']}")

if __name__ == '__main__':
    main()
