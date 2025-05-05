import pandas as pd
import re
import time
import numpy as np
import os
from datasketch import MinHash, MinHashLSH
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import multiprocessing as mp
from collections import Counter

# ------------- Your original MinHash + LSH similarity search -------------
def compute_minhash(idx, words, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in words:
        m.update(word.encode('utf8'))
    return (idx, m)

def get_all_similar_titles_parallel(titles_df, threshold=0.5, num_perm=128, num_workers=None):
    title_sets = titles_df['clean_title'].apply(lambda x: set(str(x).split()))
    title_sets = list(title_sets.items())

    if num_workers is None:
        num_workers = mp.cpu_count()

    with mp.get_context('fork').Pool(processes=num_workers) as pool:
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

    all_pairs_sorted = sorted(all_pairs, reverse=True)
    return all_pairs_sorted

# ------------- Clustering, Word Clouds, Word Counts Saving -------------
def cluster_titles_and_save_wordclouds(titles_df, all_similar_pairs, n_clusters=10):
    output_dir = "word_clouds"
    os.makedirs(output_dir, exist_ok=True)

    indices = set()
    for sim, (idx1, idx2) in all_similar_pairs:
        indices.add(idx1)
        indices.add(idx2)

    similar_titles = titles_df.iloc[list(indices)].drop_duplicates(subset=['clean_title']).reset_index(drop=True)

    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(similar_titles['clean_title'])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(tfidf_matrix)

    similar_titles['cluster'] = clusters

    wordcount_output = []

    for i in range(n_clusters):
        cluster_titles = similar_titles[similar_titles['cluster'] == i]['clean_title']
        text = ' '.join(cluster_titles)

        # Save WordClouds
        wordcloud = WordCloud(background_color='white', max_words=100).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Cluster {i}')
        save_path = os.path.join(output_dir, f"cluster_{i}_wordcloud.png")
        plt.savefig(save_path)
        plt.close()

        # Count Unique Words
        words = text.split()
        counter = Counter(words)
        wordcount_output.append(f"Cluster {i}:\n")
        for word, count in counter.items():
            wordcount_output.append(f"{word}: {count}")
        wordcount_output.append("\n")

    # Save word counts
    counts_save_path = os.path.join(output_dir, "cluster_word_counts.txt")
    with open(counts_save_path, "w") as f:
        f.write("\n".join(wordcount_output))

    print(f"✅ Saved {n_clusters} word clouds and word counts to '{output_dir}'.")

# ------------- Main Wrapper -------------
def main():
    titles = pd.read_csv('amazon_products.csv')
    titles['clean_title'] = titles['title'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))

    start_time = time.time()

    all_similar_pairs = get_all_similar_titles_parallel(
        titles[['clean_title']],
        threshold=0.7,
        num_perm=128,
        num_workers=None
    )

    print(f"\n✅ Found {len(all_similar_pairs)} similar pairs.")
    print(f"⏱️  Total time: {time.time() - start_time:.2f} seconds")

    if len(all_similar_pairs) == 0:
        print("⚠️  No similar pairs found — skipping clustering and word clouds.")
    else:
        cluster_titles_and_save_wordclouds(titles, all_similar_pairs, n_clusters=10)

if __name__ == '__main__':
    main()
