import pandas as pd
import re
import time
import multiprocessing as mp
import os
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter

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

    return sorted(all_pairs, reverse=True)

def main():
    print("📦 Running PCA LSH Clustering Pipeline...")
    titles = pd.read_csv('amazon_products.csv')
    titles['clean_title'] = titles['title'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())

    start_time = time.time()

    all_similar_pairs = get_all_similar_titles_parallel(
        titles[['clean_title']],
        threshold=0.8,
        num_perm=128,
        num_workers=None
    )

    print(f"✅ Found {len(all_similar_pairs)} similar title pairs.")
    print(f"⏱️  LSH filtering time: {time.time() - start_time:.2f} seconds")

    indices = set()
    for sim, (idx1, idx2) in all_similar_pairs:
        indices.add(idx1)
        indices.add(idx2)

    print(f"🔹 Number of unique titles in similar pairs: {len(indices)}")

    if not indices:
        print("⚠️  No similar pairs found. Skipping clustering.")
        return

    similar_titles = titles.iloc[list(indices)].drop_duplicates(subset=['clean_title']).reset_index(drop=True)
    print(f"🔹 Number of unique rows clustered (after deduplication): {len(similar_titles)}")

    # Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(similar_titles['clean_title'])

    # KMeans
    n_clusters = min(5, len(similar_titles))  # Dynamic cluster count if few titles
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(tfidf_matrix)
    similar_titles['cluster'] = clusters

    # PCA
    tfidf_dense = tfidf_matrix.toarray()
    pca = PCA(n_components=2, random_state=42)
    tfidf_2d = pca.fit_transform(tfidf_dense)

    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    # Plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tfidf_2d[:, 0], tfidf_2d[:, 1], c=clusters, cmap='tab10', s=10, alpha=0.7)
    plt.title("PCA of TF-IDF + KMeans Clusters (Filtered by LSH)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "pca_clusters_lsh_filtered.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📈 PCA plot saved to: {plot_path}")

    # Word counts
    wordcount_path = os.path.join(output_dir, "cluster_word_counts.txt")
    with open(wordcount_path, "w", encoding="utf-8") as f:
        for i in range(n_clusters):
            f.write(f"Cluster {i} Word Counts:\n")
            cluster_titles = similar_titles[similar_titles['cluster'] == i]['clean_title']
            text = ' '.join(cluster_titles)
            word_counts = Counter(text.split()).most_common()
            for word, count in word_counts:
                f.write(f"{word} : {count}\n")
            f.write("\n" + "-" * 40 + "\n\n")
    print(f"📄 Word counts per cluster saved to: {wordcount_path}")

if __name__ == '__main__':
    main()