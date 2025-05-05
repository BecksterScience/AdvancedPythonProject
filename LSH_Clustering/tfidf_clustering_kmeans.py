import pandas as pd
import numpy as np
import time
from itertools import product
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import os
from collections import Counter

def main():
    # Create folder for saving wordclouds
    output_dir = "tfidf_word_clouds"
    os.makedirs(output_dir, exist_ok=True)

    # ─── 1) Configuration ────────────────────────────────────────────────────────────
    DATA_PATH  = "amazon_products.csv"
    SAMPLE_N   = 900_000
    MAX_FEAT   = 1_000
    MAX_ITERS  = 10

    # ─── 2) Load & TF-IDF ─────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH, usecols=["title"]).dropna().drop_duplicates()

    available_n = len(df)
    if available_n < SAMPLE_N:
        print(f"⚠️  Only {available_n} titles available. Sampling all of them.")
        SAMPLE_N = available_n

    titles = df["title"].sample(SAMPLE_N, random_state=0).tolist()
    CLUSTERS = min(10, SAMPLE_N)

    tfidf = TfidfVectorizer(max_features=MAX_FEAT, stop_words="english")
    X = tfidf.fit_transform(titles).toarray()

    # ─── 3) PCA for Visualization (optional) ──────────────────────────────────────────
    pca2 = PCA(n_components=2, random_state=0)
    X2 = pca2.fit_transform(X)

    # ─── 4) K-Means Clustering ────────────────────────────────────────────────────────
    def kmeans_sklearn(data, k, max_iters=MAX_ITERS):
        km = KMeans(n_clusters=k, max_iter=max_iters, n_init=1, random_state=0)
        km.fit(data)
        return km.labels_, km.cluster_centers_

    methods = [("scikit-learn", kmeans_sklearn)]

    print(f"Clustering {X.shape[0]}×{X.shape[1]} TF-IDF vectors into k={CLUSTERS}")
    times = {}
    for name, fn in methods:
        start = time.time()
        labels, cents = fn(X, CLUSTERS)
        t = time.time() - start
        times[name] = t
        print(f" • {name:20s}: {t:.3f}s")

    # Bar chart for clustering times
    plt.figure(figsize=(6, 4))
    names = list(times.keys())
    vals = [times[n] for n in names]
    plt.bar(names, vals)
    plt.ylabel("Time (s)")
    plt.title("K-Means Runtime Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kmeans_runtime_comparison.png"))
    plt.close()

    # ─── 5) Save Word Clouds and Word Counts per Cluster ──────────────────────────────
    sk_labels, _ = kmeans_sklearn(X, CLUSTERS)
    wordcount_output = []

    for j in range(CLUSTERS):
        texts = [titles[i] for i, lab in enumerate(sk_labels) if lab == j]
        if not texts:
            continue

        # Save WordCloud
        wc = WordCloud(width=400, height=200, background_color="white").generate(" ".join(texts))
        plt.figure(figsize=(4, 2))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Cluster {j} Top Words")
        save_path = os.path.join(output_dir, f"cluster_{j}_wordcloud.png")
        plt.savefig(save_path)
        plt.close()

        # Save Word Counts
        words = ' '.join(texts).split()
        counter = Counter(words)
        wordcount_output.append(f"Cluster {j}:\n")
        for word, count in counter.items():
            wordcount_output.append(f"{word}: {count}")
        wordcount_output.append("\n")

    with open(os.path.join(output_dir, "cluster_word_counts.txt"), "w") as f:
        f.write("\n".join(wordcount_output))

    print(f"✅ Saved {CLUSTERS} word clouds, runtime chart, and word counts to '{output_dir}' folder.")

if __name__ == '__main__':
    main()
