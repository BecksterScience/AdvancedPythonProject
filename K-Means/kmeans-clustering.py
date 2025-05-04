import pandas as pd
import numpy as np
import time
from itertools import product
from numba import njit
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from wordcloud import WordCloud

#  1) Configuration 
DATA_PATH  = "/content/amazon_products.csv"  # path to your CSV
SAMPLE_N   = 50_000                         # how many titles to sample
MAX_FEAT   = 1_000                          # TF-IDF features
MAX_ITERS  = 10
CLUSTERS   = 10

def main():
    #  2) Load & TF-IDF 
    df = pd.read_csv(DATA_PATH, usecols=["title"]).dropna()
    titles = df["title"].sample(SAMPLE_N, random_state=0).tolist()

    tfidf = TfidfVectorizer(max_features=MAX_FEAT, stop_words="english")
    X = tfidf.fit_transform(titles).toarray()  # shape (SAMPLE_N, MAX_FEAT)

    #  3) PCA for 2D projection 
    pca2 = PCA(n_components=2, random_state=0)
    X2 = pca2.fit_transform(X)

    #  4) K-Means Implementations 
    def kmeans_python(data, k, max_iters=MAX_ITERS):
        """Pure-Python loops, using itertools.product for assignment."""
        n, dim = data.shape
        rng = np.random.RandomState(0)
        centroids = data[rng.choice(n, k, replace=False)].copy()
        labels = np.zeros(n, dtype=int)

        for _ in range(max_iters):
            best_d = np.full(n, np.inf)
            best_j = np.zeros(n, dtype=int)
            for i, j in product(range(n), range(k)):
                d = 0.0
                for d_ in range(dim):
                    t = data[i, d_] - centroids[j, d_]
                    d += t*t
                if d < best_d[i]:
                    best_d[i] = d
                    best_j[i] = j
            labels = best_j.copy()

            for j in range(k):
                pts = data[labels == j]
                if len(pts):
                    centroids[j] = pts.mean(axis=0)

        return labels, centroids

    def kmeans_numpy(data, k, max_iters=MAX_ITERS):
        """NumPy‐vectorized distance computations."""
        n, dim = data.shape
        rng = np.random.RandomState(0)
        centroids = data[rng.choice(n, k, replace=False)].copy()

        for _ in range(max_iters):
            dists = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
            labels = dists.argmin(axis=1)
            for j in range(k):
                pts = data[labels == j]
                if pts.shape[0]:
                    centroids[j] = pts.mean(axis=0)

        return labels, centroids

    @njit
    def kmeans_numba(data, k, max_iters=MAX_ITERS):
        """Numba‐accelerated loops."""
        n, dim = data.shape
        centroids = data[:k].copy()
        labels = np.empty(n, np.int32)
        for _ in range(max_iters):
            for i in range(n):
                best_d = 1e18
                best_j = 0
                for j in range(k):
                    d = 0.0
                    for d_ in range(dim):
                        t = data[i, d_] - centroids[j, d_]
                        d += t*t
                    if d < best_d:
                        best_d, best_j = d, j
                labels[i] = best_j
            for j in range(k):
                count = 0
                sumv = np.zeros(dim)
                for i in range(n):
                    if labels[i] == j:
                        count += 1
                        for d_ in range(dim):
                            sumv[d_] += data[i, d_]
                if count > 0:
                    for d_ in range(dim):
                        centroids[j, d_] = sumv[d_] / count
        return labels, centroids

    def kmeans_sklearn(data, k, max_iters=MAX_ITERS):
        """scikit-learn’s KMeans."""
        km = KMeans(n_clusters=k, max_iter=max_iters, n_init=1, random_state=0)
        km.fit(data)
        return km.labels_, km.cluster_centers_

    #  5) Benchmark & Bar Chart 
    methods = [
        ("Pure-Python",      kmeans_python),
        ("NumPy-vectorized", kmeans_numpy),
        ("Numba-JIT",        kmeans_numba),
        ("scikit-learn",     kmeans_sklearn),
    ]

    print(f"Clustering {X.shape[0]}×{X.shape[1]} TF-IDF vectors into k={CLUSTERS}")
    times = {}
    for name, fn in methods:
        start = time.time()
        labels, cents = fn(X, CLUSTERS)
        t = time.time() - start
        times[name] = t
        print(f" • {name:20s}: {t:.3f}s")

    # Bar chart of runtimes
    plt.figure(figsize=(6,4))
    names_list = list(times.keys())
    vals = [times[n] for n in names_list]
    plt.bar(names_list, vals)
    plt.ylabel("Time (s)")
    plt.title("K-Means Runtime Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    #  6) PCA Scatter Plot 
    # Use scikit-learn labels for coloring
    sk_labels, _ = kmeans_sklearn(X, CLUSTERS)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(
        X2[:,0], X2[:,1],
        c=sk_labels,
        cmap="tab10",
        s=10,
        alpha=0.7
    )
    plt.title(f"PCA(2) Projection of {SAMPLE_N} Titles (k={CLUSTERS})")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(
        *scatter.legend_elements(),
        title="Cluster",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )
    plt.tight_layout()
    plt.show()

    #  7) Word Clouds per Cluster 
    for j in range(CLUSTERS):
        texts = [titles[i] for i, lab in enumerate(sk_labels) if lab == j]
        if not texts:
            continue
        wc = WordCloud(width=400, height=200, background_color="white") \
             .generate(" ".join(texts))
        plt.figure(figsize=(4,2))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Cluster {j} Top Words")
    plt.show()

if __name__ == "__main__":
    main()
