#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

print("📦 Welcome to the Product Title Similarity & Clustering Toolkit!")
print("➡️  Make sure you have 'amazon_products.csv' with a 'title' column in this directory.")

if not os.path.exists("amazon_products.csv"):
    print("❌ Missing 'amazon_products.csv'. Please add your dataset and restart.")
    sys.exit(1)

# Try importing the Cython-based module with error handling
try:
    import tryingCYTHON
except ModuleNotFoundError as e:
    print("\n⚠️  Cython module `fastlsh` not found or not built for this Python version.")
    print("This likely means you're not using Python 3.10 on macOS.")
    print("To rebuild, run: python setup.py build_ext --inplace")
    print("Then run this program again.")
    sys.exit(1)

# === Imports from other local project files ===
import get_top_similar_titles_parallel_JUST_MP
import get_top_similar_titles_parallel_more
import lsh_with_clustering_final
import optimizing_LSH
import pca_LSH
import tfidf_clustering_kmeans

def main():
    while True:
        print("\nChoose a module to run:")
        print("1. Run LSH (just multiprocessing)")
        print("2. Run LSH (more advanced multiprocessing)")
        print("3. Run LSH with clustering and word clouds")
        print("4. Run Optimized LSH")
        print("5. Run PCA visualizer")
        print("6. Run TF-IDF clustering with KMeans")
        print("7. Run Cython-based experiment")
        print("8. Exit")
        choice = input("Your choice: ").strip()
        
        if choice == "1":
            get_top_similar_titles_parallel_JUST_MP.main()
        elif choice == "2":
            get_top_similar_titles_parallel_more.main()
        elif choice == "3":
            lsh_with_clustering_final.main()
        elif choice == "4":
            optimizing_LSH.main()
        elif choice == "5":
            pca_LSH.main()
        elif choice == "6":
            tfidf_clustering_kmeans.main()
        elif choice == "7":
            tryingCYTHON.main()
        elif choice == "8":
            print("Exiting.")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == '__main__':
    main()
