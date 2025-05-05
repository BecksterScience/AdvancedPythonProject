
# Product Title Similarity & Clustering Toolkit

This project includes clustering, similarity detection, and visualization using:
- MinHash + LSH (parallel + Cython)
- TF-IDF + KMeans
- PCA visualizations
- WordCloud generation

---

## 📁 Contents

- `main.py` — interactive menu to run any module
- `tryingCYTHON.py` — uses `fastlsh.cpython-310-darwin.so` (Cython-compiled)
- `fastlsh.pyx`, `setup.py` — Cython source + build script
- `fastlsh.cpython-310-darwin.so` — Precompiled Cython extension for Python 3.10 (macOS)
- `amazon_products.csv` — placeholder CSV
- Notebooks: `eda.ipynb`, `clustering_kmeans.ipynb`

---

## ⚠️ Cython Note

If `main.py` fails with:
```
ModuleNotFoundError: No module named 'fastlsh'
```
You're likely using a different OS or Python version.

To fix:
```bash
python setup.py build_ext --inplace
```

Then re-run:
```bash
python main.py
```

This will recompile `fastlsh.pyx` to work with your environment.

---

## ✅ Run Instructions

```bash
python main.py
```

Select from the menu. Make sure `amazon_products.csv` is present with a `title` column.
