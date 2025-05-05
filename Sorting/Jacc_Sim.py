import pandas as pd
import numpy as np
import os
import time
from nltk import ngrams
from tqdm import tqdm
from numba import njit
# Jaccard Similarity Function
def jaccard_similarity(str1, str2):
    """
    Calculate the Jaccard similarity between two strings.

    Parameters:
    str1 (str): First string.
    str2 (str): Second string.

    Returns:
    float: Jaccard similarity score.
    """
    set1 = (str1)
    set2 = (str2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    else:
        return intersection / union
    

# jaccard_sim_numba
@njit
def jaccard_sim_numba(set1, set2):
    """
    Calculate the Jaccard similarity between two sets using Numba for optimization.

    Parameters:
    set1 (set): First set.
    set2 (set): Second set.

    Returns:
    float: Jaccard similarity score.
    """
    set1 = set(set1)
    set2 = set(set2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    else:
        return intersection / union



if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))  # Change directory to the script's location
    # Example usage
    df = pd.read_csv('./Data/amazon_products.csv')

    jaccard_similarity__dict = {}
    # Calculate Jaccard similarity for the first two 100_000 rows
    """
    time_start = time.time()
    for i in tqdm(range(10_000), position=0, leave=True):
        # Convert str1 to bigrams
        str1 = df['title'].iloc[i]
        n_grams1 = set(ngrams(str1.split(), 2))  # Use list() instead of .to_list()
        for j in tqdm(range(i + 1, 10_001), position=1, leave=False):
            str2 = df['title'].iloc[j]
            # Convert str2 to bigrams
            n_grams2 = set(ngrams(str2.split(), 2))  # Use list() instead of .to_list()
            similarity = jaccard_similarity(n_grams1, n_grams2)
            # Store the results in a dictionary
            if(similarity >= 0.9):
                jaccard_similarity__dict[(i, j)] = similarity
    time_end = time.time()
"""
 #   print(f"Jaccard similarity calculated in {time_end - time_start:.2f} seconds.")
    # Convert the dictionary to a DataFrame
  #  jaccard_similarity_df = pd.DataFrame.from_dict(jaccard_similarity__dict, orient='index', columns=['Jaccard Similarity'])
    # Reset the index to get the row pairs as columns
   # jaccard_similarity_df.reset_index(inplace=True)
    # Save the results to a CSV file
    #jaccard_similarity_df.to_csv('./Data/jaccard_similarity_results.csv', index=False)

    
    time_start_numba = time.time()
    jaccard_similarity__dict = {}
    for i in tqdm(range(10_000), position=0, leave=True):
        # Convert str1 to bigrams
        str1 = df['title'].iloc[i]
        n_grams1 = set(ngrams(str1.split(), 2))  # Use list() instead of .to_list()
        for j in tqdm(range(i + 1, 10_001), position=1, leave=False):
            str2 = df['title'].iloc[j]
            # Convert str2 to bigrams
            n_grams2 = set(ngrams(str2.split(), 2))  # Use list() instead of .to_list()
            similarity = jaccard_similarity(n_grams1, n_grams2)
            # Store the results in a dictionary
            if(similarity >= 0.9):
                jaccard_similarity__dict[(i, j)] = similarity
    time_end_numba = time.time()
    print(f"Jaccard similarity calculated using Numba in {time_end_numba - time_start_numba:.2f} seconds.")
    # Convert the dictionary to a DataFrame
    jaccard_similarity_df = pd.DataFrame.from_dict(jaccard_similarity__dict, orient='index', columns=['Jaccard Similarity'])
    # Reset the index to get the row pairs as columns
    jaccard_similarity_df.reset_index(inplace=True)
    # Save the results to a CSV file
    jaccard_similarity_df.to_csv('./Data/jaccard_similarity_results_numba.csv', index=False)

    jaccard_similarity__dict = {}