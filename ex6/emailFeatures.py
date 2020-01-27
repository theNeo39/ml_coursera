import numpy as np
from getVocabList import getVocabList

def emailFeatures(word_indices):
    """takes in a word_indices vector and
    produces a feature vector from the word indices.
    """

# Total number of words in the dictionary
    n = 1899

# You need to return the following variables correctly.
    x = np.zeros(n)
    with open('vocab.txt') as f:
        vocab_index = []
        for line in f:
            idx, w = line.split()
            vocab_index.append(idx)
    for index in word_indices:
        if str(index) in vocab_index:
            x[index-1]=1
    return x