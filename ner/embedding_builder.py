from gensim.models import word2vec
import sys
from pathlib import Path
import numpy as np
if __name__ == '__main__':
    rootPath = sys.argv[1]
    # 对应的加载方式
    model = word2vec.Word2Vec.load("{}/embedding.model".format(rootPath))
    with Path("{}/vocab.words.txt".format(rootPath)).open() as f:
        word_to_idx = {line: idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)
    found = 0
    # Array of zeros
    embeddings = np.zeros((size_vocab, 100))
    for w, idx in word_to_idx.items():
        w = w.replace("\n", "")
        if w in model:
            found += 1
            embeddings[idx] = model[w]
    print('- done. Found {} vectors for {} words'.format(found, size_vocab))
    # Save np.array to file
    np.savez_compressed('{}/embedding.npz'.format(rootPath), embeddings=embeddings)
