"""Sparse word-context matrices and vector-space queries."""

import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.utils.extmath import randomized_svd


def tokenize(text, remove_stopwords=False):
    text = re.sub(r'<[^>]+>', ' ', str(text)).lower()
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    return [w for w in words if len(w) > 1 and (not remove_stopwords or w not in ENGLISH_STOP_WORDS)]


def phrase_tokens(sentences, min_count=20, threshold=10):
    """Merge non-overlapping bigrams by a frequency-adjusted association score."""
    counts = Counter(w for sentence in sentences for w in sentence)
    pairs = Counter(pair for sentence in sentences for pair in zip(sentence, sentence[1:]))
    accepted = {
        pair for pair, count in pairs.items()
        if count >= min_count and (count - min_count) * len(counts) / (counts[pair[0]] * counts[pair[1]]) > threshold
    }
    transformed = []
    for sentence in sentences:
        merged, i = [], 0
        while i < len(sentence):
            if i + 1 < len(sentence) and (sentence[i], sentence[i+1]) in accepted:
                merged.append(sentence[i] + '_' + sentence[i+1])
                i += 2
            else:
                merged.append(sentence[i])
                i += 1
        transformed.append(merged)
    return transformed, sorted('_'.join(p) for p in accepted)


def cooccurrence(sentences, max_vocab=3000, min_count=10, window=2):
    counts = Counter(w for sentence in sentences for w in sentence)
    words = sorted((w for w, n in counts.items() if n >= min_count), key=lambda w: (-counts[w], w))[:max_vocab]
    vocab = {w: i for i, w in enumerate(words)}
    pairs = Counter()
    for sentence in sentences:
        # Keep original positions when tokens fall outside the retained vocabulary.
        ids = [vocab.get(w, -1) for w in sentence]
        for t, source in enumerate(ids):
            if source < 0:
                continue
            for offset in range(1, window + 1):
                if t + offset < len(ids) and ids[t + offset] >= 0:
                    target = ids[t + offset]
                    pairs[source, target] += 1
                    pairs[target, source] += 1
    if not pairs:
        raise ValueError('The corpus has no retained word-context pairs')
    rows, columns = zip(*pairs)
    matrix = sparse.csr_matrix((list(pairs.values()), (rows, columns)), shape=(len(words), len(words)), dtype=np.float64)
    return words, matrix, counts


def ppmi(matrix):
    """Positive pointwise mutual information, computed only at nonzero entries."""
    c = matrix.tocoo()
    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    col_sum = np.asarray(matrix.sum(axis=0)).ravel()
    values = np.maximum(np.log(c.data * c.data.sum() / (row_sum[c.row] * col_sum[c.col])), 0)
    out = sparse.csr_matrix((values, (c.row, c.col)), shape=c.shape)
    out.eliminate_zeros()
    return out


def svd_embeddings(matrix, rank=64, seed=42, weighting='sqrt'):
    rank = min(rank, min(matrix.shape) - 1)
    u, s, _ = randomized_svd(matrix, n_components=rank, n_iter=7, random_state=seed)
    return (u * np.sqrt(s) if weighting == 'sqrt' else u), s


class VectorSpace:
    def __init__(self, words, vectors):
        self.words = list(words)
        self.index = {w: i for i, w in enumerate(words)}
        self.vectors = np.asarray(vectors)
        self.unit = self.vectors / np.maximum(np.linalg.norm(self.vectors, axis=1, keepdims=True), 1e-12)

    def query(self, vector, top_n=5, exclude=(), least=False):
        norm = np.linalg.norm(vector)
        if norm < 1e-12:
            raise ValueError('Cosine similarity is undefined for a zero vector')
        scores = self.unit @ (vector / norm)
        banned = {self.index[w] for w in exclude if w in self.index}
        ordering = np.argsort(scores if least else -scores, kind='stable')
        return [(self.words[i], float(scores[i])) for i in ordering if i not in banned][:top_n]

    def neighbors(self, word, top_n=5, least=False):
        return self.query(self.vectors[self.index[word]], top_n, exclude=[word], least=least)

    def analogy(self, a, b, c, top_n=1):
        """Answer a:b::c:? with b - a + c, excluding all three input words."""
        vector = self.vectors[self.index[b]] - self.vectors[self.index[a]] + self.vectors[self.index[c]]
        return self.query(vector, top_n, exclude=[a, b, c])

    def group_neighbors(self, words, top_n=5):
        return self.query(np.mean([self.vectors[self.index[w]] for w in words], axis=0), top_n, exclude=words)


def evaluate_analogies(space, questions):
    """Report OOV coverage separately from accuracy on covered questions."""
    covered, correct, predictions = 0, 0, []
    for a, b, c, expected in questions:
        if any(w not in space.index for w in (a, b, c, expected)):
            predictions.append({'question': [a, b, c, expected], 'prediction': None, 'status': 'out_of_vocabulary'})
            continue
        answer = space.analogy(a, b, c)[0][0]
        covered += 1
        correct += int(answer == expected)
        predictions.append({'question': [a, b, c, expected], 'prediction': answer, 'status': 'evaluated'})
    return {'total': len(questions), 'covered': covered, 'correct': correct,
            'coverage': covered / len(questions) if questions else 0,
            'accuracy_covered': correct / covered if covered else None, 'predictions': predictions}


def load_glove(path, dimension=300, limit=None):
    """Read a headerless GloVe text file without a Gensim dependency."""
    words, vectors = [], []
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            fields = line.rstrip().split()
            if len(fields) != dimension + 1:
                continue
            words.append(fields[0])
            vectors.append(np.asarray(fields[1:], dtype=np.float32))
            if limit is not None and len(words) >= limit:
                break
    if not words:
        raise ValueError('No vectors match the requested dimension')
    return VectorSpace(words, np.stack(vectors))


def project_vectors(vectors, method='pca', seed=42):
    if method == 'pca':
        return PCA(n_components=2, random_state=seed).fit_transform(vectors)
    if method == 'tsne':
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=min(3, len(vectors)-1), random_state=seed, init='pca', learning_rate='auto').fit_transform(vectors)
    raise ValueError('method must be pca or tsne')
