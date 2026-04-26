"""
tfidf_vectorizer.py
-------------------
A small in-house TF-IDF vectorizer for ranking commentary examples.

This module does not depend on scikit-learn. It provides:
- tokenization
- vocabulary building
- inverse document frequency (IDF) fitting
- TF-IDF vector creation
- cosine similarity
"""

import math
import re
from collections import Counter


def tokenize(text):
    """
    Lowercase and split text into simple word tokens.
    """
    text = str(text).lower()
    return re.findall(r"[a-z0-9']+", text)


class TfidfVectorizerInHouse:
    def __init__(self):
        self.vocabulary_ = {}
        self.idf_ = {}
        self.fitted_ = False

    def fit(self, documents):
        """
        Learn vocabulary and IDF values from a list of documents.
        """
        tokenized_docs = [set(tokenize(doc)) for doc in documents]
        doc_count = len(documents)

        # Build vocabulary
        vocab_terms = sorted(set(term for doc in tokenized_docs for term in doc))
        self.vocabulary_ = {term: idx for idx, term in enumerate(vocab_terms)}

        # Document frequency
        df_counts = Counter()
        for doc in tokenized_docs:
            for term in doc:
                df_counts[term] += 1

        # Smooth IDF
        self.idf_ = {
            term: math.log((1 + doc_count) / (1 + df_counts[term])) + 1
            for term in self.vocabulary_
        }

        self.fitted_ = True
        return self

    def transform_one(self, document):
        """
        Convert one document into a sparse TF-IDF vector represented as a dict:
            {term: tfidf_value}
        """
        if not self.fitted_:
            raise ValueError("Vectorizer must be fit before calling transform_one().")

        tokens = tokenize(document)
        if not tokens:
            return {}

        tf_counts = Counter(tokens)
        total_terms = len(tokens)

        vector = {}
        for term, count in tf_counts.items():
            if term in self.idf_:
                tf = count / total_terms
                vector[term] = tf * self.idf_[term]

        return vector

    def transform(self, documents):
        """
        Convert a list of documents into TF-IDF vectors.
        """
        return [self.transform_one(doc) for doc in documents]

    def fit_transform(self, documents):
        """
        Fit on documents, then return their TF-IDF vectors.
        """
        self.fit(documents)
        return self.transform(documents)


def cosine_similarity_sparse(vec_a, vec_b):
    """
    Compute cosine similarity between two sparse vectors stored as dicts.
    """
    if not vec_a or not vec_b:
        return 0.0

    shared_terms = set(vec_a.keys()) & set(vec_b.keys())
    dot_product = sum(vec_a[t] * vec_b[t] for t in shared_terms)

    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)