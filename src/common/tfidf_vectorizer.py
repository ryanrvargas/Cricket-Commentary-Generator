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

This design is similar to how sklearn vectorizers work but simplier
"""

import math
import re
from collections import Counter


def tokenize(text):
    """
    Lowercase and split text into simple word tokens.
    """
    text = str(text).lower() # Convert to string and lowercase
    return re.findall(r"[a-z0-9']+", text) # Find all chunks that match the regex
                                            # a - z, 0 - 9, and apostrophes, + means one or more


class TfidfVectorizerInHouse:
    def __init__(self):
        self.vocabulary_ = {} # This stores every unique word the vertorizer learns
        self.idf_ = {} # This stores the IDK scores, (Weights)
        self.fitted_ = False # This is done once the text has been vectorized

    def fit(self, documents):
        """
        Learn vocabulary and IDF values from a list of documents.
            - Documents are commentary examples
        """
        tokenized_docs = [set(tokenize(doc)) for doc in documents] # Turns each documents tokens into a set. "Four four runs" -> {"four", "runs"}
        doc_count = len(documents)

        # Build vocabulary: all unique terms across all docs, sorted for consistency
        vocab_terms = sorted(set(term for doc in tokenized_docs for term in doc))
        self.vocabulary_ = {term: idx for idx, term in enumerate(vocab_terms)} # Map term to index

        # Document frequency: count how many docs each term appears in
        df_counts = Counter()
        for doc in tokenized_docs:
            for term in doc:
                df_counts[term] += 1

        # Smooth IDF calculation: prevents div by zero, adds 1 to num/denominator
        self.idf_ = {
            term: math.log((1 + doc_count) / (1 + df_counts[term])) + 1
            for term in self.vocabulary_
        }

        self.fitted_ = True # Mark as fitted
        return self

    def transform_one(self, document):
        """
        Convert one document into a sparse TF-IDF vector represented as a dict:
            {term: tfidf_value}
        """
        if not self.fitted_:
            raise ValueError("Vectorizer must be fit before calling transform_one().")

        tokens = tokenize(document) # Tokenize input doc
        if not tokens:
            return {} # Return empty if no tokens

        tf_counts = Counter(tokens) # Count term frequencies in doc
        total_terms = len(tokens) # Total number of terms in doc

        vector = {}
        for term, count in tf_counts.items():
            if term in self.idf_: # Only use terms seen in fit
                tf = count / total_terms # Term frequency (normalized)
                vector[term] = tf * self.idf_[term] # TF-IDF weight

        return vector # Dict: {term: tfidf}

    def transform(self, documents):
        """
        Convert a list of documents into TF-IDF vectors.
        """
        return [self.transform_one(doc) for doc in documents] # List of sparse vectors

    def fit_transform(self, documents):
        """
        Fit on documents, then return their TF-IDF vectors.
        """
        self.fit(documents) # Learn vocab and IDF
        return self.transform(documents) # Return vectors


def cosine_similarity_sparse(vec_a, vec_b):
    """
    Compute cosine similarity between two sparse vectors stored as dicts.
    """
    if not vec_a or not vec_b:
        return 0.0 # If either vector is empty, similarity is 0

    shared_terms = set(vec_a.keys()) & set(vec_b.keys()) # Terms present in both vectors
    dot_product = sum(vec_a[t] * vec_b[t] for t in shared_terms) # Only sum shared terms

    norm_a = math.sqrt(sum(v * v for v in vec_a.values())) # L2 norm of vec_a
    norm_b = math.sqrt(sum(v * v for v in vec_b.values())) # L2 norm of vec_b

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0 # Avoid division by zero

    return dot_product / (norm_a * norm_b) # Cosine similarity