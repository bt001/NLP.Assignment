# analysis/utils/embedding_loader.py

"""
Embedding Loader Module
- Supports: SBERT, FastText
"""

# Imports for SBERT and FastText
from sentence_transformers import SentenceTransformer, util
import numpy as np


class SBERTWrapper:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def similarity(self, sent1, sent2):
        emb1 = self.model.encode(sent1, convert_to_tensor=True)
        emb2 = self.model.encode(sent2, convert_to_tensor=True)
        return util.cos_sim(emb1, emb2).item()

    def token_embeddings(self, sentence):
        """
        Return per-token embeddings using SBERT's internal tokenizer and encoder.
        Useful for PCA/t-SNE visualizations.
        """
        encoded = self.model.tokenize([sentence])
        output = self.model.encode([sentence], output_value='token_embeddings', convert_to_numpy=True)

        # `output` is shape (1, seq_len, hidden_dim)
        return output[0]

def load_embedding_model(embedding_type):
    if embedding_type == "sbert":
        return SBERTWrapper()
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")

