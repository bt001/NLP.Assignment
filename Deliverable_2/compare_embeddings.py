"""
Compare Embeddings Script
Deliverable 2
"""

import os
import pandas as pd
from utils.embedding_loader import load_embedding_model
from utils.reconstruction_io import load_reconstruction_data, save_similarity_results
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Configuration
EMBEDDING_TYPE = "sbert"  # Only SBERT is supported for evaluation
VISUALIZATION_METHOD = "pca"  
INPUT_FILE = "data/reconstructions.json"  
SIMILARITY_CSV = "outputs/similarities.csv"
PLOT_DIR = "outputs/pca_plots"
VERSION_LABELS = {
    "A": "A (Custom)",
    "B1": "B1 (SBERT)",
    "B2": "B2 (SpaCy + GloVe)",
    "B3": "B3 (FastText)"
}


def compute_cosine_similarities(model, data):
    """
    Given a list of {original, A, B, C} sentence sets, compute cosine similarities
    """
    results = []
    for item in data:
        original = item["original"]
        for label in ["A", "B1", "B2", "B3"]:
            recon = item[label]
            sim = model.similarity(original, recon)  
            results.append({
                "sentence_id": item["id"],
                "version_label": VERSION_LABELS.get(label, label),
                "original_text": original,
                "reconstructed_text": recon,
                "cosine_similarity": round(sim, 4)
            })
            
    return results


def plot_embeddings(embedding_model, data, method="pca"):
    """
    Create dimensionality reduction plots for each sentence set
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    for item in data:
        sent_id = item["id"]
        sentence_versions = {
            "original": item["original"],
            "A": item["A"],
            "B1": item["B1"],
            "B2": item["B2"],
            "B3": item["B3"]
        }

        # Tokenize and embed each version
        token_embeddings = {}
        for label, sent in sentence_versions.items():
            token_embeddings[label] = embedding_model.token_embeddings(sent)  # placeholder

        # Flatten and reduce
        for label, vectors in token_embeddings.items():
            if method == "pca":
                reducer = PCA(n_components=2)
            else:
                reducer = TSNE(n_components=2, random_state=42)
            reduced = reducer.fit_transform(vectors)

            plt.scatter(reduced[:, 0], reduced[:, 1], label=VERSION_LABELS.get(label, label))
            plt.xlabel("PCA Dimension 1")
            plt.ylabel("PCA Dimension 2")
            plt.legend(title="Version")
            plt.title(f"Semantic Space - Sentence {sent_id} - {label}")
            plt.grid(True)


        plt.title(f"Semantic Space Shift - Sentence {sent_id}")
        plt.legend()
        plt.savefig(f"{PLOT_DIR}/sentence_{sent_id}_{method}.png")
        plt.clf()


if __name__ == "__main__":
    print("Loading embedding model...")
    embedding_model = load_embedding_model(EMBEDDING_TYPE)

    print("Loading reconstruction data...")
    data = load_reconstruction_data(INPUT_FILE)

    print("Computing cosine similarities...")
    similarity_rows = compute_cosine_similarities(embedding_model, data)
    save_similarity_results(similarity_rows, SIMILARITY_CSV)

    print("Creating PCA/t-SNE plots...")
    plot_embeddings(embedding_model, data, VISUALIZATION_METHOD)

    print("Done.")
