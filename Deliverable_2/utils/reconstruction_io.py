# analysis/utils/reconstruction_io.py

import json
import pandas as pd

def load_reconstruction_data(filepath):
    """
    Loads a JSON file with structure:
    [
        {
            "id": 1,
            "original": "...",
            "A": "...",
            "B1": "...",
            "B2": "...",
            "B3": "..."
        },
        ...
    ]
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_similarity_results(rows, output_path):
    """
    Saves cosine similarity results to CSV
    Input format: list of dicts with keys: id, version, similarity
    """
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")
