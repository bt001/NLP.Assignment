import subprocess
import sys
import os
from pathlib import Path
import importlib.util

def run(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def is_installed(module_name):
    spec = importlib.util.find_spec(module_name)
    return spec is not None

def main():
    print("📦 NLP Assignment Setup Script")
    env_name = input("🔧 Enter a name for the new conda environment (e.g. nlp_env_1b): ").strip()
    python_version = "3.9"

    print(f"📁 Creating new conda environment: {env_name}")
    run(f"conda create -y -n "{env_name}" python={python_version}")

    print("\n✅ Conda environment created.")
    print("⚠️ NOTE: You must activate the environment manually before running the deliverables.")
    print(f"Run this in your shell:")
    print(f"  conda activate "{env_name}"\n")

    packages = [
        "transformers",
        "datasets",
        "sentence-transformers",
        "spacy",
        "nltk",
        "scipy",
        "gensim"
    ]

    print(f"📥 Installing required packages...")
    pip_install = (
        f"conda run -n "{env_name}" python -m pip install --upgrade pip && "
        f"conda run -n "{env_name}" python -m pip install " + " ".join(packages)
    )
    run(pip_install)

    print("📥 Downloading NLTK resources...")
    run(f"conda run -n "{env_name}" python -m nltk.downloader punkt")

    print("📥 Downloading SpaCy English model...")
    run(f"conda run -n "{env_name}" python -m spacy download en_core_web_lg")

    print("\n🎉 Setup complete!")
    print("After activation, you can run any of the deliverable scripts (e.g. pipeline1_sbert_qqp.py).")

if __name__ == "__main__":
    main()
