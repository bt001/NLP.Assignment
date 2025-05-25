
# 📘 NLP Assignment 2025 – Semantic Text Reconstruction

## 🔍 Overview

This project addresses the task of transforming semantically ambiguous or malformed text into fluent, well-structured, and semantically faithful reconstructions using NLP techniques. We implement and compare several automated pipelines using Python-based tools and word embeddings.

## 📂 Project Structure

```
.
├── Deliverable_1/
│   ├── 1A/
│   │   ├── deliverable1A.py
│   │   └── requirement_3.txt
│   ├── 1B/
│   │   ├── FastText_AHR/
│   │   │   ├── pipeline3_fasttext_qqp_clauses.py
│   │   │   ├── process_and_conclusions.md
│   │   │   └── reconstructed_texts_pipeline3_fasttext_qqp_clauses.txt
│   │   ├── SBERT/
│   │   │   ├── pipeline1_sbert_pawswiki_clauses_refined.py
│   │   │   ├── process_and_conclusions.md
│   │   │   └── reconstructed_texts_pipeline1_sbert_pawswiki_clauses_refined.txt
│   │   ├── SpaCy_GloVe/
│   │       ├── pipeline2_spacy_glove.py
│   │       ├── process_and_conclusions.md
│   │       └── reconstructed_texts_pipeline2_spacy_glove.txt
│   │   
│   └── 1C/
├── Deliverable_2/
│   ├── data/
│   ├── outputs/
│   │   ├── pca_plots/
│   │   ├── pca_plots.py
│   │   └── similarities.csv
│   └── utils/
│       ├── embedding_loader.py
│       ├── reconstruction_io.py
│       └── compare_embeddings.py
├── assignmentNLP2025_.pdf
├── environment.yml
├── README.md
├── setup_conda_env.ps1
├── setup_conda_env.sh
├── .env
├── .gitignore
```

## ✅ Deliverables

- **Deliverable 1 (Text Reconstruction):**

  - A: Custom rule-based reconstruction of 2 selected sentences.
  - B: Full-text reconstructions using 3 automated pipelines:
    - B1: SBERT + QQP/PAWS-Wiki (semantic similarity)
    - B2: SpaCy + GloVe + syntax-based corrections
    - B3: FastText + QQP (clause-level cosine matching)
  - C: Initial qualitative comparison.

- **Deliverable 2 (Computational Analysis):**

  - Cosine similarity scoring between original and reconstructed texts.
  - PCA visualization of semantic shift using embeddings.
  - Analysis of performance across methods A–C.

- **Deliverable 3 (Structured Report):**

  - Methodology, results, and comparative discussion.
  - Insights into NLP reconstruction limitations and trade-offs.
  - Written in Greek, `report/Deliverable3_Report_GR.md` (not shown above).

## 🛠️ Setup Instructions

This project uses **Conda** and **Poetry** for environment and dependency management. All model/data caches are redirected to the `E:/` drive.

### 1. Clone the Repository

```bash
git clone https://github.com/bt001/NLP.Assignment.git
cd NLP.Assignment
```

### 2. Create the Environment

Choose one of the following scripts depending on your OS:

- **Linux/macOS**:

  ```bash
  bash setup_conda_env.sh
  ```

- **Windows (PowerShell)**:

  ```powershell
  .\setup_conda_env.ps1
  ```

These scripts will:

- Create a Conda environment with Python >=3.10
- Install project dependencies

## 📈 Methods & Pipelines

| Pipeline | Description                                                              | Tools Used                        |
| -------- | ------------------------------------------------------------------------ | --------------------------------- |
| A        | Rule-based reconstruction using syntactic patterns and heuristics        | Python, SpaCy (partial), Regex    |
| B1       | SBERT embeddings + clause-level cosine filtering + PAWS/QQP data         | `sentence-transformers`, QQP/PAWS |
| B2       | SpaCy dependency parsing + GloVe for similarity-based grammar correction | `spaCy`, `GloVe`, POS rules       |
| B3       | FastText embeddings + clause-level semantic retrieval from QQP           | `fasttext`, cosine similarity     |

## 📊 Evaluation

- **Cosine Similarity**: Measures semantic alignment pre- and post-reconstruction.
- **PCA Plotting**: Embedding space shift visualization (for A, B1, B2, B3).

### 🔎 Observations

- B1 (SBERT): Conservative, high semantic fidelity.
- B2 (SpaCy): Grammar-oriented, shallow but safe edits.
- B3 (FastText): Unstable — frequent but sometimes off-topic reconstructions.
- A (Rule-based): Most balanced — retains meaning while improving structure.

## 📌 Notes

- All pipelines are fully automatic.
- Clause-level processing was used in B1 and B3 to improve granularity.
- `.env` contains only configuration paths — no secrets.
- Outputs are deterministic and reproducible across runs.

## 📚 References

- SBERT: Reimers & Gurevych (2019) — [ACL Anthology](https://www.aclweb.org/anthology/D19-1410/)
- FastText: Bojanowski et al. (2017) — [ACL Anthology](https://aclanthology.org/E17-2025/)
- QQP and PAWS-Wiki datasets
- SpaCy, GloVe from Stanford NLP

---

Feel free to open issues or pull requests if reproducing the results becomes inconsistent due to versioning or cache issues.
