import os
import nltk
import re
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

nltk.download("punkt")

# Load SBERT model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Load PAWS-Wiki dataset
print("Loading PAWS-Wiki dataset...")
paws = load_dataset("paws", "labeled_final", split="train")

# Formality filter
def is_formal(sentence):
    return (
        sentence is not None and
        len(sentence.split()) > 6 and
        "'" not in sentence and
        "?" not in sentence and
        " gonna " not in sentence and
        " wanna " not in sentence
    )

# Extract sentence2 from positive (duplicate) pairs
paraphrase_candidates = [ex["sentence2"] for ex in paws if ex["label"] == 1 and is_formal(ex["sentence2"])]
print(f"✓ Retained {len(paraphrase_candidates)} formal paraphrase candidates.")

# Encode with SBERT
print("Encoding candidates...")
paraphrase_embeddings = model.encode(paraphrase_candidates, convert_to_tensor=True)

# Input texts
text1 = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in 
our lives. Hope you too, to enjoy it as my deepest wishes. 
Thank your message to show our words to  the doctor, as his next contract checking, to all of us. 
I got this message to see the approved message. In fact, I have received the message from  the 
professor, to show me, this, a couple of days ago.  I am very appreciated  the full support of the 
professor, for our Springer proceedings publication"""

text2 = """During our final discuss, I told him about the new submission — the one we were waiting since 
last autumn, but the updates was confusing as it not included the full feedback from reviewer or 
maybe editor?
Anyway, I believe the team, although bit delay and less communication at recent days, they really 
tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance 
and efforts until the Springer link came finally last week, I think.
Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before 
he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so.
Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future 
targets"""

# Refined clause splitter
def split_clauses(sentence):
    return re.split(r"(?<=\w)[,;]\s+|\s+(?<!not)and\s+|\s+but\s+|\s+or\s+", sentence)

# SBERT clause-wise reconstruction
def sbert_reconstruct(text, threshold=0.6):
    sentences = sent_tokenize(text)
    reconstructed = []

    for sent in sentences:
        clauses = split_clauses(sent)
        new_clauses = []

        for clause in clauses:
            if not clause.strip():
                continue

            query_embedding = model.encode(clause.strip(), convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, paraphrase_embeddings, top_k=1)
            best = hits[0][0]
            score = best["score"]
            idx = best["corpus_id"]

            candidate = paraphrase_candidates[idx]
            length_ratio = len(candidate.split()) / max(1, len(clause.split()))
            shared_tokens = len(set(clause.lower().split()) & set(candidate.lower().split()))

            if score > threshold and 0.6 <= length_ratio <= 1.4 and shared_tokens >= 1:
                new_clauses.append(candidate)
            else:
                new_clauses.append(clause.strip())

        reconstructed.append(", ".join(new_clauses))

    return " ".join(reconstructed)

# Run reconstruction
print("Reconstructing...")
reconstructed1 = sbert_reconstruct(text1)
reconstructed2 = sbert_reconstruct(text2)

# Save output
output_path = "reconstructed_texts_pipeline1_sbert_pawswiki_clauses_refined.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("Reconstructed Text 1:\n")
    f.write(reconstructed1 + "\n\n")
    f.write("Reconstructed Text 2:\n")
    f.write(reconstructed2 + "\n")

print("✅ SBERT + PAWS-Wiki (refined clauses) pipeline complete. Output saved to:")
print(os.path.abspath(output_path))
