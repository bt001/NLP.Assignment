import os
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from gensim.downloader import load as gensim_load
from scipy.spatial.distance import cosine

nltk.download("punkt")

# Load FastText embeddings
print("Loading FastText vectors...")
fasttext_model = gensim_load("fasttext-wiki-news-subwords-300")

# Load QQP dataset (Quora Question Pairs)
print("Loading QQP dataset...")
qqp = load_dataset("quora", split="train[:5000]", trust_remote_code=True)
print(f"✓ Loaded {len(qqp)} samples from QQP dataset.")

# Filter for formal paraphrase candidates
def is_formal(sentence):
    return (
        sentence is not None and
        len(sentence.split()) > 8 and
        "'" not in sentence and
        "?" not in sentence and
        " gonna " not in sentence and
        " wanna " not in sentence
    )

# Extract sentence2 as paraphrase candidates
paraphrase_candidates = [
    ex["questions"]["text"][1]
    for ex in qqp
    if ex["is_duplicate"] and is_formal(ex["questions"]["text"][1])
]
print(f"✓ Retained {len(paraphrase_candidates)} formal candidates.")

# Compute average FastText embeddings
def sentence_embedding(text):
    words = [w for w in text.lower().split() if w in fasttext_model]
    if not words:
        return np.zeros(fasttext_model.vector_size)
    return np.mean([fasttext_model[w] for w in words], axis=0)

# Embed candidates
print("Embedding candidate sentences...")
candidate_embeddings = [sentence_embedding(s) for s in paraphrase_candidates]

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

# Reconstruction function
def fasttext_reconstruct(text, threshold=0.75):
    sentences = sent_tokenize(text)
    reconstructed = []

    for sent in sentences:
        emb = sentence_embedding(sent)
        if np.linalg.norm(emb) == 0:
            reconstructed.append(sent)
            continue

        sims = [1 - cosine(emb, c) for c in candidate_embeddings]
        best_idx = int(np.argmax(sims))
        best_score = sims[best_idx]
        best_candidate = paraphrase_candidates[best_idx]

        length_ratio = len(best_candidate.split()) / max(1, len(sent.split()))
        shared_tokens = len(set(sent.lower().split()) & set(best_candidate.lower().split()))

        if (
            best_score > threshold and
            0.7 <= length_ratio <= 1.3 and
            shared_tokens >= 2
        ):
            reconstructed.append(best_candidate)
        else:
            reconstructed.append(sent)

    return " ".join(reconstructed)

# Run reconstruction
print("Reconstructing...")
reconstructed1 = fasttext_reconstruct(text1)
reconstructed2 = fasttext_reconstruct(text2)

# Save output
output_path = "reconstructed_texts_pipeline3_fasttext_qqp.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("Reconstructed Text 1:\n")
    f.write(reconstructed1 + "\n\n")
    f.write("Reconstructed Text 2:\n")
    f.write(reconstructed2 + "\n")

print("✅ FastText + QQP pipeline complete. Output saved to:")
print(os.path.abspath(output_path))
