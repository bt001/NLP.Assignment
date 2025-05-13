import os
import nltk
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt')

# Load SBERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load more paraphrases from PAWS (up to 5000 examples)
paws = load_dataset("paws", "labeled_final", split="train")
paraphrase_candidates = [ex['sentence2'] for ex in paws if ex['label'] == 1][:5000]

# Encode the larger paraphrase candidate set
paraphrase_embeddings = model.encode(paraphrase_candidates, convert_to_tensor=True)

# Assignment texts
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

# Function to reconstruct text with lower threshold
def reconstruct_with_sbert(text, threshold=0.5):
    sentences = sent_tokenize(text)
    reconstructed = []

    for sent in sentences:
        embedding = model.encode(sent, convert_to_tensor=True)
        hits = util.semantic_search(embedding, paraphrase_embeddings, top_k=1)
        best = hits[0][0]
        if best['score'] > threshold:
            reconstructed.append(paraphrase_candidates[best['corpus_id']])
        else:
            reconstructed.append(sent)

    return " ".join(reconstructed)

# Reconstruct both texts
reconstructed1 = reconstruct_with_sbert(text1)
reconstructed2 = reconstruct_with_sbert(text2)

# Save output
output_path = "reconstructed_texts_pipeline1_sbert_paws.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("Reconstructed Text 1:\n")
    f.write(reconstructed1 + "\n\n")
    f.write("Reconstructed Text 2:\n")
    f.write(reconstructed2 + "\n")

with open(output_path, "r", encoding="utf-8") as f:
    output_content = f.read()

output_content
print("✅ Reconstruction complete. Output saved to:")
print(f"   {os.path.abspath(output_path)}")