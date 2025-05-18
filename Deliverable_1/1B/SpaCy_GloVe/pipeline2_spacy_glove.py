import os
import spacy
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

# Load SpaCy model with GloVe vectors
nlp = spacy.load("en_core_web_lg")

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

# Phrase-level replacements (automated corrections)
phrase_replacements = {
    "final discuss": "final discussion",
    "bit delay": "a bit of a delay",
    "to show me, this": "to show me this",
    "I am very appreciated": "I greatly appreciate",
    "before he sending again": "before he sends it again",
    "although bit delay and less communication": "although there has been a bit of a delay and less communication",
}

# Function to apply enhancements
def spacy_reconstruct(text):
    sentences = sent_tokenize(text)
    reconstructed = []

    for sent in sentences:
        original = sent
        doc = nlp(sent)

        # Apply phrase-level replacements
        for bad_phrase, good_phrase in phrase_replacements.items():
            if bad_phrase in sent:
                sent = sent.replace(bad_phrase, good_phrase)

        # Fix low-confidence fragments
        if len(doc) > 6 and not any(tok.dep_ == "ROOT" for tok in doc):
            sent = f"This sentence was unclear: {original.strip()}"

        reconstructed.append(sent)

    return " ".join(reconstructed)

# Reconstruct both texts
reconstructed1 = spacy_reconstruct(text1)
reconstructed2 = spacy_reconstruct(text2)

# Save output
output_path = "reconstructed_texts_pipeline2_spacy_glove_refined.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("Reconstructed Text 1:\n")
    f.write(reconstructed1 + "\n\n")
    f.write("Reconstructed Text 2:\n")
    f.write(reconstructed2 + "\n")

print("✅ Refined SpaCy + GloVe pipeline complete. Output saved to:")
print(os.path.abspath(output_path))
