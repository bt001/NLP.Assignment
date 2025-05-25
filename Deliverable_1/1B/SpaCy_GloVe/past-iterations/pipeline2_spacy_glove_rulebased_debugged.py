import os
import spacy
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

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

def spacy_rule_based_reconstruct(text):
    sentences = sent_tokenize(text)
    reconstructed = []
    log = []

    for idx, sent in enumerate(sentences):
        doc = nlp(sent)
        flagged = False

        # Rule 1: Misused verb-as-noun compounds
        for token in doc:
            if token.pos_ == "NOUN" and token.text in {"discuss", "talk", "request"}:
                if any(child.dep_ in {"amod", "compound", "poss"} for child in token.children):
                    log.append(f"R1: Suspicious noun derived from verb at sentence {idx+1}: '{token.text}'")
                    flagged = True

        # Rule 2: Passive structure with appreciated
        for token in doc:
            if token.pos_ == "AUX" and token.lemma_ == "be":
                for child in token.head.subtree:
                    if child.lemma_ == "appreciate" and child.tag_ == "VBN":
                        log.append(f"R2: Suspect passive structure at sentence {idx+1}: '{token.text} ... {child.text}'")
                        flagged = True

        if flagged:
            reconstructed.append(f"[FLAGGED] {sent}")
        else:
            reconstructed.append(sent)

    return " ".join(reconstructed), log

# Process both texts
reconstructed1, log1 = spacy_rule_based_reconstruct(text1)
reconstructed2, log2 = spacy_rule_based_reconstruct(text2)

# Save output
output_path = "reconstructed_texts_pipeline2_spacy_glove_rulebased_debugged.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("Reconstructed Text 1:\n")
    f.write(reconstructed1 + "\n\n")
    f.write("Reconstructed Text 2:\n")
    f.write(reconstructed2 + "\n\n")
    f.write("=== CHANGE LOG ===\n")
    f.write("\n".join(log1 + log2))

print("✅ Updated SpaCy rule-based pipeline with improved rules complete. Output saved to:")
print(os.path.abspath(output_path))
