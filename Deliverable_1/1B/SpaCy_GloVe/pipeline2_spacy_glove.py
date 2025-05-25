import os
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from gensim.downloader import load as gensim_load
from scipy.spatial.distance import cosine
import numpy as np

nltk.download("punkt")

nlp = spacy.load("en_core_web_lg")
glove = gensim_load("glove-wiki-gigaword-300")

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

ALLOWED_NOUN_DEPS = {"nsubj", "dobj", "pobj", "attr"}
ALLOWED_VERB_DEPS = {"ROOT", "xcomp", "ccomp", "conj"}

def looks_like_verb_noun(token):
    if token.pos_ == "NOUN" and token.dep_ in ALLOWED_NOUN_DEPS:
        try:
            guess = nlp(token.lemma_)[0]
            return guess.pos_ == "VERB"
        except Exception:
            return False
    return False

def find_glove_alternative(word, target_pos):
    if word not in glove:
        return None
    word_vec = glove[word]
    best_word = None
    best_score = -1
    for cand in glove.index_to_key[:10000]:
        if cand == word:
            continue
        if abs(len(cand) - len(word)) > 5:
            continue
        sim = 1 - cosine(word_vec, glove[cand])
        if sim > best_score:
            try:
                cand_pos = nlp(cand)[0].pos_
                if cand_pos == target_pos:
                    best_score = sim
                    best_word = cand
            except:
                continue
    return best_word if best_score > 0.7 else None

def spacy_rule_based_glove_reconstruct(text):
    sentences = sent_tokenize(text)
    reconstructed = []
    log = []

    for idx, sent in enumerate(sentences):
        doc = nlp(sent)
        modified = sent
        changed = False

        # Rule 1: Noun used where verb may be more appropriate (with dependency filtering)
        for token in doc:
            if looks_like_verb_noun(token):
                alt = find_glove_alternative(token.text.lower(), token.pos_)
                if alt and alt != token.text.lower():
                    modified = modified.replace(token.text, alt)
                    changed = True

        # Rule 2: Passive structure with any past participle verb
        for token in doc:
            if token.pos_ == "AUX" and token.lemma_ == "be":
                for child in token.head.subtree:
                    if child.tag_ == "VBN" and child.pos_ == "VERB" and child.dep_ != "auxpass":
                        alt = find_glove_alternative(child.lemma_, child.pos_)
                        if alt and alt != child.lemma_:
                            changed = True

        if changed:
            reconstructed.append(modified)
        else:
            reconstructed.append(sent)

    return " ".join(reconstructed), log

# Process both texts
reconstructed1, log1 = spacy_rule_based_glove_reconstruct(text1)
reconstructed2, log2 = spacy_rule_based_glove_reconstruct(text2)

# Save output
output_path = "reconstructed_texts_pipeline2_spacy_glove.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("Reconstructed Text 1:\n")
    f.write(reconstructed1 + "\n\n")
    f.write("Reconstructed Text 2:\n")
    f.write(reconstructed2 + "\n\n")

print("✅ SpaCy + GloVe rule-based repair (dependency-filtered) complete. Output saved to:")
print(os.path.abspath(output_path))

