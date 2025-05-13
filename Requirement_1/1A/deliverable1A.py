# generalized_reconstructor.py

"""
Deliverable 1A
"""

import re
import nltk
from nltk import word_tokenize, pos_tag

# One-time resource setup (safe to re-run)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng")

class Rule:
    def __init__(self, name, apply_func):
        self.name = name
        self.apply = apply_func

    def run(self, sentence):
        new_sentence = self.apply(sentence)
        triggered = new_sentence != sentence
        return new_sentence, triggered

class ReconstructionPipeline:
    def __init__(self, rules):
        self.rules = rules

    def reconstruct(self, sentence):
        applied_rules = []
        for rule in self.rules:
            sentence_new, triggered = rule.run(sentence)
            if triggered:
                applied_rules.append(rule.name)
                sentence = sentence_new
        return clean_spacing(sentence), applied_rules

# === Util ===

def clean_spacing(text):
    return re.sub(r'\s+([.,!?;:])', r'\1', text)

# === General and Specific Rules ===

def add_missing_subject(sentence):
    if re.match(r'^(Hope|Want|Need|Wish)\b', sentence, re.IGNORECASE):
        return "I " + sentence[0].lower() + sentence[1:]
    return sentence

def remove_duplicate_words(sentence):
    words = sentence.split()
    cleaned = []
    for i, word in enumerate(words):
        if i == 0 or word.lower() != words[i - 1].lower():
            cleaned.append(word)
    return ' '.join(cleaned)

def clarify_contract_checking(sentence):
    """[Regex] Rewrite domain-specific collocations like 'contract checking' → 'contract review'"""
    replacements = {
        "contract checking": "contract review",
        "document check": "document review",
        "paper correction": "paper revision"
    }
    lower_sentence = sentence.lower()
    for phrase, replacement in replacements.items():
        if phrase in lower_sentence:
            sentence = re.sub(re.escape(phrase), replacement, sentence, flags=re.IGNORECASE)
    return sentence

def fix_verb_editing_construction(sentence):
    return re.sub(r'plans (for|on) the editing', 'plans to edit', sentence, flags=re.IGNORECASE)

def simplify_final_wishes_phrase(sentence):
    """[Regex] Generalize 'which is one of my final wishes' → 'as I had hoped'"""
    patterns = [
        r'which is one of my (final|sincere|strong) wishes',
        r'that is one of my (final|sincere|strong) wishes',
        r'which I (sincerely|truly)? ?wish(ed)? for',
        r'which I (have)? ?been wishing for',
    ]
    replacement = 'as I had hoped'
    new_sentence = sentence
    for pattern in patterns:
        new_sentence, count = re.subn(pattern, replacement, new_sentence, flags=re.IGNORECASE)
        if count > 0:
            break
    return new_sentence
# === NLTK-Enhanced Rules ===

def fix_subject_verb_agreement_nltk(sentence):
    from nltk import word_tokenize, pos_tag

    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    modified = False

    skip_tags = {'RB', 'RBR', 'RBS'}  # adverbs
    common_base_verbs = {"plan", "check", "edit", "review", "submit"}

    i = 0
    while i < len(tagged) - 1:
        word1, tag1 = tagged[i]

        if tag1 in ('NN', 'PRP') and word1.lower() in ('he', 'she', 'it', 'doctor'):
            for j in range(i + 1, min(i + 4, len(tagged))):
                word2, tag2 = tagged[j]

                if tag2 in skip_tags:
                    continue

                if tag2 == 'VB':
                    tokens[j] = word2 + 's'
                    modified = True
                    break

                elif tag2 == 'VBG' and word2.endswith('ing'):
                    tokens[j] = word2[:-3] + 's'
                    modified = True
                    break

                elif tag2 == 'NN' and word2.lower() in common_base_verbs:
                    tokens[j] = word2 + 's'
                    modified = True
                    break

                break

        i += 1

    return ' '.join(tokens) if modified else sentence

def fix_noun_modifier_order_nltk(sentence):
    from nltk import word_tokenize, pos_tag

    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    modified = False

    transformed = []

    i = 0
    while i < len(tagged) - 2:
        t1, t2, t3 = tagged[i:i+3]
        word1, tag1 = t1
        word2, tag2 = t2
        word3, tag3 = t3

        if tag1.startswith('NN') and tag2.startswith('NN') and tag3.startswith('NN'):
            if word3.lower() in {"edit", "review", "check", "submission", "approval", "revision", "update"}:
                # Make gerund form safely
                if word3.endswith('e'):
                    verbing = word3[:-1] + 'ing'
                else:
                    verbing = word3 + 'ing'

                transformed.append((i, i + 3, f"{verbing} the {word1} {word2}"))
                i += 3
                modified = True
                continue
        i += 1

    if not modified:
        return sentence

    new_tokens = tokens[:]
    for start, end, repl in reversed(transformed):
        new_tokens[start:end] = [repl]

    return ' '.join(new_tokens)

def compress_overqualified_nouns_ntlk(sentence):
    from nltk import word_tokenize, pos_tag

    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    modified = False

    redundant_pairs = {
        ("finalized", "approved"),
        ("confirmed", "approved"),
        ("confirmed", "checked"),
        ("verified", "checked"),
        ("fully", "complete"),
        ("completely", "finished"),
    }

    new_tokens = []
    i = 0
    while i < len(tagged) - 1:
        word1, tag1 = tagged[i]
        word2, tag2 = tagged[i + 1]

        lower_pair = (word1.lower(), word2.lower())

        if lower_pair in redundant_pairs:
            new_tokens.append(word2)  # keep only the second word
            i += 2
            modified = True
        else:
            new_tokens.append(word1)
            i += 1

    if i == len(tagged) - 1:
        new_tokens.append(tagged[-1][0])  # add last word if not processed

    return ' '.join(new_tokens) if modified else sentence

def disambiguate_nominal_verb_noun_nltk(sentence):
    from nltk import word_tokenize, pos_tag

    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    modified = False

    ambiguous_roots = {
        "edit", "review", "check", "submit", "update", "approve", "correct", "revise"
    }

    disambiguated = []

    i = 0
    while i < len(tagged) - 1:
        word1, tag1 = tagged[i]
        word2, tag2 = tagged[i + 1]

        if tag1 == 'NN' and tag2 == 'NN' and word1.lower() in ambiguous_roots:
            # Convert root verb to -ing form
            if word1.endswith('e'):
                verbing = word1[:-1] + 'ing'
            else:
                verbing = word1 + 'ing'

            disambiguated.append((i, i + 2, f"{verbing} {word2}"))
            i += 2
            modified = True
        else:
            i += 1

    if not modified:
        return sentence

    new_tokens = tokens[:]
    for start, end, repl in reversed(disambiguated):
        new_tokens[start:end] = [repl]

    return ' '.join(new_tokens)

def shorten_double_modals_nltk(sentence):
    from nltk import word_tokenize, pos_tag
    """[NLTK] Remove redundant modal pairs like 'might can' → 'can'"""
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    modified = False
    new_tokens = []
    i = 0
    while i < len(tagged) - 1:
        word1, tag1 = tagged[i]
        word2, tag2 = tagged[i + 1]
        if tag1 == 'MD' and tag2 == 'MD':
            new_tokens.append(word2)
            i += 2
            modified = True
        else:
            new_tokens.append(word1)
            i += 1
    if i == len(tagged) - 1:
        new_tokens.append(tagged[-1][0])
    return ' '.join(new_tokens) if modified else sentence

def simplify_politeness_nltk(sentence):
    """[NLTK] Simplify excessive politeness like 'kindly please' → 'please'"""
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    filtered_tokens = []
    modified = False
    seen_please = False
    for word, tag in tagged:
        if word.lower() in {'kindly', 'please'}:
            if seen_please:
                modified = True
                continue
            seen_please = True
            filtered_tokens.append("please")
            modified = True
        else:
            filtered_tokens.append(word)
    return ' '.join(filtered_tokens) if modified else sentence

def clean_fillers_nltk(sentence):
    """[NLTK] Remove fillers like 'actually', 'you know', 'in fact' from start of sentence"""
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    modified = False
    filler_starters = {"actually", "basically", "i mean", "you know", "in fact"}
    cleaned_tokens = []
    skip = False
    for i, (word, tag) in enumerate(tagged):
        lower_word = word.lower()
        if lower_word in filler_starters and (i == 0 or tagged[i - 1][0] in {",", "."}):
            modified = True
            skip = True
            continue
        if skip and word in {",", "."}:
            continue
        skip = False
        cleaned_tokens.append(word)
    return ' '.join(cleaned_tokens) if modified else sentence

def fix_article_usage_nltk(sentence):
    """[NLTK] Fix improper 'a/an' usage based on following noun's phonetics"""
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    modified = False
    new_tokens = []
    for i in range(len(tagged)):
        word, tag = tagged[i]
        if word.lower() == 'a' and i + 1 < len(tagged):
            next_word, next_tag = tagged[i + 1]
            if next_tag.startswith('NN') and next_word[0].lower() in 'aeiou':
                new_tokens.append('an')
                modified = True
            else:
                new_tokens.append(word)
        else:
            new_tokens.append(word)
    return ' '.join(new_tokens) if modified else sentence

def normalize_infinitives_nltk(sentence):
    """[NLTK] Correct malformed infinitives like 'you too, to VB' → 'you VB'"""
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    modified = False
    new_tokens = []
    i = 0
    while i < len(tagged) - 3:
        w1, t1 = tagged[i]
        w2, t2 = tagged[i + 1]
        w3, t3 = tagged[i + 2]
        w4, t4 = tagged[i + 3]
        if w1.lower() == 'you' and w2.lower() == 'too' and w3 in {',', 'to'} and t4 == 'VB':
            new_tokens.extend(['you', w4])
            i += 4
            modified = True
            continue
        new_tokens.append(w1)
        i += 1
    while i < len(tagged):
        new_tokens.append(tagged[i][0])
        i += 1
    return ' '.join(new_tokens) if modified else sentence

def fix_awkward_gratitude_nltk(sentence):
    """[NLTK] Transform 'Thank your message' → 'Thank you for the message'"""
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    modified = False
    new_tokens = []
    i = 0
    while i < len(tokens) - 2:
        if tokens[i].lower() == 'thank' and tokens[i + 1].lower() in {'your', 'the'} and tagged[i + 2][1].startswith('NN'):
            new_tokens.append('Thank')
            new_tokens.extend(['you', 'for', 'the', tokens[i + 2]])
            i += 3
            modified = True
        else:
            new_tokens.append(tokens[i])
            i += 1
    while i < len(tokens):
        new_tokens.append(tokens[i])
        i += 1
    return ' '.join(new_tokens) if modified else sentence

# === Rule Set ===

rules = [
    Rule("AddMissingSubject", add_missing_subject),
    Rule("RemoveDuplicateWords", remove_duplicate_words),
    Rule("SimplifyPolitenessNLTK", simplify_politeness_nltk),
    Rule("FixArticlesNLTK", fix_article_usage_nltk),
    Rule("ShortenDoubleModalsNLTK", shorten_double_modals_nltk),
    Rule("CompressOverqualifiedNounsNLTK", compress_overqualified_nouns_ntlk),
    Rule("FixAwkwardGratitudeNLTK", fix_awkward_gratitude_nltk),
    Rule("ClarifyContractChecking", clarify_contract_checking),
    Rule("SimplifyFinalWishes", simplify_final_wishes_phrase),
    Rule("NormalizeInfinitivesNLTK", normalize_infinitives_nltk),
    Rule("FixNounModifierOrderNLTK", fix_noun_modifier_order_nltk),
    Rule("CleanFillersNLTK", clean_fillers_nltk),
    Rule("FixEditingVerbConstruction", fix_verb_editing_construction),
    Rule("FixVerbAgreementNLTK", fix_subject_verb_agreement_nltk),
    Rule("DisambiguateNominalVerbNounNLTK", disambiguate_nominal_verb_noun_nltk),
]

pipeline = ReconstructionPipeline(rules)

if __name__ == "__main__":
    sentences = [
        "Hope you too, to enjoy it as my deepest wishes.",
        "Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again.",
        "Thank your message to show our words to the doctor, as his next contract checking, to all of us.",
        "We might can go tomorrow if the document gets finalized approved.",
    ]

    print("\n === Sentence Reconstruction === \n")
    for i, s in enumerate(sentences):
        output, applied = pipeline.reconstruct(s)
        print(f"Original {i+1}: {s}")
        print(f"Reconstructed {i+1}: {output}")
        print(f"Rules Applied: {applied}")
        print()
