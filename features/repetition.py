import numpy as np
from collections import Counter
import nltk

def calculate_sentence_similarity(sentences):
    """Average Jaccard similarity between consecutive sentences."""
    if len(sentences) < 2:
        return 0.0
    sims = []
    for i in range(len(sentences) - 1):
        s1_words = set(sentences[i].lower().split())
        s2_words = set(sentences[i + 1].lower().split())
        if not s1_words or not s2_words:
            continue
        sims.append(len(s1_words & s2_words) / len(s1_words | s2_words))
    return float(np.mean(sims)) if sims else 0.0

def count_consecutive_duplicates(words):
    return sum(words[i] == words[i + 1] for i in range(len(words) - 1))

def extract_repetition_features(text: str) -> dict:
    """Repetition, redundancy, and adjacency patterns."""
    words = text.lower().split()
    sentences = nltk.sent_tokenize(text)
    return {
        'word_repetition_ratio': ((len(words) - len(set(words))) / len(words)) if words else 0.0,
        'most_common_word_freq': (Counter(words).most_common(1)[0][1] / len(words)) if words else 0.0,
        'avg_sentence_similarity': calculate_sentence_similarity(sentences),
        'consecutive_duplicates': (count_consecutive_duplicates(words) / len(words)) if words else 0.0,
    }
