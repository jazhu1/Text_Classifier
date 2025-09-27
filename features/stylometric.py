import numpy as np
from collections import Counter
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade

def extract_stylometric_features(text: str) -> dict:
    """Basic writing-style and readability features."""
    words = text.split()
    sentences = nltk.sent_tokenize(text)
    return {
        # Length features
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0.0,
        'avg_sentence_length': (len(words) / len(sentences)) if sentences else 0.0,
        'text_length': len(text),
        'word_count': len(words),
        'sentence_count': len(sentences),
        # Lexical diversity
        'unique_words_ratio': (len(set(words)) / len(words)) if words else 0.0,
        'hapax_legomena_ratio': (sum(1 for _, c in Counter(words).items() if c == 1) / len(words)) if words else 0.0,
        # Punctuation patterns
        'comma_ratio': (text.count(',') / len(words)) if words else 0.0,
        'period_ratio': (text.count('.') / len(words)) if words else 0.0,
        'exclamation_ratio': (text.count('!') / len(words)) if words else 0.0,
        'question_ratio': (text.count('?') / len(words)) if words else 0.0,
        'semicolon_ratio': (text.count(';') / len(words)) if words else 0.0,
        # Readability
        'flesch_reading_ease': flesch_reading_ease(text) if len(text) > 10 else 0.0,
        'flesch_kincaid_grade': flesch_kincaid_grade(text) if len(text) > 10 else 0.0,
        # Case & digits
        'uppercase_ratio': (sum(1 for c in text if c.isupper()) / len(text)) if text else 0.0,
        'digit_ratio': (sum(1 for c in text if c.isdigit()) / len(text)) if text else 0.0,
        # Sentence complexity
        'complex_sentences_ratio': (sum(1 for s in sentences if len(s.split()) > 20) / len(sentences)) if sentences else 0.0,
        'short_sentences_ratio': (sum(1 for s in sentences if len(s.split()) < 10) / len(sentences)) if sentences else 0.0,
    }
