import re
import numpy as np
import nltk

def count_common_words(text, words):
    common_words = {'the','be','to','of','and','a','in','that','have','i','it','for','not','on','with','he','as','you','do','at','this','but','his','by','from','they','we','say','her','she','or','an','will','my','one','all','would','there','their'}
    return (sum(1 for w in words if w.lower() in common_words) / len(words)) if words else 0.0

def count_rare_words(text, words):
    if not words:
        return 0.0
    rare_count = sum(1 for w in words if len(w) > 8 and w.isalpha())
    return rare_count / len(words)

def count_connectors(text):
    connectors = ['however','moreover','furthermore','therefore','consequently','nevertheless','nonetheless','meanwhile','subsequently','additionally','specifically','particularly','especially','notably','thus','hence']
    t = text.lower()
    return sum(t.count(c) for c in connectors)

def count_transition_words(text):
    words = ['firstly','secondly','thirdly','finally','lastly','initially','subsequently','ultimately','essentially','basically','generally','typically','usually','commonly','frequently','regularly']
    t = text.lower()
    return sum(t.count(w) for w in words)

def count_contractions(text):
    contractions = ["don't","won't","can't","shouldn't","wouldn't","couldn't","isn't","aren't","wasn't","weren't","haven't","hasn't","hadn't","i'm","you're","he's","she's","it's","we're","they're","i've","you've","we've","they've","i'll","you'll","he'll","she'll","we'll","they'll","i'd","you'd","he'd","she'd","we'd","they'd"]
    t = text.lower()
    return sum(t.count(c) for c in contractions)

def calculate_sentence_starts_diversity(sentences):
    if not sentences:
        return 0.0
    starts = []
    for s in sentences:
        words = s.strip().split()
        if words:
            starts.append(words[0].lower())
    return (len(set(starts)) / len(starts)) if starts else 0.0

def extract_enhanced_features(text: str) -> dict:
    """Intermediate-level features likely helpful for AI detection."""
    words = text.split()
    sentences = nltk.sent_tokenize(text)
    puncts = re.findall(r'[^\w\s]', text)
    punct_diversity = (len(set(puncts)) / max(1, len(puncts))) if puncts else 0.0
    sent_lengths = [len(s.split()) for s in sentences] if sentences else []
    return {
        'sentence_length_std': float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0.0,
        'sentence_length_range': (max(sent_lengths) - min(sent_lengths)) if sent_lengths else 0.0,
        'long_words_ratio': (sum(1 for w in words if len(w) > 6) / len(words)) if words else 0.0,
        'short_words_ratio': (sum(1 for w in words if len(w) <= 3) / len(words)) if words else 0.0,
        'punctuation_diversity': punct_diversity,
        'quotes_ratio': (text.count('"') / len(words)) if words else 0.0,
        'colon_ratio': (text.count(':') / len(words)) if words else 0.0,
        'parentheses_ratio': (text.count('(') / len(words)) if words else 0.0,
        'questions_ratio': (sum(1 for s in sentences if s.strip().endswith('?')) / len(sentences)) if sentences else 0.0,
        'exclamations_ratio': (sum(1 for s in sentences if s.strip().endswith('!')) / len(sentences)) if sentences else 0.0,
        'common_words_ratio': count_common_words(text, words),
        'rare_words_ratio': count_rare_words(text, words),
        'connectors_ratio': (count_connectors(text) / len(words)) if words else 0.0,
        'transition_words_ratio': (count_transition_words(text) / len(words)) if words else 0.0,
        'contraction_ratio': (count_contractions(text) / len(words)) if words else 0.0,
        'sentence_starts_diversity': calculate_sentence_starts_diversity(sentences),
        'word_length_variance': float(np.var([len(w) for w in words])) if len(words) > 1 else 0.0,
    }
