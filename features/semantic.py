import numpy as np
from collections import Counter
import nltk

def count_discourse_markers(text):
    markers = ['first','second','third','next','then','finally','in conclusion','to summarize','in summary','in addition','furthermore','moreover','on the other hand','in contrast','however','nevertheless','meanwhile']
    t = text.lower()
    return sum(t.count(m) for m in markers)

def count_hedging_words(text):
    hedging = ['might','may','could','would','perhaps','possibly','probably','likely','unlikely','seems','appears','suggests','indicates','tend to','tends to','generally','usually','often','sometimes']
    t = text.lower()
    return sum(t.count(h) for h in hedging)

def count_certainty_words(text):
    certainty = ['definitely','certainly','absolutely','clearly','obviously','undoubtedly','without doubt','always','never','must','will','shall','surely','indeed']
    t = text.lower()
    return sum(t.count(w) for w in certainty)

def count_first_person_pronouns(text):
    first_person = {'i','me','my','mine','myself','we','us','our','ours','ourselves'}
    return sum(1 for w in text.lower().split() if w in first_person)

def count_second_person_pronouns(text):
    second_person = {'you','your','yours','yourself','yourselves'}
    return sum(1 for w in text.lower().split() if w in second_person)

def count_emotional_words(text):
    emotional = {'love','hate','amazing','terrible','wonderful','awful','fantastic','horrible','great','bad','excellent','poor','beautiful','ugly','exciting','boring','interesting','dull'}
    return sum(1 for w in text.lower().split() if w in emotional)

def calculate_sentence_type_diversity(sentences):
    if not sentences:
        return 0.0
    types = []
    for s in sentences:
        s = s.strip()
        if s.endswith('?'):
            types.append('question')
        elif s.endswith('!'):
            types.append('exclamation')
        else:
            types.append('statement')
    return (len(set(types)) / len(types)) if types else 0.0

def estimate_topic_coherence(sentences):
    if len(sentences) < 2:
        return 0.0
    overlaps = []
    for i in range(len(sentences) - 1):
        w1 = set(sentences[i].lower().split())
        w2 = set(sentences[i + 1].lower().split())
        if w1 and w2:
            overlaps.append(len(w1 & w2) / len(w1 | w2))
    return float(np.mean(overlaps)) if overlaps else 0.0

def calculate_lexical_cohesion(sentences):
    if len(sentences) < 2:
        return 0.0
    all_words = []
    for s in sentences:
        all_words.extend(s.lower().split())
    if not all_words:
        return 0.0
    freq = Counter(all_words)
    repeated = sum(1 for c in freq.values() if c > 1)
    return (repeated / len(set(all_words))) if all_words else 0.0

def estimate_clauses_per_sentence(sentences):
    if not sentences:
        return 0.0
    total = 0
    indicators = [',','and','but','or','because','since','while','although','though','if','when','where','who','which','that']
    for s in sentences:
        cnt = 1
        ls = s.lower()
        for ind in indicators:
            cnt += ls.count(ind)
        total += cnt
    return total / len(sentences)

def count_coordination(text):
    coord = {'and','but','or','nor','for','so','yet'}
    return sum(1 for w in text.lower().split() if w in coord)

def count_subordination(text):
    sub = {'because','since','although','though','while','if','when','where','before','after','until','unless','whereas'}
    return sum(1 for w in text.lower().split() if w in sub)

def extract_advanced_semantic_features(text: str) -> dict:
    sentences = nltk.sent_tokenize(text)
    words = text.split()
    return {
        'discourse_markers_ratio': (count_discourse_markers(text) / len(words)) if words else 0.0,
        'hedging_ratio': (count_hedging_words(text) / len(words)) if words else 0.0,
        'certainty_ratio': (count_certainty_words(text) / len(words)) if words else 0.0,
        'first_person_ratio': (count_first_person_pronouns(text) / len(words)) if words else 0.0,
        'second_person_ratio': (count_second_person_pronouns(text) / len(words)) if words else 0.0,
        'emotional_words_ratio': (count_emotional_words(text) / len(words)) if words else 0.0,
        'sentence_type_diversity': calculate_sentence_type_diversity(sentences),
        'topic_coherence': estimate_topic_coherence(sentences),
        'lexical_cohesion': calculate_lexical_cohesion(sentences),
        'clauses_per_sentence': estimate_clauses_per_sentence(sentences),
        'coordination_ratio': (count_coordination(text) / len(sentences)) if sentences else 0.0,
        'subordination_ratio': (count_subordination(text) / len(sentences)) if sentences else 0.0,
    }
