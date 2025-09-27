from .stylometric import extract_stylometric_features
from .repetition import extract_repetition_features
from .enhanced import extract_enhanced_features
from .semantic import extract_advanced_semantic_features

def extract_all_advanced_features(text: str) -> dict:
    """Combine all feature extractors into a single dict."""
    feats = {}
    feats.update(extract_stylometric_features(text))
    feats.update(extract_repetition_features(text))
    feats.update(extract_enhanced_features(text))
    feats.update(extract_advanced_semantic_features(text))
    return feats
