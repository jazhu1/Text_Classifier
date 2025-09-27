import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from features.feature_extraction import extract_all_advanced_features
from models.base_models import get_base_models

def comprehensive_ultimate_analysis(df, text_column='text', label_column='label'):
    """Row-wise advanced feature extraction over a dataframe."""
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row[text_column]
        if not isinstance(text, str) or len(text.strip()) < 20:
            continue
        feats = extract_all_advanced_features(text)
        if label_column in df.columns:
            feats['label'] = row[label_column]
        feats['text_sample'] = text[:100] + '...' if len(text) > 100 else text
        results.append(feats)
    return pd.DataFrame(results)

def train_ultimate_ensemble_model(train_df, val_df, test_df, text_column='text', label_column='label'):
    print("=" * 70)
    print("ULTIMATE STACKING ENSEMBLE AI DETECTION PIPELINE")
    print("=" * 70)

    # Phase 1: Feature extraction
    print("\nPHASE 1: EXTRACTING ULTIMATE FEATURES")
    print("-" * 50)
    print("Extracting from training data...")
    train_features = comprehensive_ultimate_analysis(train_df, text_column, label_column)
    print("Extracting from validation data...")
    val_features = comprehensive_ultimate_analysis(val_df, text_column, label_column)

    # Phase 2: Prepare data
    print("\nPHASE 2: DATA PREPARATION")
    print("-" * 50)
    feature_cols = [c for c in train_features.columns if c not in ['label','text_sample']]
    print(f"Total features extracted: {len(feature_cols)}")
    X_train = train_features[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_features['label']
    X_val = val_features[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_val = val_features['label']

    # Feature selection
    print("Selecting top features...")
    selector = SelectKBest(score_func=f_classif, k=min(100, len(feature_cols)))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)
    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    print(f"Selected {len(selected_features)} top features")

    # Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_val_scaled = scaler.transform(X_val_sel)

    # Phase 3: Stacking ensemble
    print("\nPHASE 3: TRAINING STACKING ENSEMBLE")
    print("-" * 50)
    try:
        import xgboost as xgb
    except Exception:
        xgb = None
    try:
        import lightgbm as lgb
    except Exception:
        lgb = None

    base_models = get_base_models(y_train, xgb=xgb, lgb=lgb)
    meta_model = LogisticRegression(class_weight='balanced', C=0.5, random_state=42)
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=3,
        stack_method='predict_proba',
        n_jobs=-1
    )
    print("Fitting the Stacking Ensemble...")
    stacking_clf.fit(X_train_scaled, y_train)

    # Phase 4: Threshold tuning on validation
    print("\nPHASE 4: ENSEMBLE OPTIMIZATION")
    print("-" * 50)
    val_probs = stacking_clf.predict_proba(X_val_scaled)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0.0
    best_thr = 0.5
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t
    print(f"Optimal ensemble threshold: {best_thr:.3f}")
    final_val_preds = (val_probs >= best_thr).astype(int)
    val_accuracy = accuracy_score(y_val, final_val_preds)
    val_precision = precision_score(y_val, final_val_preds)
    val_recall = recall_score(y_val, final_val_preds)
    print(f"Validation F1 Score: {best_f1:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")

    # Phase 5: Test predictions
    print("\nPHASE 5: GENERATING TEST PREDICTIONS")
    print("-" * 50)
    # Build test feature frame with the same columns (fallback zeros for short/invalid texts)
    test_results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        text = row[text_column]
        if not isinstance(text, str) or len(text.strip()) < 20:
            feats = {col: 0 for col in feature_cols}
        else:
            feats = extract_all_advanced_features(text)
        test_results.append(feats)
    test_features = pd.DataFrame(test_results)
    X_test = test_features[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    X_test_sel = selector.transform(X_test)
    X_test_scaled = scaler.transform(X_test_sel)

    print("Making stacking ensemble predictions...")
    test_probs = stacking_clf.predict_proba(X_test_scaled)[:, 1]
    test_preds = (test_probs >= best_thr).astype(int)

    if "id" in test_df.columns:
        submission = pd.DataFrame({"id": test_df["id"], "label": test_preds})
    else:
        submission = pd.DataFrame({"id": test_df.index, "label": test_preds})

    submission.to_csv("stacking_ensemble_submission.csv", index=False)

    print("\n" + "=" * 70)
    print("STACKING PIPELINE COMPLETE")
    print("=" * 70)
    print("Submission saved as 'stacking_ensemble_submission.csv'")
    print(f"Validation F1 Score: {best_f1:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Ensemble Threshold: {best_thr:.3f}")
    print("Prediction distribution:")
    print(f"  Human (0): {(test_preds == 0).sum()}")
    print(f"  AI (1): {(test_preds == 1).sum()}")

    return submission, stacking_clf, scaler, selector, best_thr
