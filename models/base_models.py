from sklearn.ensemble import RandomForestClassifier

def get_base_models(y_train, xgb=None, lgb=None):
    """Return list of (name, estimator) tuples for stacking."""
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = (neg / pos) if pos else 1.0

    models = [
        ('rf_balanced', RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_split=3,
            class_weight='balanced', random_state=42, n_jobs=-1
        ))
    ]
    if xgb is not None:
        models.append(('xgb', xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42, n_jobs=-1, eval_metric='logloss'
        )))
    if lgb is not None:
        models.append(('lgb', lgb.LGBMClassifier(
            n_estimators=300, max_depth=10, learning_rate=0.05,
            feature_fraction=0.8, bagging_fraction=0.8,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
        )))
    return models
