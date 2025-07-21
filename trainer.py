# utils/trainer.py

from sklearn.model_selection import GridSearchCV
import xgboost as xgb

def train_model(X_train, y_train, model, params, cv_splitter, scoring='roc_auc', verbose_level=2, n_jobs=-1):
    # Special handling for XGBoost models to avoid sklearn_tags issue
    if isinstance(model, xgb.XGBClassifier):
        # Ensure proper initialization of XGBoost classifier
        model = xgb.XGBClassifier(
            use_label_encoder=False,  # Avoid deprecation warning
            eval_metric='logloss',    # Default metric for classification
            random_state=123          # For reproducibility
        )
    
    gc = GridSearchCV(
        model,
        param_grid=params,
        cv=cv_splitter,
        scoring=scoring,
        return_train_score=True,
        verbose=verbose_level,
        n_jobs=n_jobs
    )
    gc.fit(X_train, y_train)
    return gc
