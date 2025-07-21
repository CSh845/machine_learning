# models/models.py

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

def get_mlp_model():
    model = MLPClassifier(random_state=123)
    params = {
        "hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "activation": ['tanh', 'relu'],
        "solver": ['sgd', 'adam'],
        "learning_rate_init": [0.001, 0.01],
        "max_iter": [10000],
        "early_stopping": [True, False],
        "validation_fraction": [0.1, 0.2]
    }
    return model, params

def get_rf_model():
    model = RandomForestClassifier(random_state=123)
    params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    }
    return model, params

def get_svm_model():
    model = SVC(probability=True, random_state=123)
    params = {
        "C": [0.1, 1, 10, 100],
        "gamma": [0.001, 0.01, 0.1, 1],
        "kernel": ['linear', 'rbf']
    }
    return model, params

def get_xgb_model():
    model = xgb.XGBClassifier(eval_metric='logloss', random_state=123)
    params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
    return model, params

def get_all_models():
    return {
        'MLP': get_mlp_model,
        'RandomForest': get_rf_model,
        'SVM': get_svm_model,
        'XGBoost': get_xgb_model
    }
