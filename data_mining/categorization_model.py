import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble      import RandomForestClassifier
from sklearn.svm           import SVC
from sklearn.metrics       import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

def train_and_evaluate_model(X, y, word2vec_model=None, bert_model=None):
    try:
        X_res, y_res = SMOTE().fit_resample(X, y)
        logger.info("Data resampled using SMOTE.")

        skf = StratifiedKFold(n_splits=5)
        models = {
            'RandomForest': RandomForestClassifier(),
            'SVM':          SVC(),
            'XGBoost':      xgb.XGBClassifier()
        }

        best_model    = None
        best_accuracy = 0.0
        best_y_pred   = None
        best_y_test   = None

        for name, model in models.items():
            for train_i, test_i in skf.split(X_res, y_res):
                X_tr, X_te = X_res[train_i], X_res[test_i]
                y_tr, y_te = y_res[train_i], y_res[test_i]

                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                acc    = accuracy_score(y_te, y_pred)
                logger.info(f"{name} fold accuracy: {acc:.4f}")
                logger.info(classification_report(y_te, y_pred))

                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model    = model
                    best_y_pred   = y_pred
                    best_y_test   = y_te

                cm = confusion_matrix(y_te, y_pred)
                Path('reports').mkdir(exist_ok=True)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix – {name}')
                plt.savefig(f'reports/confusion_{name}.png')
                plt.close()

        if best_model is None:
            logger.error("Nenhum modelo treinado com sucesso.")
            return None, None, None

        logger.info(f"Best model: {best_model.__class__.__name__} (acc={best_accuracy:.4f})")
        return best_model, best_y_pred, best_y_test

    except Exception as e:
        logger.error(f"Error in train_and_evaluate_model: {e}")
        return None, None, None
