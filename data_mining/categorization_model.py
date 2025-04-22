import logging
from pathlib import Path
from typing import Any, List, Tuple, Optional

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def train_and_evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 8,
    n_splits: int = 5
) -> Tuple[Optional[Any], np.ndarray, np.ndarray]:

    try:
        X_res, y_res = SMOTE().fit_resample(X, y)
        logger.info("Data resampled using SMOTE: %d -> %d samples", len(y), len(y_res))

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        models = {
            'RandomForest': RandomForestClassifier(),
            'SVM':          RandomForestClassifier(),
            'XGBoost':      xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }

        best_model = None
        best_accuracy = 0.0
        best_y_pred = np.array([])
        best_y_test = np.array([])

        for name, model in models.items():
            fold = 1
            for train_idx, test_idx in skf.split(X_res, y_res):
                X_train, X_test = X_res[train_idx], X_res[test_idx]
                y_train, y_test = y_res[train_idx], y_res[test_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                logger.info("%s fold %d accuracy: %.4f", name, fold, acc)
                fold += 1
                
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model = model
                    best_y_pred = y_pred
                    best_y_test = y_test

                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    ax=ax
                )
                ax.set_title(f'Confusion Matrix – {name}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                reports = Path('reports')
                reports.mkdir(exist_ok=True)
                path = reports / f'confusion_matrix_{name}.png'
                fig.tight_layout()
                fig.savefig(path)
                plt.close(fig)
                logger.info("✔ Saved confusion matrix: %s", path)

        if best_model is None:
            logger.error("Nenhum modelo treinado com sucesso.")
            return None, np.array([]), np.array([])

        logger.info("Best model: %s with accuracy %.4f", type(best_model).__name__, best_accuracy)
        return best_model, best_y_pred, best_y_test

    except Exception as e:
        logger.exception("Error during model training and evaluation: %s", e)
        return None, np.array([]), np.array([])
