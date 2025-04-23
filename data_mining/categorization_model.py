import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from scipy.stats import uniform, randint

logger = logging.getLogger(__name__)

def train_and_evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 3,
    n_iter: int = 20
) -> Tuple[Optional[Any], np.ndarray, np.ndarray]:

    try:
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        logger.info("✔ Data resampled: %d → %d samples", len(y), len(y_res))

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        lgbm = lgb.LGBMClassifier(random_state=42)
        dist_lgb = {
            'num_leaves': randint(20, 100),
            'n_estimators': randint(50, 200),
            'learning_rate': uniform(0.01, 0.2)
        }
        rand_lgb = RandomizedSearchCV(
            lgbm, dist_lgb, n_iter=n_iter, cv=skf,
            scoring='accuracy', n_jobs=-1, random_state=42
        )
        rand_lgb.fit(X_res, y_res)
        best_lgb = rand_lgb.best_estimator_
        lgb_score = rand_lgb.best_score_
        logger.info("✔ LightGBM best: %.4f with %s", lgb_score, rand_lgb.best_params_)

        svc = SVC(probability=True, random_state=42)
        dist_svm = {
            'C': uniform(0.1, 10),
            'gamma': ['scale'],
            'kernel': ['rbf']
        }
        rand_svm = RandomizedSearchCV(
            svc, dist_svm, n_iter=10, cv=skf,
            scoring='accuracy', n_jobs=-1, random_state=42
        )
        rand_svm.fit(X_res, y_res)
        best_svm = rand_svm.best_estimator_
        svm_score = rand_svm.best_score_
        logger.info("✔ SVM best: %.4f with %s", svm_score, rand_svm.best_params_)

        weights = [1.0, lgb_score / max(lgb_score, svm_score), svm_score / max(lgb_score, svm_score)]
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('lgb', best_lgb), ('svm', best_svm)],
            voting='soft',
            weights=weights
        )
        ensemble.fit(X_res, y_res)
        logger.info("✔ Ensemble created with weights %s", weights)

        y_pred = ensemble.predict(X)
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix – Ensemble')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        reports = Path('reports')
        reports.mkdir(exist_ok=True)
        path = reports / 'confusion_matrix_ensemble.png'
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        logger.info("✔ Saved ensemble confusion matrix: %s", path)

        return ensemble, y_pred, y

    except Exception as e:
        logger.exception("❌ Error in training/evaluation: %s", e)
        return None, np.array([]), np.array([])
