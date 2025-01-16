import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

def train_and_evaluate_model(X, y, word2vec_model, bert_model):
    try:
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logger.info("Data resampled using SMOTE.")

        lasso = Lasso(alpha=0.01)
        selector = SelectFromModel(lasso)

        skf = StratifiedKFold(n_splits=5)

        models = {
            'RandomForest': RandomForestClassifier(),
            'SVM': SVC(),
            'XGBoost': xgb.XGBClassifier()
        }

        best_model = None
        best_accuracy = 0
        best_y_pred = None

        for model_name, model in models.items():
            pipeline = Pipeline([
                ('feature_selection', selector),
                ('classification', model)
            ])
            
            all_y_pred = np.array([])  # Armazena todas as previsões
            all_y_test = np.array([])  # Armazena todos os rótulos verdadeiros

            for train_index, test_index in skf.split(X_resampled, y_resampled):
                X_train, X_test = X_resampled[train_index], X_resampled[test_index]
                y_train, y_test = y_resampled[train_index], y_resampled[test_index]
                
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                all_y_pred = np.concatenate([all_y_pred, y_pred])
                all_y_test = np.concatenate([all_y_test, y_test])

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_y_pred = y_pred

                logger.info(f"Results for {model_name}:")
                logger.info(classification_report(y_test, y_pred))

                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(10, 7))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix for {model_name}')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                Path('reports').mkdir(parents=True, exist_ok=True)
                plt.savefig(f'reports/confusion_matrix_{model_name}.png')
                plt.close()

        # Retornar todas as previsões concatenadas
        return all_y_pred

    except Exception as e:
        logger.error(f"Error during model training and evaluation: {e}")
        return None
