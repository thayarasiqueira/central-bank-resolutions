import logging
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import numpy as np
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.optimizers import Adam
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

        for model_name, model in models.items():
            pipeline = Pipeline([
                ('feature_selection', selector),
                ('classification', model)
            ])
            
            for train_index, test_index in skf.split(X_resampled, y_resampled):
                X_train, X_test = X_resampled[train_index], X_resampled[test_index]
                y_train, y_test = y_resampled[train_index], y_resampled[test_index]
                
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
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

        keras_model = Sequential([
            Dense(128, activation='relu', input_shape=(X.shape[1],)),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        keras_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled)
        keras_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        y_pred_keras = (keras_model.predict(X_test) > 0.5).astype("int32")
        logger.info("Results for Keras Model:")
        logger.info(classification_report(y_test, y_pred_keras))

        cm_keras = confusion_matrix(y_test, y_pred_keras)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_keras, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix for Keras Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('reports/confusion_matrix_keras.png')
        plt.close()

    except Exception as e:
        logger.error(f"Error during model training and evaluation: {e}")
