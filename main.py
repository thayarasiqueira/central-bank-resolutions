import sys
import argparse
import json
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, TFDistilBertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

from data_mining.preprocessing import preprocess_text
from data_analysis.complexity_analysis import calculate_complexity_metrics
from data_mining.categorization_model import train_and_evaluate_model
from data_analysis.validation import validate_sample
from data_analysis.statistical_analysis import analyze_complexity_vs_accuracy
from data_analysis.longitudinal_analysis import plot_trends
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH    = PROJECT_ROOT / 'data' / 'raw' / 'resolutions_data.json'
REPORTS_DIR  = PROJECT_ROOT / 'reports'
MODELS_DIR   = PROJECT_ROOT / 'models'
LOGS_DIR     = PROJECT_ROOT / 'logs'
LOG_FILE     = LOGS_DIR / 'pipeline.log'


def ensure_dirs():
    for d in (REPORTS_DIR, MODELS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def configure_logging():
    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def bert_vectorize_texts(texts, tokenizer, model, batch_size: int = 4, max_length: int = 128):
    logger = logging.getLogger(__name__)
    embeddings = []
    total = len(texts)
    logger.info(f"▶ Vectorizing {total} texts with DistilBERT in batches of {batch_size}")
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = texts[start:end]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='tf'
        )
        outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls_emb)
    result = np.vstack(embeddings)
    logger.info("✔ Finished DistilBERT vectorization")
    return result


def run_data_mining():
    logger = logging.getLogger(__name__)
    logger.info("=== Starting Data Mining ===")

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        resolutions = json.load(f)

    texts = []
    comp_metrics = []
    for r in resolutions:
        proc = preprocess_text(r.get('content', ''))
        cm = calculate_complexity_metrics(proc)
        texts.append(proc)
        comp_metrics.append(list(cm.values()))
    comp_arr = np.array(comp_metrics)
    logger.info("✔ Preprocessing and complexity metrics done")

    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
    X_tfidf = tfidf.fit_transform(texts).toarray()
    logger.info("✔ TF-IDF features extracted: %s", X_tfidf.shape)

    Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    X_bert = bert_vectorize_texts(texts, tokenizer, model)

    X = np.hstack([X_bert, comp_arr, X_tfidf])
    logger.info("✔ Combined features: X shape = %s", X.shape)

    cats = [r.get('category', f"cat_{i%2}") for i, r in enumerate(resolutions)]
    le = LabelEncoder().fit(cats)
    y = le.transform(cats)
    X_train, X_test, comp_train, comp_test, y_train, y_test = train_test_split(
        X, comp_arr, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info("✔ Data split: train=%d, test=%d", len(y_train), len(y_test))

    best_model, _, _ = train_and_evaluate_model(X_train, y_train)
    if best_model is None:
        logger.error("No best model returned")
        sys.exit(1)
    logger.info("✔ Best model trained on training set: %s", best_model.__class__.__name__)

    joblib.dump(best_model, MODELS_DIR / 'best_model.pkl')
    joblib.dump(le,        MODELS_DIR / 'label_encoder.pkl')
    joblib.dump(tfidf,     MODELS_DIR / 'tfidf_vectorizer.pkl')
    logger.info("✔ Saved model, label encoder and TF-IDF vectorizer")

    y_pred_test = best_model.predict(X_test)
    df_test = pd.DataFrame(comp_test, columns=[
        'avg_sentence_length','lexical_density','flesch_index','syntactic_depth'
    ])
    df_test['per_instance_accuracy'] = (y_pred_test == y_test).astype(int)
    metrics_csv = REPORTS_DIR / 'complexity_vs_accuracy_test.csv'
    df_test.to_csv(metrics_csv, index=False)
    logger.info("✔ Saved hold-out complexity_vs_accuracy_test.csv")

    cm = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix – Hold‑out')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    reports = REPORTS_DIR
    reports.mkdir(exist_ok=True)
    holdout_cm_path = reports / 'confusion_matrix_holdout.png'
    fig.tight_layout()
    fig.savefig(holdout_cm_path)
    plt.close(fig)
    logger.info(f"✔ Saved hold‑out confusion matrix: {holdout_cm_path}")


def run_data_analysis():
    logger = logging.getLogger(__name__)
    test_metrics = REPORTS_DIR / 'complexity_vs_accuracy_test.csv'
    if not test_metrics.exists():
        logger.warning("'%s' not found; run data mining first", test_metrics.name)
        run_data_mining()

    validate_sample(DATA_PATH)
    df = pd.read_csv(test_metrics)
    analyze_complexity_vs_accuracy(
        df.drop(columns=['per_instance_accuracy']).to_dict(orient='records'),
        df['per_instance_accuracy'].values
    )
    plot_trends(DATA_PATH)


def main():
    configure_logging()
    parser = argparse.ArgumentParser(description="Pipeline Central Bank Resolutions")
    sub = parser.add_subparsers(dest='cmd')
    sub.add_parser('mining',   help='Run data mining only')
    sub.add_parser('analysis', help='Run data analysis only')
    sub.add_parser('everything', help='Run both mining and analysis')
    args = parser.parse_args()

    if args.cmd == 'mining':
        run_data_mining()
    elif args.cmd == 'analysis':
        run_data_analysis()
    else:
        run_data_mining()
        run_data_analysis()

if __name__ == '__main__':
    main()
