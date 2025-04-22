import sys
import argparse
import json
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from gensim.models import Word2Vec

from data_mining.preprocessing import preprocess_text
from data_analysis.complexity_analysis import calculate_complexity_metrics
from data_mining.categorization_model import train_and_evaluate_model
from data_analysis.validation import validate_sample
from data_analysis.statistical_analysis import analyze_complexity_vs_accuracy
from data_analysis.longitudinal_analysis import plot_trends

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

def bert_vectorize_texts(texts, tokenizer, model, batch_size=8, max_length=128):
    embeddings = []
    total = len(texts)
    logger = logging.getLogger(__name__)
    logger.info(f"▶ Vectorizing {total} texts in batches of {batch_size}")
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        logger.info(f"   • Processing texts {start+1} to {end} of {total}")
        batch = texts[start:end]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='tf'
        )
        outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        embeddings.append(cls_emb.numpy())
    result = np.vstack(embeddings)
    logger.info("✔ Finished BERT vectorization")
    return result

def run_data_mining():
    logger = logging.getLogger(__name__)
    logger.info("=== Starting Data Mining ===")

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        resolutions = json.load(f)
    logger.info(f"Loaded {len(resolutions)} resolutions")

    texts = []
    for r in resolutions:
        content = preprocess_text(r.get('content', ''))
        r['content_processed'] = content
        r['complexity_metrics'] = calculate_complexity_metrics(content)
        texts.append(content)
    logger.info("Preprocessing and complexity metrics done")

    Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
    bert_tok = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    try:
        X = bert_vectorize_texts(texts, bert_tok, bert_model)
    except Exception as e:
        logger.error(f"Error during BERT vectorization: {e}")
        sys.exit(1)

    cats = [r.get('category', f"cat_{i%2}") for i, r in enumerate(resolutions)]
    le = LabelEncoder().fit(cats)
    y = le.transform(cats)

    best_model, _, _ = train_and_evaluate_model(X, y)
    if best_model is None:
        logger.error("No best model returned")
        sys.exit(1)
    logger.info(f"Best model: {best_model.__class__.__name__}")

    joblib.dump(best_model, MODELS_DIR / 'best_model.pkl')
    joblib.dump(le,         MODELS_DIR / 'label_encoder.pkl')
    logger.info("Saved best model and label encoder")

    y_pred_full = best_model.predict(X)
    df_metrics = pd.DataFrame([r['complexity_metrics'] for r in resolutions])
    df_metrics['per_instance_accuracy'] = (y_pred_full == y).astype(int)
    metrics_csv = REPORTS_DIR / 'complexity_vs_accuracy.csv'
    df_metrics.to_csv(metrics_csv, index=False)
    logger.info(f"Saved complexity_vs_accuracy CSV: {metrics_csv}")

def run_data_analysis():
    logger = logging.getLogger(__name__)
    metrics_file = REPORTS_DIR / 'complexity_vs_accuracy.csv'

    if not metrics_file.exists():
        logger.warning(f"'{metrics_file.name}' not found; running data mining first")
        run_data_mining()

    validate_sample(DATA_PATH)

    df = pd.read_csv(metrics_file)
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
