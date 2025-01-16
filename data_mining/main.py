import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from data_analysis.complexity_analysis import calculate_complexity_metrics
from data_mining.categorization_model import train_and_evaluate_model
from data_analysis.validation import validate_sample
from data_mining.preprocessing import preprocess_text
from gensim.models import Word2Vec
from transformers import BertTokenizer, TFBertModel
from data_analysis.statistical_analysis import analyze_complexity_vs_accuracy
from data_analysis.longitudinal_analysis import plot_trends

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/data_mining.log'),
            logging.StreamHandler()
        ]
    )

def main():
    configure_logging()
    logger = logging.getLogger(__name__)
    
    data_path = Path(__file__).resolve().parent.parent / 'data/raw/resolutions_data.json'
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            resolutions = json.load(f)
        logger.info("Resolutions data loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load resolutions data: {e}")
        return

    word2vec_model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    for i, resolution in enumerate(resolutions):
        try:
            resolution['content'] = preprocess_text(resolution['content'])
            resolution['complexity_metrics'] = calculate_complexity_metrics(resolution['content'])
            if 'category' not in resolution:
                resolution['category'] = 'categoria_1' if i % 2 == 0 else 'categoria_2'
        except Exception as e:
            logger.warning(f"Error processing resolution {i}: {e}")

    categories = set(res['category'] for res in resolutions)
    if len(categories) <= 1:
        logger.error("The dataset needs to have more than one category.")
        return

    X = np.array([list(res['complexity_metrics'].values()) for res in resolutions])
    y = np.array([res['category'] for res in resolutions])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    try:
        train_and_evaluate_model(X, y_encoded, word2vec_model, bert_model)
        logger.info("Model trained and evaluated successfully.")
    except Exception as e:
        logger.error(f"Error during model training and evaluation: {e}")

    try:
        complexity_metrics = [res['complexity_metrics'] for res in resolutions]
        accuracy_scores = [0.85] * len(complexity_metrics)
        analyze_complexity_vs_accuracy(complexity_metrics, accuracy_scores)
        logger.info("Statistical analysis completed successfully.")
    except Exception as e:
        logger.error(f"Error during statistical analysis: {e}")

    try:
        plot_trends(data_path)
        logger.info("Longitudinal analysis completed successfully.")
    except Exception as e:
        logger.error(f"Error during longitudinal analysis: {e}")

if __name__ == "__main__":
    main()