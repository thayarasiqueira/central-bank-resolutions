import logging
from pathlib import Path
import pandas as pd
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
import pyphen

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

nlp = spacy.load('pt_core_news_sm')
dic = pyphen.Pyphen(lang='pt_BR')


def calculate_complexity_metrics(text: str) -> dict:
    try:
        sentences = sent_tokenize(text, language='portuguese')
        words = word_tokenize(text, language='portuguese')
        unique_words = set(words)

        avg_sentence_length = (
            sum(len(word_tokenize(s, language='portuguese')) for s in sentences) / len(sentences)
            if sentences else 0
        )

        lexical_density = len(unique_words) / len(words) if words else 0

        total_syllables = sum(dic.inserted(w).count('-') + 1 for w in words)

        flesch_index = 248.835 \
            - 84.6 * (total_syllables / len(words) if words else 0) \
            - 1.015 * (len(words) / len(sentences) if sentences else 0)

        doc = nlp(text)
        sent_list = list(doc.sents)
        syntactic_depth = (
            sum(len(list(sent.root.subtree)) for sent in sent_list) / len(sent_list)
            if sent_list else 0
        )

        report = {
            'avg_sentence_length': round(avg_sentence_length, 2),
            'lexical_density':     round(lexical_density, 2),
            'flesch_index':        round(flesch_index, 2),
            'syntactic_depth':     round(syntactic_depth, 2)
        }

        df = pd.DataFrame([report])
        csv_path = REPORTS_DIR / 'complexity_metrics_report.csv'
        df.to_csv(
            csv_path,
            mode='a',
            header=not csv_path.exists(),
            index=False
        )

        logger.info("✔ Complexity metrics calculated and saved to %s", csv_path)
        return report

    except Exception as e:
        logger.exception("❌ Failed to calculate complexity metrics: %s", e)
        return {}
