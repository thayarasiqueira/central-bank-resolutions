import logging
import nltk
from nltk.tokenize    import sent_tokenize, word_tokenize
from textstat         import flesch_reading_ease
import spacy
import pandas as pd
from pathlib import Path

nlp = spacy.load('pt_core_news_sm')
logger = logging.getLogger(__name__)

def calculate_complexity_metrics(text: str) -> dict:
    try:
        sents = sent_tokenize(text, language='portuguese')
        words = word_tokenize(text, language='portuguese')
        uniq  = set(words)

        avg_sent_len = (
            sum(len(word_tokenize(s, language='portuguese')) for s in sents) / len(sents)
            if sents else 0
        )
        lex_density = len(uniq) / len(words) if words else 0
        flesch_idx  = flesch_reading_ease(text)

        doc        = nlp(text)
        sent_list  = list(doc.sents)
        synt_depth = (
            sum(len(list(sent.root.subtree)) for sent in sent_list) / len(sent_list)
            if sent_list else 0
        )

        report = {
            'avg_sentence_length': round(avg_sent_len, 2),
            'lexical_density':     round(lex_density, 2),
            'flesch_index':        round(flesch_idx, 2),
            'syntactic_depth':     round(synt_depth, 2)
        }

        Path('reports').mkdir(exist_ok=True)
        pd.DataFrame([report]).to_csv(
            'reports/complexity_metrics_report.csv',
            mode='a', header=not Path('reports/complexity_metrics_report.csv').exists(),
            index=False
        )

        logger.info("Complexity metrics calculated.")
        return report

    except Exception as e:
        logger.error(f"Error in calculate_complexity_metrics: {e}")
        return {}
