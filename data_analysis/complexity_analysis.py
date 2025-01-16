import logging
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from textstat import flesch_reading_ease
import spacy
import pandas as pd
from pathlib import Path

nlp = spacy.load('pt_core_news_sm')
logger = logging.getLogger(__name__)

def calculate_complexity_metrics(text):
    try:
        sentences = sent_tokenize(text, language='portuguese')
        words = word_tokenize(text, language='portuguese')
        unique_words = set(words)
        
        avg_sentence_length = sum(len(word_tokenize(s, language='portuguese')) for s in sentences) / len(sentences)
        lexical_density = len(unique_words) / len(words)
        flesch_index = flesch_reading_ease(text)
        
        doc = nlp(text)
        syntactic_depth = sum(len(list(sent.root.subtree)) for sent in doc.sents) / len(list(doc.sents))
        
        logger.info("Complexity metrics calculated successfully.")
        
        report = {
            'avg_sentence_length': avg_sentence_length,
            'lexical_density': lexical_density,
            'flesch_index': flesch_index,
            'syntactic_depth': syntactic_depth
        }
        
        df = pd.DataFrame([report])
        df.to_csv('reports/complexity_metrics_report.csv', mode='a', header=not Path('reports/complexity_metrics_report.csv').exists(), index=False)
        
        return report
    except Exception as e:
        logger.error(f"Error calculating complexity metrics: {e}")
        return {} 