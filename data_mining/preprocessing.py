import re
import nltk
import spacy
from pathlib import Path

def safe_nltk_download(resource):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1])

for res in ('tokenizers/punkt', 'corpora/stopwords', 'corpora/wordnet'):
    safe_nltk_download(res)

nlp = spacy.load('pt_core_news_sm')

def preprocess_text(text: str) -> str:
    text = re.sub(r'RESOLUÇÃO BCB Nº \d+, DE \d+ DE \w+ DE \d+', '', text)
    text = re.sub(r'\W|\d', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()

    doc = nlp(text)
    tokens = [tok.lemma_ for tok in doc if not tok.is_stop and not tok.is_punct]
    return ' '.join(tokens)
