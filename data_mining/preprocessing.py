import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load('pt_core_news_sm')

def preprocess_text(text):
    text = re.sub(r'RESOLUÇÃO BCB Nº \d+, DE \d+ DE \w+ DE \d+', '', text)
    
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    text = text.lower()
    
    words = word_tokenize(text, language='portuguese')
    
    stop_words = set(stopwords.words('portuguese'))
    words = [word for word in words if word not in stop_words]
    
    doc = nlp(' '.join(words))
    words = [token.lemma_ for token in doc]
    
    return ' '.join(words)
