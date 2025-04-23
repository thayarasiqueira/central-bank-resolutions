# Central Bank Resolutions Classification

This repository provides a complete pipeline to collect, validate, analyze and automatically classify resolutions from the Central Bank of Brazil (BCB), with a quantitative study of how linguistic complexity impacts classification quality.

---

## 📂 Project Structure

```
.
├── data_collection
│   ├── main.py                  # Orchestrates scraping and content validation
│   ├── resolution_collector.py  # Selenium scraper for BCB resolutions
│   └── content_validator.py     # Basic content checks on .txt files
│
├── data
│   ├── raw                      # Raw JSON of collected resolutions
│   └── processed                # (future) processed data
│
├── data_mining
│   ├── preprocessing.py         # Text cleaning, tokenization, lemmatization
│   ├── categorization_model.py  # Baseline: TF-IDF + DistilBERT + ensemble
│   └── bert_finetuning.py       # Hybrid DistilBERT fine-tuning with TF-IDF
│
├── data_analysis
│   ├── complexity_analysis.py   # Compute linguistic complexity metrics
│   ├── statistical_analysis.py  # Correlations & scatter plots
│   ├── longitudinal_analysis.py # Document length trends over time
│   └── validation.py            # Sample validation reporting
│
├── reports                      # Generated charts (.png, .html) & CSVs
├── models                       # Trained models (pickles / saved-model)
├── logs                         # Log files
├── tests                        # pytest unit tests
├── main.py                      # Orchestrates mining & analysis
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## ⚙️ Installation

1. Clone the repository and enter its folder:  
   ```bash
   git clone https://github.com/thayarasiqueira/central-bank-resolutions.git
   cd central-bank-resolutions
   ```

2. Create and activate a virtual environment (Linux/macOS):  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install Python dependencies:  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Download spaCy’s Portuguese model:  
   ```bash
   python -m spacy download pt_core_news_sm
   ```

---

## 🚀 How to run

### 1. Data collection  
Scrape the BCB website and validate raw .txt files:  
```bash
python data_collection/main.py
```
- Output: `data/raw/resolutions_data.json` and `reports/sample_validation_report.txt`

### 2. Mining (feature extraction & training)  
Extract features, train model and produce hold-out accuracy CSV:  
```bash
python main.py mining
```
- Outputs in `reports/`:  
  - `complexity_vs_accuracy_test.csv`  
  - `confusion_matrix_holdout.png`  

### 3. Analysis (charts & statistics)  
Generate correlation matrix, scatter plots and trend chart:  
```bash
python main.py analysis
```
- Outputs in `reports/`:  
  - `correlation_matrix.html`  
  - `*_vs_accuracy.html`  
  - `longitudinal_trends.png`  

### 4. Full pipeline  
Run both mining and analysis in sequence:  
```bash
python main.py everything
```

---

## 📊 Baseline Results

- **Hold-out accuracy** (light ensemble): ~43 %  
- **Correlations** between linguistic metrics and accuracy:  
  - Lexical density: +0.13  
  - Avg. sentence length: +0.06  
  - Flesch index: –0.06  
  - Syntactic depth: +0.04  

These baseline findings demonstrate the moderate but measurable impact of textual complexity on classification performance.

---

## 🔧 Best Practices & Extensions

- Regenerate `requirements.txt` via `pip freeze > requirements.txt` for exact versions.  
- Use `--model-type [classic|bert]` in `main.py` to compare pipelines.  
- Add CI (GitHub Actions) to run `pytest` and lint on each push.  
- Future work:  
  - Fine-tune Legal-BERT or BERTimbau for Portuguese legal text.  
  - Implement stacking and probability calibration.  
  - Incorporate named-entity features (dates, numbers, organizations).

---

## 📝 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---