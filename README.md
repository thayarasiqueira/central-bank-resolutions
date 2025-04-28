**Central Bank Resolutions Classification**

This project implements an end-to-end pipeline for analyzing linguistic complexity and performing automatic classification of Central Bank of Brazil resolutions. It includes data collection, complexity metric extraction, model training, and evaluation.

---

## 📂 Repository Structure
```
├── data_collection/          # Scripts for scraping and extracting resolution texts
│   ├── main.py               # Entry point: fetches JSON data from BCB site
│   └── content_validator.py  # Validates completeness and readability of JSON entries
│
├── data_analysis/            # Complexity metric computation
│   └── complexity_analysis.py # Computes lexical density, sentence length, syntactic features
│
├── models/                   # Model training and evaluation
│   └── categorization_model.py# Implements preprocessing, SMOTE, cross‑validation, classifiers
│
├── outputs/                  # Generated artifacts (JSON, figures, logs)
│
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🔧 Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/thayarasiqueira/central-bank-resolutions.git
   cd central-bank-resolutions
   ```
2. Create a virtual environment and install:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

---

## 📝 Data Collection

- **Source**: Resolutions published on the Central Bank of Brazil website (2020–2024).
- **Script**: `data_collection/main.py`
  - Fetches resolution metadata and content into `outputs/resolutions_data.json`.
  - Runs `content_validator.validate_resolution_content` on each JSON entry to exclude incomplete or too-short texts.

### Validation criteria
- Exclude entries with missing `text` or fewer than 100 characters.
- Exclude revoked resolutions or non‑text attachments via JSON flags.
- A manual spot check is implemented within the validator for 10% of items.

---

## 📊 Complexity Metrics

Metrics are computed in `data_analysis/complexity_analysis.py`:

| Metric               | Implementation detail                                 |
|----------------------|-------------------------------------------------------|
| Sentence length      | Average number of tokens per sentence                 |
| Lexical density      | **Type–token ratio**: `unique_words / total_words`    |
| Flesch readability   | Adapted Flesch score for Portuguese via `textstat`     |
| Syntactic size       | Average subtree size from spaCy dependency parse      |

Run:
```bash
python data_analysis/complexity_analysis.py --input outputs/resolutions_data.json --output outputs/complexity_metrics.csv
```

---

## 🤖 Modeling & Evaluation

Implemented in `models/categorization_model.py`:

1. **Preprocessing**: text normalization, Portuguese tokenization, lemmatization.
2. **Resampling**: SMOTE on the training folds.
3. **Cross‑validation**: stratified 3‑fold (matches code) for parameter tuning.
4. **Classifiers**:
   - Random Forest
   - Support Vector Machine (SVM)
   - LightGBM (in lieu of XGBoost)
   - Ensemble voting classifier
5. **Evaluation**:
   - Hold‑out test (20% split)
   - Confusion matrix
   - Pearson correlations between complexity metrics and per‑instance accuracy

To train and evaluate:
```bash
python models/categorization_model.py --metrics outputs/complexity_metrics.csv --labels outputs/labels.csv --save-model outputs/model.pkl
```

---

## 📈 Results
- Results figures (scatter‑plots, correlation matrices) are saved under `outputs/figures/`.
- Confusion matrices for hold‑out and ensemble are logged to `outputs/logs/`.

---

## ⚙️ Configuration
- All hyperparameters and file paths can be adjusted via command‑line flags in each script.
