# Fake News Detection — Project Report

**Date:** April 17, 2026  
**Dataset:** WELFake_Dataset.csv

---

## 1. Problem Statement

Binary classification of news articles as **Real** or **Fake** using NLP and classical machine learning.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | WELFake Dataset |
| Input column | `text` |
| Label column | `label` (0 = fake, 1 = real) |
| Dev mode rows | 5,000 (for benchmarking) |
| Full dataset | All rows (used for final saved model) |

---

## 3. Preprocessing & Feature Engineering

**Text Vectorization:** TF-IDF (`TfidfVectorizer`)

| Parameter | Value |
|---|---|
| `stop_words` | English |
| `max_features` | 5,000 |

**Train/Test Split:**

| Parameter | Value |
|---|---|
| Test size | 20% |
| Train size | 80% |
| `random_state` | 42 |
| Stratified | Yes (`stratify=y`) |

No manual stemming or regex cleaning was applied in the final pipeline — `TfidfVectorizer` with English stop words handled text normalization.

---

## 4. Models Benchmarked

Three models were trained and evaluated in `Fake_news_detection.py`:

| Model | Key Hyperparameters |
|---|---|
| Logistic Regression | `max_iter=1000`, `random_state=42` |
| Multinomial Naive Bayes | Default |
| Random Forest | `n_estimators=100`, `random_state=42`, `n_jobs=-1` |

---

## 5. Results

Confusion matrices are saved as PNG files in the project root:

- `confusion_matrix_Logistic_Regression.png`
- `confusion_matrix_Naive_Bayes.png`
- `confusion_matrix_Random_Forest.png`

| Model | Result |
|---|---|
| Logistic Regression | See confusion matrix |
| Naive Bayes | See confusion matrix |
| Random Forest | See confusion matrix — **selected as best performer** in benchmarking |

> Note: Exact accuracy numbers are generated at runtime and printed to console. The README identifies **Random Forest** as the top-performing model based on confusion matrix results.

---

## 6. Final Deployed Model

Despite Random Forest winning the benchmark, the **saved/deployed model is Logistic Regression** (`Train and Save Model.py`). This is likely a practical trade-off — Logistic Regression is significantly faster to train and serialize, and its accuracy on the full WELFake dataset is strong enough for production use.

| Artifact | File |
|---|---|
| Trained model | `fake_news_model.joblib` |
| TF-IDF vectorizer | `fake_news_vectorizer.joblib` |
| Deployment model | Logistic Regression (`max_iter=1000`) |

---

## 7. Deployment

A **Streamlit** web app (`app.py`) serves predictions:

- User pastes a news article
- Text is vectorized using the saved `TfidfVectorizer`
- Logistic Regression model predicts `real` or `fake`
- Confidence score (from `predict_proba`) is shown

**Run command:**
```bash
python3 -m streamlit run app.py
```

---

## 8. Tech Stack

| Category | Library |
|---|---|
| Data | Pandas, NumPy |
| ML | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Serialization | Joblib |
| App | Streamlit |
| Language | Python 3.14 |

---

## 9. Key Observations

1. **Discrepancy:** The README references Flask for deployment, but the actual `app.py` uses Streamlit.
2. **Benchmark vs. Deploy:** Random Forest was the best benchmark model, but Logistic Regression was chosen for the deployed artifact — likely for speed and simplicity.
3. **Dev mode:** `Fake_news_detection.py` runs on 5,000 rows by default (`DEV_MODE = True`). The full dataset is only used in `Train and Save Model.py` (`DEV_MODE = False`), so benchmark accuracy numbers may not reflect full-dataset performance.
