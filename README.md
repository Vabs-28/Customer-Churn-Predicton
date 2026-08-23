# Bank Customer Churn Predictor

An end-to-end machine learning project that predicts whether a bank customer is likely to churn (leave the bank), based on their profile — credit score, geography, gender, age, tenure, balance, number of products, credit card status, activity status, and salary. Built with a Keras Artificial Neural Network and deployed as an interactive Streamlit app.

---

## Overview

Banks lose far more acquiring a new customer than retaining an existing one. This project flags high-risk customers in advance so a retention team can step in before they actually leave — a call from a relationship manager, a better offer, a loyalty perk — turning a reactive problem ("why did we lose that customer?") into a proactive one ("who's at risk, right now?").

**Live demo:** *[add your Streamlit Cloud link here once deployed]*

---

## Features

- Cleaned and encoded a 10,000-row bank customer dataset
- Trained a Keras ANN classifier with `EarlyStopping` to prevent overfitting
- Ran a `GridSearchCV` hyperparameter search across 16 architecture combinations
- Built a bonus regression variant of the same pipeline to predict `EstimatedSalary`
- Verified inference end-to-end on unseen data before deployment
- Deployed as an interactive Streamlit web app with dynamic form inputs

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas / NumPy | Data loading and manipulation |
| Scikit-learn | Encoding, scaling, train/test split, hyperparameter search |
| TensorFlow / Keras | Building, training, and saving the neural network |
| SciKeras | Wraps the Keras model so `GridSearchCV` can search over it |
| TensorBoard | Visualizes training/validation loss and accuracy |
| Streamlit | Interactive web app for live predictions |
| Pickle | Persists fitted encoders/scaler for consistent reuse |

---

## Dataset

- **File:** `Churn_Modelling.csv`
- **Size:** 10,000 rows × 14 columns
- **Missing values:** none
- **Target:** `Exited` (1 = churned, 0 = stayed)

| Column | Description |
|---|---|
| CreditScore | Customer's credit score |
| Geography | Country: France / Germany / Spain |
| Gender | Male / Female |
| Age | Customer's age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products held (1–4) |
| HasCrCard | Owns a credit card (0/1) |
| IsActiveMember | Actively using the account (0/1) |
| EstimatedSalary | Estimated annual salary |
| Exited | Target — did the customer churn? |

`RowNumber`, `CustomerId`, and `Surname` are dropped during preprocessing — they're identifiers with no real relationship to churn behavior.

### A few things the data shows

- Class imbalance: ~80% stayed vs ~20% churned
- Germany churns at ~32%, roughly double France and Spain
- Inactive members churn almost 2x more than active ones
- `NumOfProducts` has a sharp, non-linear relationship with churn — 2 products is the safest group; 3–4 products is almost always churn

---

## Project Structure

```
Customer-Churn-Predictor/
├── Churn_Modelling.csv              # Raw dataset
├── experiments.ipynb                 # Main notebook: cleaning, encoding, scaling, ANN training
├── hyperparametertuningann.ipynb     # GridSearchCV search over ANN architectures
├── salaryregression.ipynb            # Same pipeline, predicts salary instead of churn
├── prediction.ipynb                  # Loads saved model + encoders, tests one new customer
├── app.py                            # Streamlit web app
├── requirements.txt                  # Python dependencies
├── model.h5                          # Trained Keras model
├── label_encoder_gender.pkl          # Fitted LabelEncoder (Gender)
├── onehot_encoder_geo.pkl            # Fitted OneHotEncoder (Geography)
└── scaler.pkl                        # Fitted StandardScaler (all numeric features)
```

---

## How It Works

1. **Clean** — drop identifier columns (`RowNumber`, `CustomerId`, `Surname`)
2. **Encode** — `Gender` via `LabelEncoder`, `Geography` via `OneHotEncoder`
3. **Split** — 80/20 train/test split (`random_state=42`)
4. **Scale** — `StandardScaler` fit on training data only, applied to both sets
5. **Build** — a `Dense(64, relu) → Dense(32, relu) → Dense(1, sigmoid)` ANN
6. **Train** — Adam optimizer, binary cross-entropy loss, with `EarlyStopping` (patience=10, restores best weights) and TensorBoard logging
7. **Save** — model and all fitted preprocessing objects persisted to disk
8. **Deploy** — Streamlit app reloads everything and serves live predictions

---

## Model Performance

| Metric | Value |
|---|---|
| Validation accuracy | ~85–86% |
| Best hyperparameter search result | 16 neurons, 1 layer — 85.84% CV accuracy |
| Salary regression MAE | ~50,168 (target proved largely unpredictable from available features) |

> Training log shows mild overfitting after epoch ~7 — handled via `EarlyStopping` with `restore_best_weights=True`.

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/Vabs-28/Customer-Churn-Predicton.git
cd Customer-Churn-Predicton
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal (typically `http://localhost:8501`).

### Retrain the model (optional)

Open `experiments.ipynb` in Jupyter and run all cells — this regenerates `model.h5`, `scaler.pkl`, `label_encoder_gender.pkl`, and `onehot_encoder_geo.pkl`.

---

## Usage

1. Launch the app with `streamlit run app.py`
2. Fill in a customer's profile — geography, gender, age, tenure, balance, credit score, salary, number of products, credit card and activity status
3. The app returns a churn probability and a plain verdict ("likely to churn" / "not likely to churn")

---

## Known Limitations

- Class imbalance (~80/20) isn't explicitly handled — accuracy alone can be misleading at this ratio
- The best architecture found by `GridSearchCV` hasn't been applied to the deployed model yet
- `model.h5` uses the legacy HDF5 format; TensorFlow now recommends `.keras`
- No caching in the Streamlit app — the model reloads on every interaction
- Fixed 0.5 decision threshold, not tuned to business cost
- `EstimatedSalary` in this dataset shows little real relationship with the other features

---

## Roadmap / Future Improvements

- [ ] Address class imbalance (`class_weight` or resampling)
- [ ] Apply the tuned architecture found via `GridSearchCV`
- [ ] Add `@st.cache_resource` to the Streamlit app
- [ ] Migrate to the `.keras` save format
- [ ] Add precision/recall/F1/AUC-ROC alongside accuracy
- [ ] Add a proper README badge set, CI, and unit tests

---

## License

*[Add your license here, e.g. MIT — see the LICENSE file]*

---

## Acknowledgements

Dataset: the widely-used "Churn Modelling" bank customer dataset.
