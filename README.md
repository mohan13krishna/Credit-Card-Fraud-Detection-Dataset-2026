# Credit Card Fraud Detection Dataset 2026

**100,000 transactions · 43 features · 3 targets · Full year 2026 · Research-calibrated**

The original Kaggle fraud dataset from 2013 has anonymised V1 to V28 PCA columns that nobody can interpret. This dataset replaces it with real named features, full timestamps, customer profiles, velocity signals, and 3 prediction targets across 3 difficulty levels.

---

## Quick Stats

| Property | Value |
|---|---|
| Rows | 100,000 |
| Columns | 43 |
| Fraud rate | 2.06% |
| Unique customers | 8,000 |
| Unique merchants | 3,000 |
| Date range | 2026-01-01 to 2026-12-31 |
| File size | 25.7 MB |
| Missing values | 0 |
| Duplicate rows | 0 |

---

## Files

```
credit_card_fraud_detection_2026.csv   # Main dataset (100K rows, 43 columns)
README.md                              # This file
```

---

## Target Variables

| Target | Type | Task | Fraud Rate / Notes |
|---|---|---|---|
| `is_fraud` | Binary | Classification | 2.06% positive rate -- beginner friendly |
| `fraud_type` | Multiclass | Classification | 5 classes: Legitimate, Card-Not-Present, Card-Present, Account-Takeover, Synthetic-Identity |
| `fraud_amount_lost` | Float | Regression | USD lost; 0 for legitimate transactions |

---

## Key Fraud Signals (Research-Calibrated)

| Signal | Fraud Rate | vs Baseline |
|---|---|---|
| Night transactions (1am to 4am) | 8.5% | 4.1x |
| High-risk country merchant | 11.1% | 5.4x |
| New account under 90 days | 4.9% | 2.4x |
| Velocity 3 or more tx in past hour | 5.6% | 2.7x |
| Chip used | 0.8% | 0.4x (protective) |
| PIN used | 0.8% | 0.4x (protective) |
| Online Retail category | 3.3% | 1.6x |

---

## Feature Groups

```
Transaction Identifiers  (5)  -- transaction_id, customer_id, card_id, merchant_id, transaction_datetime
Transaction Details      (6)  -- amount, merchant_category, merchant_country, hour, day_of_week, card_present
Customer Profile         (7)  -- age, gender, state, account_age_days, credit_limit, credit_score, num_cards
Velocity Features        (6)  -- tx_last_1h, tx_last_24h, tx_last_7d, amount_last_1h, amount_last_24h, days_since_last
Risk Signals             (7)  -- distance_from_home, distance_from_last_tx, international, high_risk_country,
                                 amount_vs_avg_ratio, used_chip, used_pin
Device and Channel       (4)  -- transaction_channel, device_type, browser, is_weekend
Account Behaviour        (5)  -- avg_amount_30d, std_amount_30d, num_declined_30d, chargebacks_ever, email_domain
Targets                  (3)  -- is_fraud, fraud_type, fraud_amount_lost
```

---

## Usage Examples

### Load the Dataset
```python
import pandas as pd

df = pd.read_csv('credit_card_fraud_detection_2026.csv')
df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'])
print(df.shape)         # (100000, 43)
print(df.is_fraud.mean())  # 0.0206
```

### Binary Classification with XGBoost
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report

df_enc = df.drop(['transaction_id','customer_id','card_id',
                  'merchant_id','transaction_datetime',
                  'fraud_type','fraud_amount_lost'], axis=1).copy()

for col in df_enc.select_dtypes(include='object').columns:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col])

X = df_enc.drop('is_fraud', axis=1)
y = df_enc['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    scale_pos_weight=y_train.value_counts()[0] / y_train.value_counts()[1],
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:,1]
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
```

### Time-Ordered Train/Test Split (No Leakage)
```python
df_sorted = df.sort_values('transaction_datetime').reset_index(drop=True)
split = int(len(df_sorted) * 0.8)
train_df = df_sorted.iloc[:split]
test_df  = df_sorted.iloc[split:]
print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
```

### Fraud Type Multiclass
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

fraud_only = df_enc[df_enc['is_fraud'] == 1].copy()
# Build multiclass model on fraud subtypes only
```

---

## Calibration Sources

| Metric | Dataset Value | Source |
|---|---|---|
| Fraud rate | 2.06% | Nilson Report 2023 |
| CNP fraud share | 51.3% | Federal Reserve Payments Study 2023 |
| Night-hour multiplier | 4.1x | FICO Falcon benchmark |
| High-risk country multiplier | 5.4x | ECB Card Fraud Report 2022 |
| Mean fraud amount lost | $136.18 | TransUnion Global Fraud Report 2023 |

---

## Suggested Notebook Analyses

- Hourly fraud rate bar chart -- the 1am to 4am spike is dramatic and visually striking
- Merchant category fraud heatmap -- Online Retail and Travel stand out clearly
- Velocity feature distribution: fraud vs legitimate boxplots
- SHAP summary plot on XGBoost model -- transaction_hour and distance_from_home will be top features
- Precision-recall curve -- more meaningful than ROC for highly imbalanced data
- Time-ordered train/test split to simulate real deployment
- Two-stage model: first predict is_fraud, then predict fraud_amount_lost on positive predictions

---

## License

CC0: Public Domain. Free for academic, commercial, and personal use.

---

## Author

**Mohan Krishna Thalla**
- Kaggle: https://www.kaggle.com/mohankrishnathalla
- Dataset: [https://www.kaggle.com/datasets/mohankrishnathalla/credit-card-fraud-detection-2026](https://www.kaggle.com/datasets/mohankrishnathalla/credit-card-fraud-detection-dataset-2026)

---

**Version:** 1.0 | **Updated:** October 2026 | **Records:** 100,000 | **License:** CC0
