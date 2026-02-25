# Hybrid M5 Baseline
### Neural Multi-Horizon Forecasting + Residual Boosting (Proof of Concept)

This repository contains an experimental baseline built on the **M5 Forecasting Accuracy** dataset from Kaggle.

**Purpose:** Test whether a global neural multi-horizon model can be systematically improved by modeling its residuals with gradient boosting (hybrid NN + GBDT).

---

## Overview

- **Approach:**  
  Train a global neural network to predict 28-day windows → compute residuals on a deterministic validation cut → train a LightGBM residual model → add residual predictions to the NN forecast.

- **Status:** Proof-of-concept (POC). Not competition ready.
- **Key Result (v1):** Hybrid RMSE improved over NN baseline on validation cut.

---

## Core Idea

1. Train a **global multi-series neural network** to forecast `H = 28` days from `HISTORY = 180` days.
2. Compute residuals on a held-out time cut.
3. Train LightGBM to predict:

   residual = y_true − y_nn

4. Produce hybrid forecasts:

   y_hybrid = y_nn + y_gbdt_residual

If RMSE(hybrid) < RMSE(nn) then residual structure exists and hybrid modeling is justified.

---

## Architecture

### Neural Model (Global Multi-Horizon)

**Inputs**
- `sales_hist` → 180 past daily sales `(180, 1)`
- `item_id` → categorical embedding
- `store_id` → categorical embedding

**Network**
- Stacked causal `Conv1D`
- `GlobalAveragePooling1D`
- Concatenate embeddings
- Dense layers
- Output → 28 forecast steps

**Loss:** MSE  
**Optimizer:** AdamW  

---

### Residual Model (LightGBM)

Trained on per-series, per-horizon residuals.

**Features**
- `item_id`
- `store_id`
- `horizon`
- `nn_pred`

**Target**
- `residual = true − nn_pred`

**Purpose:**
- Correct horizon bias
- Capture cross-sectional nonlinearities
- Reduce structured NN error

---

## Data Requirements
**Place M5 CSV files inside:**
- ./datasets/
  - sales_train_validation.csv
  - calendar.csv
  - sell_prices.csv


---

## Window Configuration

| Parameter | Value |
|-----------|-------|
| HISTORY   | 180   |
| HORIZON   | 28    |
| STRIDE    | 28    |
| VAL_SPLIT_DAYS | 200 |

**Validation window per series:**
- past_val = series[train_cutoff - HISTORY : train_cutoff]
- future_val = series[train_cutoff : train_cutoff + HORIZON]


---

## Pipeline Design
- Sliding windows generated using `tf.signal.frame`
- Streaming via `tf.data.Dataset.flat_map`
- `.shuffle() → .batch() → .repeat() → .prefetch()`
- Deterministic validation windows
- Residual DataFrame constructed for LightGBM

**Designed to:**
- Avoid massive precomputed arrays
- Scale across all M5 series
- Remain architecture-focused

---

## Current Experimental Results (v1)
- NN baseline RMSE: 2.454108
- Hybrid RMSE: 1.940528
- Improvement (RMSE): 0.513580

**Interpretation:**
The residual learner reduces validation RMSE significantly, confirming structured residual signal.

---

## Repository Structure
- hybrid_m5_baseline.py
- datasets/m5-forecasting-accuracy.zip (must be extracted first)
- README.md


---

## How To Run

### 1. Create environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install tensorflow lightgbm pandas numpy scikit-learn
```

### 2. Add dataset files
Place the CSV files inside ./datasets/.

### 3. Run experiment
```bash
python hybrid_m5_baseline_v1.py
```

#### Limitations:
- Does not compute WRMSSE
- No hierarchical reconciliation
- No calendar or price features (v1)
- LightGBM trained on single validation cut
- Hyperparameters not tuned
- Metric is flattened RMSE (not leaderboard metric)
- This is architecture validation only.

#### Next Experiments:
- Add calendar & price features
- Replace Conv1D with TCN or attention
- Rolling residual training
- Implement WRMSSE
- Learnable residual fusion (instead of static addition)
- Multi-stage residual stacking
- Differentiable boosting approximation

#### Status:
- Proof-of-concept complete.
- Hybrid improvement confirmed.
- Architecture open for mutation.
