# IPL Next Over Runs Predictor 🏏📈

## Overview
This project is an end-to-end, data-driven machine learning system designed to predict the number of runs scored in the **next over** of an IPL T20 cricket match.

Unlike match-level predictions, this project focuses on **micro-level market behavior**, making it suitable for sports analytics, trading decision support, and real-time inference use cases.

The project was built as an **unguided project**, covering data engineering, modeling, evaluation, deployment, and debugging.

---

## Problem Statement
Given the current match state at the end of an over, predict how many runs will be scored in the **next over** using historical IPL ball-by-ball data.

This type of prediction is useful for:
- Over/Under market analysis
- Momentum detection
- Short-horizon trading and risk modeling
- Real-time decision support systems

---

## Data
- Source: Cricsheet (IPL ball-by-ball data in JSON format)
- Raw data consists of individual match JSON files
- Converted to:
  - Ball-level dataset
  - Over-level aggregated dataset
- Strict care taken to avoid **data leakage**

### Key Aggregations
- Runs per over
- Wickets per over
- Rolling averages (last 3 overs)
- Match context features (run rate, wickets remaining, over phase)

---

## Feature Engineering
Core features used:
- `over`
- `runs_in_over`
- `runs_last_3_overs`
- `wickets_last_3_overs`
- `current_run_rate`
- `wickets_remaining`
- `over_phase` (Powerplay / Middle / Death)

All features are available at prediction time, ensuring deployability in live settings.

---

## Model
- Algorithm: **XGBoost Regressor**
- Target: `runs_next_over`
- Evaluation Metric: **Mean Absolute Error (MAE)**

### Performance
- Final MAE ≈ **3.8 runs**
- Model intentionally constrained to context-only features
- No player-level data to avoid availability and leakage issues

> Note: The model is designed for robustness and interpretability rather than overfitting for raw accuracy.

---

## Trading Signal Logic (Decision Support)
Predictions are converted into signals using a rule-based approach:

- **OVER**: Predicted runs exceed market expectation by a defined threshold
- **UNDER**: Predicted runs fall significantly below expectation
- **NO TRADE**: Insufficient edge

This demonstrates how ML outputs can be integrated into a disciplined decision-making framework.

---

## Deployment
- Built and deployed using **Streamlit**
- Interactive UI allows users to simulate match states
- Model is trained at app startup using a lightweight demo dataset for reproducibility
- Fully cloud-compatible (no local paths or dependencies)

---
## TRY : https://iplnextoverrun.streamlit.app/

<img width="773" height="641" alt="image" src="https://github.com/user-attachments/assets/61e320b1-c479-4b80-96b6-f6843de23822" />

