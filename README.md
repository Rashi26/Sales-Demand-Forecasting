# Retail Sales Demand Forecasting

Forecasting daily grocery sales for a multi-store retail chain using real-world data from the [Corporación Favorita Kaggle competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data), a large Ecuadorian grocery retailer.

The project builds a **global XGBoost forecasting model** that learns across multiple stores simultaneously, and progressively layers in calendar effects, holidays, promotions, macroeconomic indicators, target transformation, and interpretability analysis (SHAP) on top of a solid time-series-safe baseline.

## Project Overview

The notebook walks through a full, realistic forecasting workflow:

1. **Data ingestion** — Pulling the raw dataset via the Kaggle API and filtering it down to a clean, workable time series.
2. **Exploratory analysis & decomposition** — Visualizing trend, weekly seasonality, and noise in daily sales.
3. **Feature engineering** — Calendar features, lag features (1/7/28-day), and rolling-window averages.
4. **Chronological train/test splitting** — Avoiding data leakage by training only on the past and testing on the future (train on 2013–2016, test on 2017).
5. **Single-store baseline model** — An XGBoost regressor forecasting Store 1's grocery sales.
6. **Global multi-store model** — Scaling to Stores 1–3 with store-aware grouped lag/rolling features and a categorical store identifier, so one model serves all locations.
7. **Feature importance & SHAP interpretability** — Understanding which features drive predictions, and why, both globally (beeswarm plots) and for individual predictions (waterfall plots).
8. **Exogenous variables** — Adding national holiday flags, oil price (Ecuador is an oil-dependent economy), and promotional (`onpromotion`) indicators.
9. **Target transformation** — Log-transforming sales (`log1p` / `expm1`) to stabilize variance.
10. **Advanced feature engineering** — Interaction terms (holiday × day-of-week) and a linear trend feature.
11. **Hyperparameter tuning** — Time-series-safe search using `TimeSeriesSplit` + `RandomizedSearchCV`.
12. **Error analysis** — Residual plots, actual-vs-predicted plots, and per-store error breakdowns to diagnose where the model struggles.

See [`PROJECT_REPORT.md`](PROJECT_REPORT.md) for a full write-up of the methodology, results, and findings.

## Key Results

| Model Iteration | MAE |
|---|---|
| Baseline global model (Stores 1–3) | 1071.95 |
| + National holiday features | **614.35** |
| + Oil price features | 622.50 |
| + Promotional (`onpromotion`) feature | 636.80 |
| + Log-transformed target | **590.15** (best) |
| + Interaction & trend features | 1072.45 |
| + Hyperparameter tuning (RandomizedSearchCV) | 1117.73 |

The two changes that meaningfully improved forecast accuracy were **national holiday flags** and **log-transforming the sales target**. Oil prices, added interaction/trend features, and the hyperparameter search did not outperform the simpler feature set on this dataset — a reminder that added complexity doesn't always translate to better generalization on unseen data.

## Tech Stack

- **Python** — pandas, NumPy
- **Modeling** — XGBoost (`XGBRegressor`)
- **Statistics** — statsmodels (seasonal decomposition)
- **Interpretability** — SHAP (TreeExplainer, beeswarm & waterfall plots)
- **Tuning/validation** — scikit-learn (`RandomizedSearchCV`, `TimeSeriesSplit`)
- **Visualization** — Matplotlib

## Dataset

- Source: [Store Sales – Time Series Forecasting (Kaggle)](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)
- Files used: `train.csv`, `holidays_events.csv`, `oil.csv`
- Scope: Grocery I product family, Stores 1–3, daily granularity, 2013–2017

The raw data is not included in this repository due to size and Kaggle's terms of use. To reproduce the results, download the competition data from Kaggle and place it in a `store-sales/` directory, or run the notebook's Kaggle API download cell with your own API credentials.

## Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib statsmodels xgboost scikit-learn shap
```

You will also need a [Kaggle API token](https://www.kaggle.com/docs/api) to download the dataset directly from the notebook (or you can download it manually from the competition page).

### Running the Notebook

1. Clone this repository.
2. Place your Kaggle API credentials where the notebook expects them (or download the dataset manually into `store-sales/`).
3. Open `Retail_Sales_Demand_Forecasting.ipynb` in Jupyter or Google Colab.
4. Run all cells in order — later sections build on DataFrames created earlier in the notebook.

## Repository Structure

```
.
├── Retail_Sales_Demand_Forecasting.ipynb   # Main analysis notebook
├── README.md                               # This file
└── PROJECT_REPORT.md                       # Detailed methodology, results, and insights
```

## Future Improvements

- Investigate why interaction/trend features and hyperparameter tuning underperformed — likely candidates include overfitting, an insufficiently broad search, or feature redundancy.
- Build a store-specific model or additional store-level features for Store 3, which consistently showed the highest residual error.
- Expand beyond the `GROCERY I` family and Stores 1–3 to test generalization across the full product/store hierarchy.
- Explore alternative models (LightGBM, Prophet, or a hybrid statistical + ML ensemble) as a comparison point against XGBoost.

## License

This project uses data from a Kaggle competition; refer to the [competition's rules](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/rules) for data usage terms. Code in this repository is available under the MIT License unless noted otherwise.
