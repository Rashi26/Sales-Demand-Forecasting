# Project Report: Retail Sales Demand Forecasting

## 1. Objective

The goal of this project was to build a demand forecasting model for daily retail sales using real-world data from Corporación Favorita, a large Ecuadorian grocery chain. Rather than working with a toy or synthetic dataset, the project uses the full Kaggle "Store Sales – Time Series Forecasting" dataset, filtered to a manageable subset (Grocery I family, Stores 1–3), to practice the realistic end-to-end workflow of a production time-series forecasting problem: data cleaning, feature engineering, chronologically-safe validation, exogenous data integration, interpretability, and error analysis.

## 2. Data

- **Source:** [Corporación Favorita Grocery Sales dataset (Kaggle)](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)
- **Core file:** `train.csv` — daily sales by store and product family, 2013–2017
- **Supplementary files:**
  - `holidays_events.csv` — national, regional, and local holiday calendar, including transferred holidays
  - `oil.csv` — daily oil price, a proxy for Ecuador's oil-dependent economy
- **Scope of analysis:** Product family `GROCERY I`, Store 1 for the initial single-series baseline, then Stores 1, 2, and 3 for the global model
- **Preprocessing:** Data was resampled to a continuous daily frequency (filling closed days with 0 sales) and reshaped into a complete store × date grid, since gaps in the raw data would otherwise corrupt rolling/lag calculations.

## 3. Methodology

### 3.1 Exploratory Data Analysis

Daily sales were plotted alongside a 30-day rolling mean to visually separate trend from noise, and `statsmodels.seasonal_decompose` was used (with a 7-day period) to formally decompose the series into trend, weekly seasonality, and residual components.

### 3.2 Feature Engineering

Because tree-based models like XGBoost have no innate concept of time, the datetime index was translated into explicit features:

- **Calendar features:** day of week, month, year, day of month, weekend flag
- **Lag features:** sales 1, 7, and 28 days prior
- **Rolling window features:** 7-day and 14-day trailing rolling means (shifted to avoid leakage)

### 3.3 Chronological Validation

Unlike typical ML problems, time series data cannot be randomly split without leaking future information into training. All models in this project were trained on 2013–2016 data and evaluated on 2017 data, and hyperparameter tuning used `TimeSeriesSplit` rather than standard k-fold cross-validation for the same reason.

### 3.4 From Single-Store to Global Model

The initial model forecasted only Store 1. To scale to multiple stores (1, 2, 3) as a single **global model**, two changes were essential:

1. **Grouped lag/rolling calculations** — lags and rolling windows were computed per `store_nbr` group; otherwise a store's "yesterday" value could incorrectly pull from a different store's last row.
2. **Categorical store identifier** — `store_nbr` was cast to a pandas `category` dtype and XGBoost was configured with `enable_categorical=True`, letting the model learn store-specific patterns through a single set of trees rather than maintaining separate models per store.

### 3.5 Interpretability (SHAP)

Beyond XGBoost's built-in feature importance (gain and weight), SHAP's `TreeExplainer` was used to:

- Produce a **beeswarm plot** showing, across a sample of 1,000 test rows, the direction and magnitude of each feature's effect on predicted sales.
- Produce a **waterfall plot** for an individual prediction, decomposing it from the model's baseline expectation down to feature-by-feature contributions.

This distinguishes *how often* a feature is used (weight) from *how much it actually reduces error* (gain) and *which direction* it pushes predictions (SHAP).

### 3.6 Exogenous Variables

Three external signals were incrementally added and evaluated:

- **National holidays** — a binary flag for non-transferred national holidays, merged onto the daily grid, with lag/rolling features rebuilt afterward so the model could see whether *recent* days (not just the current day) were holidays.
- **Oil price** — forward/backward-filled to handle non-trading weekends, plus a 30-day rolling average to capture slower macroeconomic effects on shopping behavior.
- **Promotions (`onpromotion`)** — the count/flag of items on promotion, merged in from the raw sales data.

### 3.7 Target Transformation

A `log1p` transformation was applied to the sales target before training, with `expm1` used to invert predictions back to the original scale before computing error metrics. This is a standard technique for stabilizing variance in right-skewed sales data, where a small number of very high-sales days would otherwise dominate the loss function.

### 3.8 Advanced Features & Hyperparameter Tuning

Additional features were tested: an interaction term (`is_national_holiday × dayofweek`) and a per-store linear trend counter. Hyperparameters (`max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`) were tuned with `RandomizedSearchCV` over a `TimeSeriesSplit` cross-validator.

### 3.9 Error Analysis

The final model's residuals (actual − predicted) were examined via:

- Actual-vs-predicted scatter plots
- Residual distribution histograms
- Residuals vs. predicted-value scatter plots (to check for heteroscedasticity)
- Residuals over time, by store
- Average absolute residual per store

## 4. Results

| Model Iteration | Mean Absolute Error (MAE) | Change vs. Baseline |
|---|---|---|
| Single-store baseline (Store 1 only) | 346.51 | — |
| Global model baseline (Stores 1–3) | 1071.95 | — |
| + National holiday features | 614.35 | -42.7% |
| + Oil price features | 622.50 | -41.9% |
| + Promotional (`onpromotion`) feature | 636.80 | -40.6% |
| + Log-transformed target | **590.15** | **-44.9% (best)** |
| + Interaction & linear trend features | 1072.45 | ~0% |
| + Hyperparameter tuning (RandomizedSearchCV) | 1117.73 | +4.3% (worse) |

*(Note: the single-store and global-model MAEs are not directly comparable — the global model's MAE is averaged across three stores with different sales volumes, including higher-volume/higher-variance Store 3.)*

### Feature Importance

- By **gain**, lag features (`lag_1`, `lag_7`, `lag_28`) and rolling means were the strongest drivers of error reduction.
- By **weight**, the model split most frequently on the categorical `store_nbr` identifier, reflecting its role in helping the tree ensemble separate store-specific baselines.
- Oil price and its 30-day average ranked low by gain, consistent with the marginal MAE regression observed when they were added.

### Error Analysis Findings

- **Actual vs. predicted:** points generally tracked the ideal diagonal, but scatter increased at higher sales values, and the model showed a tendency to underpredict sales spikes.
- **Residual distribution:** approximately centered at zero (suggesting no strong systematic bias) but with a heavier right tail, indicating occasional large underpredictions.
- **Residuals vs. predicted:** a fanning-out pattern (heteroscedasticity) — error magnitude grows as predicted sales increase, meaning the model is less reliable for high-sales days.
- **Residuals over time / by store:** Store 3 showed the largest and most volatile residuals of the three stores, suggesting store-specific dynamics (e.g., different customer base, local promotions, or demand patterns) that the shared global feature set doesn't fully capture.

## 5. Discussion

Two interventions clearly helped: **national holiday flags** and the **log transformation of the sales target**. Both address well-understood weaknesses of naive lag-based models — holidays because they create sales spikes uncorrelated with recent history, and the log transform because it reduces the influence of high-variance, high-magnitude sales days on the loss function.

Three interventions did **not** help on this dataset:

- **Oil price** — plausible in theory (Ecuador's economy is oil-dependent) but too indirect and slow-moving to affect daily grocery demand at the 3-store scale tested here.
- **Interaction and trend features** — these added complexity without a clear signal, and the resulting MAE regression suggests either overfitting or that the specific interaction chosen (holiday × day-of-week) wasn't well-suited to this data.
- **Hyperparameter tuning** — a `RandomizedSearchCV` with a limited number of iterations did not outperform sensible default parameters, underscoring that broader or more targeted search space design would be needed to realize gains here, and that default XGBoost hyperparameters are already a strong starting point for this class of problem.

The error analysis reinforces that most of the model's remaining error is concentrated in (a) high-sales days across all stores and (b) Store 3 specifically. This points to two natural next steps: modeling variance more explicitly (e.g., quantile regression or a two-stage model for spike detection) and considering store-specific features or models where a single global model underperforms.

## 6. Conclusion

This project demonstrates a realistic, leakage-safe approach to multi-store retail demand forecasting with XGBoost — from a single-series baseline to a global model, then layering exogenous data and interpretability analysis. The best-performing configuration combined a global multi-store model, grouped lag/rolling features, national holiday flags, and a log-transformed target, achieving an MAE of ~590 units against an average daily volume of several thousand units. The exercise also highlights an important, easily overlooked lesson in applied forecasting: not every additional feature or tuning pass improves generalization, and the more valuable engineering effort here was in encoding calendar knowledge (holidays) and getting the target's scale right, not in adding more complexity.

## 7. Known Issues / Notes for Reproduction

- The `create_multi_features` function was redefined multiple times through the notebook as new features (`onpromotion`, interaction terms, trend) were added; running cells out of order will use whichever version was defined most recently.
- An `early_stopping_rounds` argument passed directly to `.fit()` raised an error in a later XGBoost version and was removed in favor of relying on `n_estimators` from the tuned hyperparameters — a good example of an API compatibility issue to watch for when reproducing this notebook with a different XGBoost version.
- A `FutureWarning` from `groupby` with categorical dtypes during residual analysis was identified and addressed by explicitly passing `observed=False`.
