# Bar Moan Bank — AI in Finance Assignment

## Project Goal
Predict next-month individual stock volatility for 500+ US stocks.
Select 500 lowest-predicted-vol stocks monthly, maximize net revenue 
after capital costs (costs double when market vol > 30%).

## Data
- Source: CRSP via WRDS (share codes 10/11, exchanges 1/2/3, 1990–2023)
- Filters: price > $1, min 15 trading days/month
- Master file: data/clean_monthly_panel.parquet

## Column Schema (agreed format — do not change without telling the group)
date, permno, ret, real_vol_monthly, real_vol_ann,
mkt_vol_ann, high_vol_regime, n_days

## Models
- Benchmark: naive persistence (last month's vol, pick 500 lowest)
- Model A: K-means regime detection (k=2) + Gradient Boosting
- Model B: LM sentiment from 10-K filings (SEC EDGAR) + Gradient Boosting

## Critical Rules
- NEVER use future data in any rolling window
- Always TimeSeriesSplit, never KFold
- 10-K sentiment: align to month AFTER filing date (not fiscal year end)
- Predictions use only data available at prediction time

## Notebooks
1_data_pipeline.ipynb     → Person 1: WRDS pull, cleaning, vol computation
2_model_A_prediction.ipynb → Person 2: regime detection + Gradient Boosting  
3_model_B_nlp.ipynb        → Person 3: SEC EDGAR + LM scores + Gradient Boosting
4_backtest.ipynb           → Person 1: monthly selection loop, net revenue
5_validation.ipynb         → All: comparison, charts, statistical tests