# Bar Moan Bank — AI in Finance Assignment

## Assignment Overview
Bar Moan Bank manages an equity trading book. Each month the bank either 
exits the market or selects 500+ stocks in equal weight, aiming to maximize 
net investment revenue after capital requirement costs.

Capital costs are proportional to each stock's realized volatility.
Penalty doubles when market volatility exceeds 30%.

We develop and compare two volatility forecasting models, evaluated against 
a naive benchmark, and recommend one in a final management report.

## Models
- **Benchmark:** Naive persistence — always pick 500 stocks with lowest 
  realized vol from previous month. No model required.
- **Model A:** K-means regime detection (k=2) + Gradient Boosting
- **Model B:** LM sentiment scores from 10-K filings + Gradient Boosting

## Data
- Source: CRSP via WRDS (share codes 10/11, exchanges 1/2/3, 1990–2023)
- Filters: price > $1, minimum 15 trading days per month
- Master file: data/clean_monthly_panel.parquet (load this, never raw data)

## Setup

### 1. Clone the repo
git clone https://github.com/yourteam/bar-moan-trading

### 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn wrds requests nltk

### 3. WRDS Access
You need a WRDS account to run notebook 1.
Ask the group for the pre-cleaned parquet file if you don't have access.

### 4. Run notebooks in order
1. notebooks/1_data_pipeline.ipynb       → CRSP pull, cleaning, vol computation
2. notebooks/2_model_A_prediction.ipynb  → regime detection + Gradient Boosting
3. notebooks/3_model_B_nlp.ipynb         → 10-K sentiment + Gradient Boosting
4. notebooks/4_backtest.ipynb            → monthly selection loop, net revenue
5. notebooks/5_validation.ipynb          → model comparison, charts, statistics

## Rules
- Never commit directly to main — always open a pull request
- Never use future data in any rolling window
- Always use TimeSeriesSplit, never KFold
- Align 10-K sentiment to month AFTER filing date, not fiscal year end