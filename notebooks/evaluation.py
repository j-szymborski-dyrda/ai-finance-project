import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path('data')

# ── Function: summarise_portfolio ─────────────────────────
# Computes performance statistics for one model, saves
# the result to data/summary_<label>.json and
# saves monthly results to data/monthly_results_<label>.csv

def summarise_portfolio(monthly_results, label, oos_start='2010-01-01'):
    r = monthly_results.copy()
    r['date'] = pd.to_datetime(r['date'])

    # Full period
    mean_monthly = r['avg_net_revenue'].mean()
    std_monthly  = r['avg_net_revenue'].std()
    sharpe       = (mean_monthly / std_monthly) * np.sqrt(12)

    best_month  = r.loc[r['avg_net_revenue'].idxmax()]
    worst_month = r.loc[r['avg_net_revenue'].idxmin()]

    crisis = r[r['high_vol_regime'] == True]
    normal = r[r['high_vol_regime'] == False]

    # OOS period
    oos        = r[r['date'] >= oos_start]
    oos_mean   = oos['avg_net_revenue'].mean()
    oos_std    = oos['avg_net_revenue'].std()
    oos_sharpe = (oos_mean / oos_std) * np.sqrt(12)

    # Print
    print("=" * 55)
    print(f"  {label.upper()} — SUMMARY STATISTICS")
    print("=" * 55)
    print(f"Months:                    {len(r)}")
    print(f"Avg monthly net revenue:   {mean_monthly:.4%}")
    print(f"Avg annual net revenue:    {mean_monthly*12:.4%}")
    print(f"Std monthly net revenue:   {std_monthly:.4%}")
    print(f"Sharpe ratio (annualized): {sharpe:.4f}")
    print(f"Best month:  {best_month['date'].strftime('%Y-%m')} ({best_month['avg_net_revenue']:.4%})")
    print(f"Worst month: {worst_month['date'].strftime('%Y-%m')} ({worst_month['avg_net_revenue']:.4%})")
    print(f"\n--- CRISIS VS NORMAL ---")
    print(f"Normal months ({len(normal):>3}):  avg {normal['avg_net_revenue'].mean():.4%}/month")
    print(f"Crisis months ({len(crisis):>3}):  avg {crisis['avg_net_revenue'].mean():.4%}/month")
    print(f"\n--- OOS PERIOD ({oos_start[:4]}–2023) ---")
    print(f"Months:                    {len(oos)}")
    print(f"Avg annual net revenue:    {oos_mean*12:.4%}")
    print(f"Sharpe ratio (annualized): {oos_sharpe:.4f}")
    print("=" * 55)

    # Save summary to JSON
    summary = {
        'label'                         : label,
        'full_period_annual_net_revenue' : mean_monthly * 12,
        'full_period_sharpe'            : sharpe,
        'full_period_worst_month'       : worst_month['avg_net_revenue'],
        'full_period_best_month'        : best_month['avg_net_revenue'],
        'crisis_avg_monthly'            : crisis['avg_net_revenue'].mean(),
        'normal_avg_monthly'            : normal['avg_net_revenue'].mean(),
        'oos_annual_net_revenue'        : oos_mean * 12,
        'oos_sharpe'                    : oos_sharpe,
    }

    DATA_PATH.mkdir(exist_ok=True)
    with open(DATA_PATH / f'summary_{label}.json', 'w') as f:
        json.dump(summary, f, indent=4)

    # Save monthly results to CSV
    r.to_csv(DATA_PATH / f'monthly_results_{label}.csv', index=False)

    print(f"\nSaved to data/summary_{label}.json ✅")
    print(f"Saved to data/monthly_results_{label}.csv ✅")
    return summary

def compare_models():
    # Auto-detect all summary JSON files in data/
    files = sorted(DATA_PATH.glob('summary_*.json'))

    if not files:
        print("No summary files found in data/. Run summarise_portfolio() in each model notebook first.")
        return

    summaries = []
    for f in files:
        with open(f, 'r') as file:
            summaries.append(json.load(file))

    # Build comparison table
    metrics = {
        'Annual net revenue (full)' : 'full_period_annual_net_revenue',
        'Sharpe ratio (full)'       : 'full_period_sharpe',
        'Worst month'               : 'full_period_worst_month',
        'Best month'                : 'full_period_best_month',
        'Crisis avg/month'          : 'crisis_avg_monthly',
        'Normal avg/month'          : 'normal_avg_monthly',
        'Annual net revenue (OOS)'  : 'oos_annual_net_revenue',
        'Sharpe ratio (OOS)'        : 'oos_sharpe',
    }

    table = pd.DataFrame(
        {s['label']: {label: s[key] for label, key in metrics.items()} for s in summaries}
    )

    # Format as percentages where relevant
    pct_rows = ['Annual net revenue (full)', 'Worst month', 'Best month',
                'Crisis avg/month', 'Normal avg/month', 'Annual net revenue (OOS)']

    print("\n" + "=" * 55)
    print("  MODEL COMPARISON")
    print("=" * 55)
    display(table.style.format(
        {col: lambda x, r=None: f"{x:.4f}" for col in table.columns}
    ).format(
        {col: (lambda x: f"{x:.2%}") for col in table.columns},
        subset=pd.IndexSlice[pct_rows, :]
    ))

    return table

# ── Function: plot_cumulative_net_revenue ──────────────────
# Auto-detects all monthly_results_*.csv files in data/
# Saves to data/plot_cumulative_net_revenue.png

def plot_cumulative_net_revenue():
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    files = sorted(DATA_PATH.glob('monthly_results_*.csv'))

    if not files:
        print("No monthly results files found in data/. Run summarise_portfolio() first.")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['steelblue', 'darkorange', 'green', 'red']

    for i, f in enumerate(files):
        label = f.stem.replace('monthly_results_', '')
        df = pd.read_csv(f, parse_dates=['date'])
        df = df.sort_values('date')
        df['cumulative_net_revenue'] = (1 + df['avg_net_revenue']).cumprod()
        ax.plot(df['date'], df['cumulative_net_revenue'],
                label=label, linewidth=1.5, color=colors[i % len(colors)])

    # Shade crisis months using first file as reference
    ref = pd.read_csv(files[0], parse_dates=['date'])
    crisis_months = ref[ref['high_vol_regime'] == True]
    for _, row in crisis_months.iterrows():
        ax.axvspan(row['date'], row['date'], alpha=0.3, color='red', linewidth=4)

    crisis_patch = mpatches.Patch(color='red', alpha=0.3, label='Crisis months (market vol > 30%)')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [crisis_patch])

    ax.set_title('Cumulative Net Revenue (1990–2023)')
    ax.set_ylabel('Cumulative Net Revenue')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(DATA_PATH / 'plot_cumulative_net_revenue.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to data/plot_cumulative_net_revenue.png ✅")

    # ── Function: plot_crisis_vs_normal ───────────────────────
# Auto-detects all summary_*.json files in data/
# Saves to data/plot_crisis_vs_normal.png

def plot_crisis_vs_normal():
    import matplotlib.pyplot as plt
    import numpy as np

    files = sorted(DATA_PATH.glob('summary_*.json'))

    if not files:
        print("No summary files found in data/. Run summarise_portfolio() first.")
        return

    labels  = []
    crisis  = []
    normal  = []

    for f in files:
        with open(f, 'r') as file:
            s = json.load(file)
        labels.append(s['label'])
        crisis.append(s['crisis_avg_monthly'] * 100)
        normal.append(s['normal_avg_monthly'] * 100)

    x      = np.arange(len(labels))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, normal, width, label='Normal months', color='steelblue')
    ax.bar(x + width/2, crisis, width, label='Crisis months', color='red', alpha=0.7)

    ax.set_title('Average Monthly Net Revenue: Crisis vs Normal')
    ax.set_ylabel('Average Monthly Net Revenue (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(DATA_PATH / 'plot_crisis_vs_normal.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to data/plot_crisis_vs_normal.png ✅")

    # ── Function: summarise_predictions ───────────────────────
# Computes statistical validation metrics for a model that
# predicts volatility. Compares against the naive baseline
# (real_vol_ann_lag1) which is what the benchmark uses.
#
# Parameters:
#   monthly_results : DataFrame with columns:
#       - real_vol_ann      : actual realized volatility
#       - real_vol_ann_lag1 : naive baseline (last month's vol)
#       - predicted_vol     : model's predicted volatility
#       - high_vol_regime   : True/False crisis flag
#       - date              : month end date
#   label : str, name of the model

def summarise_predictions(monthly_results, label):
    from sklearn.metrics import mean_squared_error

    r = monthly_results.copy()
    r['date'] = pd.to_datetime(r['date'])

    y_actual   = r['real_vol_ann']
    y_pred     = r['predicted_vol']
    y_baseline = r['real_vol_ann_lag1']

    # ── R-squared formula ──────────────────────────────────
    # OOS R-squared relative to naive baseline
    # Positive means model beats simply using last month's vol
    ss_res_model    = ((y_actual - y_pred) ** 2).sum()
    ss_res_baseline = ((y_actual - y_baseline) ** 2).sum()
    r2_oos = 1 - (ss_res_model / ss_res_baseline)

    # Standard R-squared vs mean
    ss_tot = ((y_actual - y_actual.mean()) ** 2).sum()
    # ── Function: compare_predictions ─────────────────────────
# Auto-detects all predictions_*.json files in data/
# and prints a side by side comparison table

def compare_predictions():
    files = sorted(DATA_PATH.glob('predictions_*.json'))

    if not files:
        print("No prediction files found in data/. Run summarise_predictions() first.")
        return

    summaries = []
    for f in files:
        with open(f, 'r') as file:
            summaries.append(json.load(file))

    metrics = {
        'OOS R-squared (full)'   : 'r2_oos',
        'R-squared (vs mean)'    : 'r2',
        'RMSE model'             : 'rmse_model',
        'RMSE baseline'          : 'rmse_baseline',
        'OOS R-squared (normal)' : 'r2_oos_normal',
        'OOS R-squared (crisis)' : 'r2_oos_crisis',
        'RMSE normal months'     : 'rmse_normal',
        'RMSE crisis months'     : 'rmse_crisis',
    }

    table = pd.DataFrame(
        {s['label']: {label: s[key] for label, key in metrics.items()} for s in summaries}
    )

    print("\n" + "=" * 55)
    print("  PREDICTION COMPARISON")
    print("=" * 55)
    display(table.style.format("{:.4f}"))

    return table