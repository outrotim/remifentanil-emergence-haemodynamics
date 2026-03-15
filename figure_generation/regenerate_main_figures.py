#!/usr/bin/env python3
"""
Regenerate Figures 2, 3, 4 for BJA submission.

Figure 2: 2-panel RCS dose-response (HR rebound, MAP rebound) — main text only
          OWSI complete-case RCS is supplementary material only
Figure 3: Taper dynamics composite (scatter, quartile bars, attenuation forest)
Figure 4: Robustness composite (IPTW forest, missingness by quartile)
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS = Path(__file__).parent / 'results'
MAIN = Path(__file__).parent / 'main_figures'
DATA = Path(__file__).parent.parent / 'data'

# Ensure output dir
MAIN.mkdir(exist_ok=True)

# ── Shared style ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 600,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

COLORS = {
    'hr': '#2E86C1',
    'map': '#27AE60',
    'owsi': '#8E44AD',
    'taper': '#E67E22',
    'iptw_crude': '#95A5A6',
    'iptw_adj': '#2C3E50',
    'miss': '#E74C3C',
}


def load_rcs(filename):
    """Load pre-computed RCS JSON with curve data."""
    fpath = RESULTS.parent.parent / 'results' / filename
    if not fpath.exists():
        fpath = RESULTS / filename
    with open(fpath) as f:
        d = json.load(f)
    x = np.array(d['curve_data']['x'])
    y = np.array(d['curve_data']['y'])
    yl = np.array(d['curve_data']['y_lower'])
    yu = np.array(d['curve_data']['y_upper'])
    return x, y, yl, yu, d


def format_p(p):
    if p < 0.001:
        return 'P < 0.001'
    else:
        return f'P = {p:.3f}'


def plot_rcs_panel(ax, x, y, yl, yu, d, ylabel, color, letter, title,
                   rug_data=None):
    """Plot single RCS panel with CI band and optional rug plot."""
    ax.fill_between(x, yl, yu, alpha=0.15, color=color, linewidth=0)
    ax.plot(x, y, color=color, linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.set_xlabel('Remifentanil total dose (\u03bcg/kg)', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)

    # Rug plot along x-axis
    if rug_data is not None:
        ax.plot(rug_data, np.full_like(rug_data, ax.get_ylim()[0]),
                '|', color=color, alpha=0.08, markersize=3, markeredgewidth=0.3)

    n = d['n']
    p_nl = d['p_nonlinear']
    p_str = format_p(p_nl)
    txt = f'n = {n:,}\nP$_{{nonlinear}}$ {p_str}'
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, fontsize=7.5,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.92))
    ax.set_xlim(0, 75)
    ax.set_title(f'{letter}. {title}', fontsize=10, fontweight='bold', loc='left')


# ================================================================
# Figure 2: 3-panel RCS (HR, MAP, complete-case OWSI)
# ================================================================
def compute_complete_case_owsi_rcs():
    """Re-run RCS for complete-case OWSI from raw data."""
    print('  Computing complete-case OWSI RCS from raw data...')
    import statsmodels.api as sm
    from scipy import stats

    df = pd.read_csv(DATA / 'oih_master_dataset.csv')

    # Build complete-case OWSI: need HR_rebound, MAP_rebound, FTN_rescue_mcg_kg all present
    required = ['HR_rebound', 'MAP_rebound', 'FTN_rescue_mcg_kg', 'RFTN_total_mcg_kg']
    df_cc = df.dropna(subset=required).copy()

    # Z-score each component
    for col in ['HR_rebound', 'MAP_rebound', 'FTN_rescue_mcg_kg']:
        m, s = df_cc[col].mean(), df_cc[col].std()
        df_cc[f'{col}_z'] = (df_cc[col] - m) / s

    df_cc['OWSI_complete'] = df_cc[['HR_rebound_z', 'MAP_rebound_z', 'FTN_rescue_mcg_kg_z']].mean(axis=1)

    # Covariates
    covariates = ['age', 'bmi', 'asa', 'opdur', 'TWA_BIS', 'intraop_ppf']
    if 'sex' in df_cc.columns:
        df_cc['sex_num'] = (df_cc['sex'] == 'F').astype(int)
        covariates.append('sex_num')

    cols = ['RFTN_total_mcg_kg', 'OWSI_complete'] + covariates
    df_clean = df_cc[cols].dropna().copy()
    print(f'  Complete-case OWSI n = {len(df_clean)}')

    # Knot placement (5, 35, 65, 95 percentiles)
    exposure = 'RFTN_total_mcg_kg'
    outcome = 'OWSI_complete'
    knots = np.percentile(df_clean[exposure], [5, 35, 65, 95])

    # Linear model
    cov_str = ' + '.join(covariates)
    linear_formula = f'{outcome} ~ {exposure} + {cov_str}'
    linear_model = sm.OLS.from_formula(linear_formula, data=df_clean).fit()

    # Spline model
    spline_formula = f'{outcome} ~ cr({exposure}, df=3) + {cov_str}'
    spline_model = sm.OLS.from_formula(spline_formula, data=df_clean).fit()

    # LRT
    lr_stat = -2 * (linear_model.llf - spline_model.llf)
    df_diff = spline_model.df_model - linear_model.df_model
    p_nonlinear = 1 - stats.chi2.cdf(lr_stat, df_diff)

    # Prediction curve
    x_pred = np.linspace(df_clean[exposure].quantile(0.025),
                         df_clean[exposure].quantile(0.975), 200)
    pred_df = pd.DataFrame({exposure: x_pred})
    for cov in covariates:
        pred_df[cov] = df_clean[cov].median()
    y_pred = spline_model.predict(pred_df)

    # Bootstrap CI (200 iterations for speed)
    n_boot = 200
    y_boot = np.zeros((n_boot, len(x_pred)))
    rng = np.random.RandomState(42)
    for b in range(n_boot):
        idx = rng.choice(len(df_clean), len(df_clean), replace=True)
        boot_df = df_clean.iloc[idx]
        try:
            bm = sm.OLS.from_formula(spline_formula, data=boot_df).fit()
            y_boot[b, :] = bm.predict(pred_df)
        except Exception:
            y_boot[b, :] = np.nan

    y_lower = np.nanpercentile(y_boot, 2.5, axis=0)
    y_upper = np.nanpercentile(y_boot, 97.5, axis=0)

    result = {
        'exposure': exposure,
        'outcome': 'OWSI_complete',
        'n': len(df_clean),
        'n_knots': 4,
        'knots': knots.tolist(),
        'linear_r2': linear_model.rsquared,
        'spline_r2': spline_model.rsquared,
        'p_nonlinear': float(p_nonlinear),
        'curve_data': {
            'x': x_pred.tolist(),
            'y': y_pred.tolist(),
            'y_lower': y_lower.tolist(),
            'y_upper': y_upper.tolist()
        }
    }

    # Save for future use
    outfile = RESULTS / 'rcs_RFTN_total_mcg_kg_OWSI_complete.json'
    with open(outfile, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  Saved complete-case OWSI RCS to {outfile}')
    return result


def generate_figure2():
    """Figure 2: 2-panel RCS dose-response (HR and MAP rebound only).

    OWSI complete-case RCS is demoted to supplementary material and must NOT
    appear in the main-text Figure 2. This ensures the figure matches the
    manuscript legend which describes exactly 2 panels (A and B).
    """
    print('\n=== Generating Figure 2 (2-panel: HR + MAP only) ===')

    # Load dose data for rug plots (after outlier removal)
    df = pd.read_csv(DATA / 'oih_master_dataset.csv')
    outlier_mask = (df['RFTN_total_mcg_kg'] > 200) | (df['RFTN_Ce_peak'] > 100)
    rftn_cols = [c for c in df.columns if 'RFTN' in c]
    df.loc[outlier_mask, rftn_cols] = np.nan
    dose_vals = df['RFTN_total_mcg_kg'].dropna().values

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=600,
                             constrained_layout=True)

    # Panel A: HR rebound (n = 2,854; P_nonlinear = 0.024)
    x, y, yl, yu, d = load_rcs('rcs_RFTN_total_mcg_kg_HR_rebound.json')
    plot_rcs_panel(axes[0], x, y, yl, yu, d, 'HR rebound (bpm)',
                   COLORS['hr'], 'A', 'HR rebound', rug_data=dose_vals)

    # Panel B: MAP rebound (n = 2,842; P_nonlinear < 0.001)
    x, y, yl, yu, d = load_rcs('rcs_RFTN_total_mcg_kg_MAP_rebound.json')
    plot_rcs_panel(axes[1], x, y, yl, yu, d, 'MAP rebound (mmHg)',
                   COLORS['map'], 'B', 'MAP rebound', rug_data=dose_vals)

    fig.savefig(MAIN / 'Figure_2_RCS_composite.pdf', bbox_inches='tight', dpi=600)
    fig.savefig(MAIN / 'Figure_2_RCS_composite.png', bbox_inches='tight', dpi=600)
    plt.close()
    print('  Figure 2 saved (2-panel, 600 DPI).')


# ================================================================
# Figure 3: Taper dynamics composite
# ================================================================
def generate_figure3():
    """Figure 3: Taper dynamics (3 panels)."""
    print('\n=== Generating Figure 3 ===')

    with open(RESULTS / 'taper_dynamics.json') as f:
        td = json.load(f)
    with open(RESULTS / 'taper_expansion.json') as f:
        te = json.load(f)
    with open(RESULTS / 'reviewer2_analyses.json') as f:
        r2 = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    # ── Panel A: Taper type comparison (box/bar plot) ──
    ax = axes[0]
    taper_hr = td['taper_type\u2192HR_rebound']
    categories = ['More gradual\n(below median)', 'Less gradual\n(above median)']
    means = [taper_hr['gradual_mean'], taper_hr['abrupt_mean']]
    ns = [taper_hr['n_gradual'], taper_hr['n_abrupt']]
    colors_bar = ['#27AE60', '#E74C3C']

    bars = ax.bar(categories, means, color=colors_bar, width=0.55, alpha=0.85,
                  edgecolor='white', linewidth=1.2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)

    # Add value labels
    for bar, m, n in zip(bars, means, ns):
        ypos = m + 0.15 if m > 0 else m - 0.25
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f'{m:.2f}\n(n={n:,})', ha='center', va='bottom' if m > 0 else 'top',
                fontsize=8, fontweight='bold')

    p_val = taper_hr['p']
    ax.set_ylabel('HR rebound (bpm)', fontsize=9)
    ax.set_title('A. Taper type comparison', fontsize=10, fontweight='bold', loc='left')
    ax.text(0.97, 0.97, f'Mann-Whitney\n{format_p(p_val)}',
            transform=ax.transAxes, fontsize=7.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.92))
    ax.set_ylim(-2.2, 3.0)

    # ── Panel B: Taper slope quartile trend ──
    ax = axes[1]
    qa = te['taper_quartile_analysis']['HR_rebound']
    q_labels = ['Q1\n(most\ngradual)', 'Q2', 'Q3', 'Q4\n(most\nabrupt)']
    q_means = [qa['quartile_means']['Q1_gradual'], qa['quartile_means']['Q2'],
               qa['quartile_means']['Q3'], qa['quartile_means']['Q4_abrupt']]
    q_ns = [qa['quartile_ns']['Q1_gradual'], qa['quartile_ns']['Q2'],
            qa['quartile_ns']['Q3'], qa['quartile_ns']['Q4_abrupt']]

    # Gradient colors
    q_colors = ['#27AE60', '#82E0AA', '#F5B041', '#E74C3C']
    bars = ax.bar(q_labels, q_means, color=q_colors, width=0.6, alpha=0.85,
                  edgecolor='white', linewidth=1.2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)

    for bar, m, n in zip(bars, q_means, q_ns):
        ypos = m + 0.12 if m > 0 else m - 0.18
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f'{m:.1f}', ha='center', va='bottom' if m > 0 else 'top',
                fontsize=7.5)

    kw_p = qa['kruskal_wallis_p']
    trend_r = qa['spearman_r_trend']
    ax.set_ylabel('HR rebound (bpm)', fontsize=9)
    ax.set_title('B. Taper slope quartiles', fontsize=10, fontweight='bold', loc='left')
    ax.text(0.97, 0.97,
            f'Kruskal-Wallis {format_p(kw_p)}\nr$_{{trend}}$ = {trend_r:.3f}',
            transform=ax.transAxes, fontsize=7.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.92))
    ax.set_ylim(-2.5, 3.2)

    # ── Panel C: Partial correlation attenuation (forest-style) ──
    ax = axes[2]
    pc = r2['major3_taper']['partial_correlations']['HR_rebound']

    labels = ['Unadjusted', '+ Rate', '+ Rate, Ce', '+ Rate, Ce, Dose']
    rhos = [pc['unadjusted']['rho'], pc['ctrl_rate']['rho'],
            pc['ctrl_rate_Ce']['rho'], pc['ctrl_rate_Ce_dose']['rho']]
    pvals = [pc['unadjusted']['p'], pc['ctrl_rate']['p'],
             pc['ctrl_rate_Ce']['p'], pc['ctrl_rate_Ce_dose']['p']]

    y_pos = np.arange(len(labels))[::-1]  # top to bottom
    marker_colors = ['#2C3E50', '#2E86C1', '#3498DB', '#85C1E9']

    for i, (rho, p, yp, lbl, mc) in enumerate(zip(rhos, pvals, y_pos, labels, marker_colors)):
        marker = 'o' if p < 0.05 else 'D'
        ax.plot(rho, yp, marker=marker, markersize=9, color=mc,
                markeredgecolor='white', markeredgewidth=1.2, zorder=5)
        sig_str = '*' if p < 0.05 else ''
        ax.text(rho + 0.008, yp, f'\u03c1 = {rho:.3f}{sig_str}',
                va='center', fontsize=8, color=mc)

    ax.axvline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Spearman \u03c1 (taper slope \u2192 HR rebound)', fontsize=9)
    ax.set_title('C. Partial correlation attenuation', fontsize=10, fontweight='bold', loc='left')
    ax.set_xlim(-0.02, 0.25)

    # VIF annotation
    vif = r2['major3_taper']['vif']['RFTN_taper_slope']
    ax.text(0.97, 0.03, f'Taper VIF = {vif:.2f}',
            transform=ax.transAxes, fontsize=7.5, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF9E7',
                      edgecolor='#B7950B', alpha=0.9))

    # Legend
    filled = plt.Line2D([], [], marker='o', color='gray', markersize=7, linestyle='None', label='P < 0.05')
    hollow = plt.Line2D([], [], marker='D', color='gray', markersize=7, linestyle='None', label='P \u2265 0.05')
    ax.legend(handles=[filled, hollow], fontsize=7, loc='lower left',
              framealpha=0.9, edgecolor='#CCCCCC')

    plt.tight_layout()
    fig.savefig(MAIN / 'Figure_3_taper_dynamics.pdf', bbox_inches='tight')
    fig.savefig(MAIN / 'Figure_3_taper_dynamics.png', bbox_inches='tight')
    plt.close()
    print('  Figure 3 saved.')


# ================================================================
# Figure 4: Robustness composite (IPTW forest + missingness)
# ================================================================
def generate_figure4():
    """Figure 4: Robustness (2 panels)."""
    print('\n=== Generating Figure 4 ===')

    with open(RESULTS / 'iptw_results.json') as f:
        iptw = json.load(f)
    with open(RESULTS / 'missingness_report.json') as f:
        miss = json.load(f)
    with open(RESULTS / 'reviewer2_analyses.json') as f:
        r2 = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    # ── Panel A: IPTW forest plot ──
    ax = axes[0]
    endpoints = ['HR_rebound', 'MAP_rebound', 'FTN_rescue_mcg_kg', 'OSI', 'NHD_pct']
    labels = ['HR rebound\n(bpm)', 'MAP rebound\n(mmHg)',
              'FTN rescue\n(\u03bcg/kg)', 'OWSI\n(Z-score)', 'NHD\n(%)']
    y_pos = np.arange(len(endpoints))[::-1]

    for i, (ep, lbl) in enumerate(zip(endpoints, labels)):
        d = iptw[ep]
        crude = d['crude_diff']
        adj = d['iptw_diff']
        ci_lo = d['ci_low']
        ci_hi = d['ci_high']
        sig = d['significant']

        yp = y_pos[i]

        # Crude estimate (gray diamond)
        ax.plot(crude, yp + 0.15, marker='D', markersize=7,
                color=COLORS['iptw_crude'], markeredgecolor='white',
                markeredgewidth=0.8, zorder=5)

        # IPTW estimate with CI (dark)
        fc = COLORS['iptw_adj'] if sig else '#BDC3C7'
        ax.plot(adj, yp - 0.15, marker='s', markersize=8,
                color=fc, markeredgecolor='white',
                markeredgewidth=0.8, zorder=5)
        ax.plot([ci_lo, ci_hi], [yp - 0.15, yp - 0.15],
                color=fc, linewidth=2, zorder=4)

    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel('High vs Low infusion rate difference (IPTW-adjusted)', fontsize=9)
    ax.set_title('A. IPTW causal estimates', fontsize=10, fontweight='bold', loc='left')

    # Meta annotation — show 3-stage IPTW pipeline
    meta = iptw['_meta']
    ax.text(0.97, 0.03,
            f'PS-eligible n = 2,866\n'
            f'PS trim \u2192 n = 2,584\n'
            f'Analytic n = {meta["n"]:,} (hemo.)\n'
            f'PS AUC = {meta["ps_auc"]:.3f}',
            transform=ax.transAxes, fontsize=7, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.92))

    # Legend
    crude_h = plt.Line2D([], [], marker='D', color=COLORS['iptw_crude'],
                         markersize=7, linestyle='None', label='Crude')
    adj_h = plt.Line2D([], [], marker='s', color=COLORS['iptw_adj'],
                       markersize=7, linestyle='None', label='IPTW-adjusted')
    ax.legend(handles=[crude_h, adj_h], fontsize=7.5, loc='upper left',
              framealpha=0.9, edgecolor='#CCCCCC')

    # ── Panel B: Missingness by dose quartile ──
    ax = axes[1]
    hr_miss = miss['quartile_missingness']['HR_rebound']
    quartiles = ['Q1\n(lowest)', 'Q2', 'Q3', 'Q4\n(highest)']
    pcts = [hr_miss['pct_by_quartile']['Q1'], hr_miss['pct_by_quartile']['Q2'],
            hr_miss['pct_by_quartile']['Q3'], hr_miss['pct_by_quartile']['Q4']]

    bar_colors = ['#F1948A', '#EC7063', '#E74C3C', '#CB4335']
    bars = ax.bar(quartiles, pcts, color=bar_colors, width=0.55, alpha=0.85,
                  edgecolor='white', linewidth=1.2)

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 1.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8.5,
                fontweight='bold')

    ax.set_ylabel('Missing HR rebound data (%)', fontsize=9)
    ax.set_xlabel('Remifentanil dose quartile', fontsize=9)
    ax.set_title('B. Informative missingness', fontsize=10, fontweight='bold', loc='left')
    ax.set_ylim(0, 85)

    # Chi-square annotation
    chi2 = hr_miss['chi2']
    ax.text(0.97, 0.97,
            f'\u03c7\u00b2 = {chi2:.1f}\nP < 0.001',
            transform=ax.transAxes, fontsize=7.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.92))

    # IPOW comparison annotation
    ipow = r2['major1_ipow']
    cc_beta = ipow['ipow_regression_HR']['complete_case_beta']
    ipow_beta = ipow['ipow_regression_HR']['ipow_beta']
    pct_chg = ipow['ipow_regression_HR']['pct_change']
    obs_auc = ipow['observation_model']['auc']
    ax.text(0.03, 0.97,
            f'IPOW validation:\nCC \u03b2 = {cc_beta:.4f}\nIPOW \u03b2 = {ipow_beta:.4f}\n\u0394 = {pct_chg:.2f}%\nObs. AUC = {obs_auc:.4f}',
            transform=ax.transAxes, fontsize=7, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#EBF5FB',
                      edgecolor='#2E86C1', alpha=0.9))

    plt.tight_layout()
    fig.savefig(MAIN / 'Figure_4_robustness.pdf', bbox_inches='tight')
    fig.savefig(MAIN / 'Figure_4_robustness.png', bbox_inches='tight')
    plt.close()
    print('  Figure 4 saved.')


# ================================================================
if __name__ == '__main__':
    generate_figure2()
    generate_figure3()
    generate_figure4()
    print('\nAll main figures regenerated successfully.')
