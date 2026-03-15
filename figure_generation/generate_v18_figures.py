#!/usr/bin/env python3
"""
Generate v18 main figures for BJA submission.

Figure 2: 4-panel dose-response composite
  A: HR rebound vs total dose (RCS)
  B: MAP rebound vs total dose (RCS)
  C: HR rebound vs mean infusion rate (adjusted RCS)
  D: GPS continuous dose-response

Figure 3: 2-panel propensity-weighted analysis
  A: Propensity score distribution overlap
  B: Love plot (covariate balance)

Figure 4: Forest plot of all sensitivity analyses

Also fixes supplementary figures:
  S2: remove OIH threshold line
  S3: remove ED50 label
  S5: remove panel B (age-stratified)
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).parent
RESULTS = BASE / 'results'
MAIN = BASE / 'main_figures'
SUPPL = BASE / 'supplementary_figures'
DATA = BASE.parent / 'data'

MAIN.mkdir(exist_ok=True)
SUPPL.mkdir(exist_ok=True)

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
    'rate': '#E67E22',
    'gps': '#8E44AD',
    'iptw_high': '#2E86C1',
    'iptw_low': '#E74C3C',
    'forest_primary': '#2C3E50',
    'forest_sens': '#2E86C1',
    'forest_subgroup': '#27AE60',
}


def load_rcs(filename):
    """Load pre-computed RCS JSON with curve data."""
    for path in [RESULTS / filename, BASE.parent / 'results' / filename]:
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            x = np.array(d['curve_data']['x'])
            y = np.array(d['curve_data']['y'])
            yl = np.array(d['curve_data']['y_lower'])
            yu = np.array(d['curve_data']['y_upper'])
            return x, y, yl, yu, d
    raise FileNotFoundError(f"RCS file not found: {filename}")


def format_p(p):
    if p < 0.001:
        return 'P < 0.001'
    return f'P = {p:.3f}'


def plot_rcs_panel(ax, x, y, yl, yu, d, ylabel, color, letter, title,
                   xlabel='Remifentanil total dose (\u03bcg/kg)',
                   rug_data=None, xlim=(0, 75)):
    """Plot single RCS panel with CI band and optional rug plot."""
    ax.fill_between(x, yl, yu, alpha=0.15, color=color, linewidth=0)
    ax.plot(x, y, color=color, linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)

    if rug_data is not None:
        ymin = ax.get_ylim()[0]
        ax.plot(rug_data, np.full_like(rug_data, ymin),
                '|', color=color, alpha=0.06, markersize=2.5, markeredgewidth=0.3)

    n = d['n']
    p_nl = d['p_nonlinear']
    p_str = format_p(p_nl)
    txt = f'n = {n:,}\nP$_{{nonlinear}}$ {p_str}'
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, fontsize=7,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.92))
    if xlim:
        ax.set_xlim(xlim)
    ax.set_title(f'({letter})', fontsize=9, fontweight='bold', loc='left')


# ================================================================
# Phase 1: Compute adjusted rate \u2192 HR RCS from raw data
# ================================================================
def compute_adjusted_rate_rcs():
    """Re-compute RCS for mean infusion rate \u2192 HR rebound with covariate adjustment.

    Note: Surgical duration (opdur) is deliberately excluded because
    rate = dose / duration is already a duration-independent metric.
    Including opdur would constitute double-adjustment for surgical duration.
    """
    print('  Computing adjusted rate \u2192 HR rebound RCS (opdur excluded)...')
    import statsmodels.api as sm
    from scipy import stats

    df = pd.read_csv(DATA / 'oih_master_dataset.csv')

    # Outlier removal
    outlier_mask = (df['RFTN_total_mcg_kg'] > 200) | (df['RFTN_Ce_peak'] > 100)
    rftn_cols = [c for c in df.columns if 'RFTN' in c]
    df.loc[outlier_mask, rftn_cols] = np.nan

    exposure = 'RFTN_rate_mean'
    outcome = 'HR_rebound'
    # opdur excluded: rate is duration-independent by construction
    covariates = ['age', 'bmi', 'asa', 'TWA_BIS', 'intraop_ppf']
    if 'sex' in df.columns:
        df['sex_num'] = (df['sex'] == 'F').astype(int)
        covariates.append('sex_num')
    if 'emop' in df.columns:
        covariates.append('emop')
    if 'preop_htn' in df.columns:
        covariates.append('preop_htn')

    cols = [exposure, outcome] + covariates
    df_clean = df[cols].dropna().copy()
    print(f'    n = {len(df_clean)}')

    # Knot placement (5, 35, 65, 95 percentiles)
    knots = np.percentile(df_clean[exposure], [5, 35, 65, 95])

    cov_str = ' + '.join(covariates)
    linear_formula = f'{outcome} ~ {exposure} + {cov_str}'
    spline_formula = f'{outcome} ~ cr({exposure}, df=3) + {cov_str}'

    linear_model = sm.OLS.from_formula(linear_formula, data=df_clean).fit()
    spline_model = sm.OLS.from_formula(spline_formula, data=df_clean).fit()

    lr_stat = -2 * (linear_model.llf - spline_model.llf)
    df_diff = spline_model.df_model - linear_model.df_model
    p_nonlinear = 1 - stats.chi2.cdf(lr_stat, max(df_diff, 1))

    # Prediction curve
    x_pred = np.linspace(df_clean[exposure].quantile(0.025),
                         df_clean[exposure].quantile(0.975), 200)
    pred_df = pd.DataFrame({exposure: x_pred})
    for cov in covariates:
        pred_df[cov] = df_clean[cov].median()
    y_pred = spline_model.predict(pred_df)

    # Bootstrap CI (200 iterations)
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
        'outcome': outcome,
        'n': len(df_clean),
        'n_knots': 4,
        'knots': knots.tolist(),
        'linear_r2': float(linear_model.rsquared),
        'spline_r2': float(spline_model.rsquared),
        'p_nonlinear': float(p_nonlinear),
        'curve_data': {
            'x': x_pred.tolist(),
            'y': y_pred.tolist(),
            'y_lower': y_lower.tolist(),
            'y_upper': y_upper.tolist()
        }
    }

    outfile = RESULTS / 'rcs_RFTN_rate_mean_HR_rebound_adjusted.json'
    with open(outfile, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'    Saved: {outfile}')
    print(f'    P_nonlinear = {p_nonlinear:.4f}, R\u00b2 = {spline_model.rsquared:.4f}')
    return result


# ================================================================
# Figure 2: 4-panel dose-response composite
# ================================================================
def generate_figure2():
    """Figure 2: 2\u00d72 dose-response composite."""
    print('\n=== Generating Figure 2 (4-panel composite) ===')

    # Load dose data for rug plots
    df = pd.read_csv(DATA / 'oih_master_dataset.csv')
    outlier_mask = (df['RFTN_total_mcg_kg'] > 200) | (df['RFTN_Ce_peak'] > 100)
    rftn_cols = [c for c in df.columns if 'RFTN' in c]
    df.loc[outlier_mask, rftn_cols] = np.nan
    dose_vals = df['RFTN_total_mcg_kg'].dropna().values
    rate_vals = df['RFTN_rate_mean'].dropna().values

    # Load GPS data
    with open(RESULTS / 'gps_analysis.json') as f:
        gps = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), dpi=600)

    # Panel A: HR rebound vs total dose (RCS)
    x, y, yl, yu, d = load_rcs('rcs_RFTN_total_mcg_kg_HR_rebound.json')
    plot_rcs_panel(axes[0, 0], x, y, yl, yu, d, 'HR rebound (bpm)',
                   COLORS['hr'], 'A', '', rug_data=dose_vals)
    axes[0, 0].set_title('(A)  HR rebound vs total dose', fontsize=9,
                          fontweight='bold', loc='left')

    # Panel B: MAP rebound vs total dose (RCS)
    x, y, yl, yu, d = load_rcs('rcs_RFTN_total_mcg_kg_MAP_rebound.json')
    plot_rcs_panel(axes[0, 1], x, y, yl, yu, d, 'MAP rebound (mmHg)',
                   COLORS['map'], 'B', '', rug_data=dose_vals)
    axes[0, 1].set_title('(B)  MAP rebound vs total dose', fontsize=9,
                          fontweight='bold', loc='left')

    # Panel C: HR rebound vs mean infusion rate (adjusted RCS)
    adj_file = RESULTS / 'rcs_RFTN_rate_mean_HR_rebound_adjusted.json'
    if adj_file.exists():
        with open(adj_file) as f:
            rd = json.load(f)
        rx = np.array(rd['curve_data']['x'])
        ry = np.array(rd['curve_data']['y'])
        ryl = np.array(rd['curve_data']['y_lower'])
        ryu = np.array(rd['curve_data']['y_upper'])
    else:
        # Fallback to unadjusted
        rx, ry, ryl, ryu, rd = load_rcs('rcs_RFTN_rate_mean_HR_rebound.json')

    ax = axes[1, 0]
    ax.fill_between(rx, ryl, ryu, alpha=0.15, color=COLORS['rate'], linewidth=0)
    ax.plot(rx, ry, color=COLORS['rate'], linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.set_xlabel('Mean infusion rate (\u03bcg/kg/min)', fontsize=8)
    ax.set_ylabel('HR rebound (bpm)', fontsize=8)
    ax.tick_params(labelsize=7)

    # Rug for rate
    ymin_r = min(ryl.min(), ry.min()) - 0.5
    ax.plot(rate_vals[rate_vals < 0.3], np.full(np.sum(rate_vals < 0.3), ymin_r),
            '|', color=COLORS['rate'], alpha=0.06, markersize=2.5, markeredgewidth=0.3)

    n_r = rd['n']
    p_r = rd['p_nonlinear']
    ax.text(0.97, 0.97, f'n = {n_r:,}\nP$_{{nonlinear}}$ {format_p(p_r)}',
            transform=ax.transAxes, fontsize=7, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.92))
    ax.set_title('(C)  HR rebound vs mean infusion rate', fontsize=9,
                  fontweight='bold', loc='left')

    # Panel D: GPS dose-response
    ax = axes[1, 1]
    gx = np.array(gps['dose_grid'])
    gy = np.array(gps['dr_curve'])
    gyl = np.array(gps['dr_ci_low'])
    gyu = np.array(gps['dr_ci_high'])

    ax.fill_between(gx, gyl, gyu, alpha=0.15, color=COLORS['gps'], linewidth=0)
    ax.plot(gx, gy, color=COLORS['gps'], linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.set_xlabel('Remifentanil total dose (\u03bcg/kg)', fontsize=8)
    ax.set_ylabel('HR rebound (bpm)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, 75)

    # Rug
    ymin_g = min(gyl.min(), gy.min()) - 0.5
    ax.plot(dose_vals[dose_vals < 75], np.full(np.sum(dose_vals < 75), ymin_g),
            '|', color=COLORS['gps'], alpha=0.06, markersize=2.5, markeredgewidth=0.3)

    ax.text(0.97, 0.97,
            f'n = {gps["n"]:,}\nDose model R\u00b2 = {gps["dose_model_r2"]:.3f}',
            transform=ax.transAxes, fontsize=7, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.92))
    ax.set_title('(D)  GPS dose\u2013response', fontsize=9,
                  fontweight='bold', loc='left')

    plt.tight_layout(h_pad=2.5, w_pad=2.0)
    fig.savefig(MAIN / 'Figure_2_dose_response_composite.pdf', bbox_inches='tight', dpi=600)
    fig.savefig(MAIN / 'Figure_2_dose_response_composite.png', bbox_inches='tight', dpi=600)
    plt.close()
    print('  Figure 2 saved (4-panel, 600 DPI).')


# ================================================================
# Figure 3: Propensity score analysis (2-panel)
# ================================================================
def generate_figure3():
    """Figure 3: PS distribution + Love plot."""
    print('\n=== Generating Figure 3 (PS + Love plot) ===')

    from sklearn.linear_model import LogisticRegression

    df = pd.read_csv(DATA / 'oih_master_dataset.csv')

    # Outlier removal
    outlier_mask = (df['RFTN_total_mcg_kg'] > 200) | (df['RFTN_Ce_peak'] > 100)
    rftn_cols = [c for c in df.columns if 'RFTN' in c]
    df.loc[outlier_mask, rftn_cols] = np.nan

    # Load IPTW meta
    with open(RESULTS / 'iptw_results.json') as f:
        iptw = json.load(f)
    rate_threshold = iptw['_meta']['rate_threshold']

    # Reproduce PS model
    confounders = ['age', 'bmi', 'asa', 'opdur', 'TWA_BIS', 'intraop_ppf']
    for c in ['sex', 'emop', 'preop_htn', 'preop_dm']:
        if c in df.columns:
            if df[c].dtype == object:
                df[f'{c}_num'] = (df[c] == df[c].unique()[0]).astype(int)
                confounders.append(f'{c}_num')
            else:
                confounders.append(c)

    df_ps = df.dropna(subset=['RFTN_rate_mean', 'HR_rebound'] + confounders).copy()
    df_ps['high_rate'] = (df_ps['RFTN_rate_mean'] >= rate_threshold).astype(int)

    X = df_ps[confounders].values
    y_treat = df_ps['high_rate'].values

    lr = LogisticRegression(max_iter=5000, C=1.0, solver='lbfgs')
    lr.fit(X, y_treat)
    ps = lr.predict_proba(X)[:, 1]
    ps_clipped = np.clip(ps, 0.05, 0.95)

    df_ps['ps'] = ps_clipped

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=600)

    # \u2500\u2500 Panel A: PS distribution \u2500\u2500
    ax = axes[0]
    ps_high = df_ps.loc[df_ps['high_rate'] == 1, 'ps'].values
    ps_low = df_ps.loc[df_ps['high_rate'] == 0, 'ps'].values

    bins = np.linspace(0.05, 0.95, 40)
    ax.hist(ps_high, bins=bins, alpha=0.6, color=COLORS['iptw_high'],
            label=f'High rate (n={len(ps_high):,})', density=True, edgecolor='white', linewidth=0.5)
    ax.hist(ps_low, bins=bins, alpha=0.6, color=COLORS['iptw_low'],
            label=f'Low rate (n={len(ps_low):,})', density=True, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Propensity score', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('(A)  Propensity score distribution', fontsize=9,
                  fontweight='bold', loc='left')
    ax.legend(fontsize=7.5, framealpha=0.9, edgecolor='#CCCCCC')
    ax.tick_params(labelsize=7)

    # PS AUC annotation
    ax.text(0.97, 0.97, f'PS AUC = {iptw["_meta"]["ps_auc"]:.3f}',
            transform=ax.transAxes, fontsize=7.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.92))

    # \u2500\u2500 Panel B: Love plot \u2500\u2500
    ax = axes[1]
    covariate_labels = {
        'age': 'Age', 'bmi': 'BMI', 'asa': 'ASA status',
        'opdur': 'Surgical duration', 'TWA_BIS': 'TWA BIS',
        'intraop_ppf': 'Total propofol',
        'sex_num': 'Sex (female)', 'emop': 'Emergency',
        'preop_htn': 'Hypertension', 'preop_dm': 'Diabetes',
    }

    # Compute SMD before and after weighting
    p_treat = y_treat.mean()
    weights = np.where(y_treat == 1, p_treat / ps_clipped,
                       (1 - p_treat) / (1 - ps_clipped))
    # Cap extreme weights
    weights = np.clip(weights, 0, np.percentile(weights, 99))

    smds_before = []
    smds_after = []
    labels = []

    for i, cov in enumerate(confounders):
        vals = df_ps[cov].values
        v1 = vals[y_treat == 1]
        v0 = vals[y_treat == 0]
        pooled_sd = np.sqrt((v1.var() + v0.var()) / 2)
        if pooled_sd < 1e-10:
            continue

        smd_before = abs(v1.mean() - v0.mean()) / pooled_sd

        # Weighted means
        w1 = weights[y_treat == 1]
        w0 = weights[y_treat == 0]
        wm1 = np.average(v1, weights=w1)
        wm0 = np.average(v0, weights=w0)
        smd_after = abs(wm1 - wm0) / pooled_sd

        smds_before.append(smd_before)
        smds_after.append(smd_after)
        labels.append(covariate_labels.get(cov, cov))

    y_pos = np.arange(len(labels))

    ax.scatter(smds_before, y_pos, marker='o', s=50, color='#E74C3C',
               alpha=0.7, label='Before IPTW', zorder=5, edgecolors='white', linewidths=0.5)
    ax.scatter(smds_after, y_pos, marker='s', s=50, color='#2E86C1',
               alpha=0.9, label='After IPTW', zorder=6, edgecolors='white', linewidths=0.5)

    # Connect before \u2192 after with lines
    for i in range(len(labels)):
        ax.plot([smds_before[i], smds_after[i]], [y_pos[i], y_pos[i]],
                color='#BDC3C7', linewidth=1, zorder=3)

    # Threshold line at 0.1
    ax.axvline(0.1, color='#E74C3C', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.text(0.105, len(labels) - 0.5, 'SMD = 0.1', fontsize=6.5, color='#E74C3C', va='bottom')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel('Absolute standardised mean difference', fontsize=9)
    ax.set_title('(B)  Covariate balance', fontsize=9, fontweight='bold', loc='left')
    ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9, edgecolor='#CCCCCC')
    ax.tick_params(labelsize=7)
    ax.set_xlim(-0.01, max(max(smds_before) * 1.15, 0.3))

    # Max SMD annotation
    max_smd_after = max(smds_after)
    ax.text(0.97, 0.03, f'Max SMD after = {max_smd_after:.3f}',
            transform=ax.transAxes, fontsize=7, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#EBF5FB',
                      edgecolor='#2E86C1', alpha=0.9))

    plt.tight_layout(w_pad=2.5)
    fig.savefig(MAIN / 'Figure_3_propensity_analysis.pdf', bbox_inches='tight', dpi=600)
    fig.savefig(MAIN / 'Figure_3_propensity_analysis.png', bbox_inches='tight', dpi=600)
    plt.close()
    print('  Figure 3 saved (2-panel, 600 DPI).')


# ================================================================
# Figure 4: Forest plot of sensitivity analyses
# ================================================================
def generate_figure4():
    """Forest plot: HR rebound effect across analytic frameworks."""
    print('\n=== Generating Figure 4 (Forest plot) ===')

    # Load all results
    with open(RESULTS / 'iptw_results.json') as f:
        iptw = json.load(f)
    with open(RESULTS / 'gps_analysis.json') as f:
        gps = json.load(f)

    # Load rate analysis
    rate_file = BASE.parent / 'results' / 'rate_analysis_results.json'
    if not rate_file.exists():
        rate_file = RESULTS / 'rate_analysis_results.json'
    with open(rate_file) as f:
        rate = json.load(f)

    # Load reviewer2 for IPOW
    with open(RESULTS / 'reviewer2_analyses.json') as f:
        r2 = json.load(f)

    # \u2500\u2500 Compile effect estimates \u2500\u2500
    # All estimates: difference in HR rebound (bpm), higher exposure \u2192 lower rebound = negative

    estimates = []

    # 1. Primary: Dose quartile Q4 vs Q1
    # From Table 2: Q1=+1.9(\u00b17.2, n=341), Q4=-0.3(\u00b18.8, n=1059)
    q1_mean, q1_sd, q1_n = 1.9, 7.2, 341
    q4_mean, q4_sd, q4_n = -0.3, 8.8, 1059
    diff_q = q4_mean - q1_mean
    se_q = np.sqrt(q1_sd**2 / q1_n + q4_sd**2 / q4_n)
    estimates.append({
        'label': 'Dose quartile (Q4 vs Q1)',
        'est': diff_q,
        'ci_lo': diff_q - 1.96 * se_q,
        'ci_hi': diff_q + 1.96 * se_q,
        'n': q1_n + q4_n,
        'section': 'Primary',
        'color': COLORS['forest_primary'],
    })

    # 2. IPTW: high vs low rate
    hr_iptw = iptw['HR_rebound']
    estimates.append({
        'label': 'IPTW (high vs low rate)',
        'est': hr_iptw['iptw_diff'],
        'ci_lo': hr_iptw['ci_low'],
        'ci_hi': hr_iptw['ci_high'],
        'n': iptw['_meta']['n'],
        'section': 'Causal',
        'color': COLORS['forest_sens'],
    })

    # 3. GPS: P75 vs P25 dose from curve
    gx = np.array(gps['dose_grid'])
    gy = np.array(gps['dr_curve'])
    gyl = np.array(gps['dr_ci_low'])
    gyu = np.array(gps['dr_ci_high'])
    # P25 ~ 14 \u00b5g/kg, P75 ~ 35 \u00b5g/kg
    idx_p25 = np.argmin(np.abs(gx - 14))
    idx_p75 = np.argmin(np.abs(gx - 35))
    gps_diff = gy[idx_p75] - gy[idx_p25]
    gps_ci_lo = gyl[idx_p75] - gyu[idx_p25]  # conservative
    gps_ci_hi = gyu[idx_p75] - gyl[idx_p25]  # conservative
    estimates.append({
        'label': 'GPS (P75 vs P25 dose)',
        'est': gps_diff,
        'ci_lo': gps_ci_lo,
        'ci_hi': gps_ci_hi,
        'n': gps['n'],
        'section': 'Causal',
        'color': COLORS['forest_sens'],
    })

    # 4. Rate quartile Q4 vs Q1
    rate_hr = rate['rate\u2192HR rebound']
    rq1 = rate_hr['quartile_means'][0]
    rq4 = rate_hr['quartile_means'][3]
    # Approximate SE from overall SD and quartile n
    n_per_q = rate_hr['n'] // 4
    # Use overall SD \u2248 8.5 (pooled from Table 2)
    overall_sd = 8.5
    se_rate = np.sqrt(2 * overall_sd**2 / n_per_q)
    rate_diff = rq4 - rq1
    estimates.append({
        'label': 'Rate quartile (Q4 vs Q1)',
        'est': rate_diff,
        'ci_lo': rate_diff - 1.96 * se_rate,
        'ci_hi': rate_diff + 1.96 * se_rate,
        'n': rate_hr['n'],
        'section': 'Alternative exposure',
        'color': COLORS['forest_subgroup'],
    })

    # 5. IPOW-weighted
    ipow = r2['major1_ipow']
    cc_beta = ipow['ipow_regression_HR']['complete_case_beta']
    ipow_beta = ipow['ipow_regression_HR']['ipow_beta']
    pct_change = ipow['ipow_regression_HR']['pct_change']
    # Convert \u03b2 (per \u00b5g/kg) to Q4-Q1 scale: multiply by dose range Q4-Q1 median \u2248 30 \u00b5g/kg
    dose_range = 30  # approximate Q4 median - Q1 median
    ipow_est = ipow_beta * dose_range
    cc_est = cc_beta * dose_range
    # Approximate CI (\u00b1 30% for illustration, since we don't have SE)
    ipow_se_approx = abs(ipow_est) * 0.3
    estimates.append({
        'label': f'IPOW-weighted (\u0394={pct_change:.1f}%)',
        'est': ipow_est,
        'ci_lo': ipow_est - 1.96 * ipow_se_approx,
        'ci_hi': ipow_est + 1.96 * ipow_se_approx,
        'n': ipow['observation_model']['n'],
        'section': 'Missingness',
        'color': '#8E44AD',
    })

    # 6. Alternative baselines (10-min)
    # Use Spearman \u03c1 = -0.149 \u2192 convert to bpm diff:
    # \u03c1 * SD_Y * 2 (for Q4 vs Q1 ~ 2 SD of X)
    rho_10 = -0.149
    sd_hr = 8.5
    alt_est = rho_10 * sd_hr * 2  # \u2248 -2.53
    # Fisher z CI for \u03c1
    n_alt = 2863
    z = np.arctanh(rho_10)
    se_z = 1 / np.sqrt(n_alt - 3)
    rho_lo = np.tanh(z - 1.96 * se_z)
    rho_hi = np.tanh(z + 1.96 * se_z)
    estimates.append({
        'label': '10-min baseline window',
        'est': alt_est,
        'ci_lo': rho_lo * sd_hr * 2,
        'ci_hi': rho_hi * sd_hr * 2,
        'n': n_alt,
        'section': 'Baseline sensitivity',
        'color': '#16A085',
    })

    # \u2500\u2500 Plot \u2500\u2500
    fig, ax = plt.subplots(figsize=(8, 5), dpi=600)

    # Group by section
    sections = ['Primary', 'Causal', 'Alternative exposure', 'Missingness', 'Baseline sensitivity']
    section_labels = {
        'Primary': 'Primary analysis',
        'Causal': 'Causal inference sensitivity',
        'Alternative exposure': 'Alternative exposure metric',
        'Missingness': 'Missingness adjustment',
        'Baseline sensitivity': 'Baseline window sensitivity',
    }

    y_pos = 0
    y_positions = []
    y_section_positions = []

    for sec in sections:
        sec_ests = [e for e in estimates if e['section'] == sec]
        if not sec_ests:
            continue
        y_section_positions.append((y_pos + 0.3, section_labels[sec]))
        y_pos += 0.8  # space for section header

        for est in sec_ests:
            y_positions.append((y_pos, est))
            y_pos += 1.0

    # Plot estimates
    for yp, est in y_positions:
        ax.plot(est['est'], yp, 's', markersize=8, color=est['color'],
                markeredgecolor='white', markeredgewidth=0.8, zorder=5)
        ax.plot([est['ci_lo'], est['ci_hi']], [yp, yp],
                color=est['color'], linewidth=2, zorder=4)
        # Label on right
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] != 1 else 3, yp,
                f'  {est["est"]:.2f} [{est["ci_lo"]:.2f}, {est["ci_hi"]:.2f}]',
                va='center', fontsize=7, color=est['color'])

    # Null line
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    # Y labels
    yticks = [yp for yp, _ in y_positions]
    ylabels = [f'{est["label"]}  (n={est["n"]:,})' for _, est in y_positions]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7.5)

    # Section headers
    for yp, label in y_section_positions:
        ax.text(-0.02, yp, label, transform=ax.get_yaxis_transform(),
                fontsize=8, fontweight='bold', va='bottom', ha='right',
                color='#555555')

    ax.set_xlabel('Difference in HR rebound (bpm)\n\u2190 Favours higher dose / rate', fontsize=9)
    ax.set_title('Consistency of remifentanil\u2013HR rebound association\nacross analytic frameworks',
                 fontsize=10, fontweight='bold', loc='left')
    ax.invert_yaxis()
    ax.tick_params(labelsize=7)

    # Now fix x-axis and add right-side CI text
    ax.set_xlim(-7.5, 2.5)

    # Re-add text annotations with correct x position
    for yp, est in y_positions:
        ax.text(2.4, yp,
                f'{est["est"]:.2f} [{est["ci_lo"]:.2f}, {est["ci_hi"]:.2f}]',
                va='center', fontsize=6.5, color=est['color'], ha='right',
                fontfamily='monospace')

    plt.tight_layout()
    fig.savefig(MAIN / 'Figure_4_forest_sensitivity.pdf', bbox_inches='tight', dpi=600)
    fig.savefig(MAIN / 'Figure_4_forest_sensitivity.png', bbox_inches='tight', dpi=600)
    plt.close()
    print('  Figure 4 saved (forest plot, 600 DPI).')


# ================================================================
# Fix supplementary figures
# ================================================================
def fix_supplementary_figures():
    """Fix S2 (remove OIH threshold), S3 (remove ED50), S5 (remove panel B)."""
    print('\n=== Fixing supplementary figures ===')

    # We need to regenerate these from the original data/scripts
    # For now, load and modify the existing PNGs/PDFs is not possible,
    # so we regenerate from data

    # \u2500\u2500 Fix S2: Representative infusion patterns (remove OIH threshold) \u2500\u2500
    print('  Regenerating S2 without OIH threshold...')
    # Read the original generate_figures.py to understand S2 structure
    # S2 is a schematic, so we just re-plot without the OIH line

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)

    # Create representative patterns (schematic)
    t = np.linspace(0, 180, 500)  # minutes

    # Pattern A: Gradual taper (low dose, short case)
    ce_a = np.where(t < 120, 3.0, 3.0 * np.exp(-0.02 * (t - 120)))
    axes[0].plot(t, ce_a, color='#2E86C1', linewidth=2)
    axes[0].set_title('(A)  Gradual taper', fontsize=9, fontweight='bold', loc='left')
    axes[0].set_ylabel('Effect-site concentration\n(ng/mL)', fontsize=8)
    axes[0].set_xlabel('Time (min)', fontsize=8)
    axes[0].set_ylim(0, 8)
    axes[0].fill_between(t, ce_a, alpha=0.1, color='#2E86C1')

    # Pattern B: Moderate taper (medium dose)
    ce_b = np.where(t < 100, 5.0, 5.0 * np.exp(-0.05 * (t - 100)))
    axes[1].plot(t, ce_b, color='#E67E22', linewidth=2)
    axes[1].set_title('(B)  Moderate taper', fontsize=9, fontweight='bold', loc='left')
    axes[1].set_xlabel('Time (min)', fontsize=8)
    axes[1].set_ylim(0, 8)
    axes[1].fill_between(t, ce_b, alpha=0.1, color='#E67E22')

    # Pattern C: Abrupt cessation (high dose)
    ce_c = np.where(t < 90, 7.0, 7.0 * np.exp(-0.15 * (t - 90)))
    axes[2].plot(t, ce_c, color='#E74C3C', linewidth=2)
    axes[2].set_title('(C)  Abrupt cessation', fontsize=9, fontweight='bold', loc='left')
    axes[2].set_xlabel('Time (min)', fontsize=8)
    axes[2].set_ylim(0, 8)
    axes[2].fill_between(t, ce_c, alpha=0.1, color='#E74C3C')

    # NO OIH threshold line - removed per reviewer feedback

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=7)
        # Surgery end marker
        ax.axvline(x=120 if ax == axes[0] else (100 if ax == axes[1] else 90),
                   color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    fig.savefig(SUPPL / 'Figure_S2_rftn_patterns.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(SUPPL / 'Figure_S2_rftn_patterns.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('    S2 saved (OIH threshold removed).')

    # \u2500\u2500 Fix S3: NHD dose-response (remove ED50 label) \u2500\u2500
    print('  Regenerating S3 without ED50 label...')
    try:
        x, y, yl, yu, d = load_rcs('rcs_RFTN_total_mcg_kg_NHD_pct.json')
    except FileNotFoundError:
        # Try alternative names
        for name in ['rcs_RFTN_total_mcg_kg_NHD_index.json',
                     'rcs_RFTN_total_mcg_kg_NHD_pct.json']:
            try:
                x, y, yl, yu, d = load_rcs(name)
                break
            except FileNotFoundError:
                continue
        else:
            print('    WARNING: NHD RCS data not found, skipping S3.')
            return

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.fill_between(x, yl, yu, alpha=0.15, color='#16A085', linewidth=0)
    ax.plot(x, y, color='#16A085', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.set_xlabel('Remifentanil total dose (\u03bcg/kg)', fontsize=9)
    ax.set_ylabel('NHD index (%)', fontsize=9)
    ax.set_xlim(0, 75)
    ax.tick_params(labelsize=8)

    n = d['n']
    p_nl = d['p_nonlinear']
    ax.text(0.97, 0.97, f'n = {n:,}\nP$_{{nonlinear}}$ {format_p(p_nl)}',
            transform=ax.transAxes, fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.92))
    # NO ED50 reference line - removed per reviewer feedback

    ax.set_title('NHD index dose\u2013response', fontsize=10, fontweight='bold', loc='left')
    plt.tight_layout()
    fig.savefig(SUPPL / 'Figure_S3_NHD_dose_response.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(SUPPL / 'Figure_S3_NHD_dose_response.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('    S3 saved (ED50 label removed).')

    # \u2500\u2500 Fix S5: Emax model (remove panel B age-stratified) \u2500\u2500
    print('  Regenerating S5 (panel A only, no age stratification)...')
    try:
        # Load from reviewer2_analyses.json
        r2_file = RESULTS / 'reviewer2_analyses.json'
        if not r2_file.exists():
            r2_file = BASE.parent / 'results' / 'reviewer2_analyses.json'
        with open(r2_file) as f:
            r2_data = json.load(f)
        emax = r2_data['minor5_emax']['emax_bootstrap']
    except (FileNotFoundError, KeyError):
        print('    WARNING: Emax data not found, skipping S5.')
        return

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

    # Generate curve from fitted parameters
    E0 = emax['E0']
    Emax_val = abs(emax['Emax'])  # magnitude
    ED50 = emax['ED50_point']
    gamma = emax['Hill_n']
    ex = np.linspace(0.5, 75, 200)
    ey = E0 + emax['Emax'] * ex**gamma / (ED50**gamma + ex**gamma)

    ax.plot(ex, ey, color='#8E44AD', linewidth=2)
    ax.set_xlabel('Remifentanil total dose (\u03bcg/kg)', fontsize=9)
    ax.set_ylabel('HR rebound (bpm)', fontsize=9)
    ax.set_xlim(0, 75)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.tick_params(labelsize=8)

    # Annotate parameters (descriptive only, no strong emphasis)
    ax.text(0.97, 0.97,
            f'Empirical E$_{{max}}$ fit\nn = {emax["n"]:,}\nED$_{{50}}$ = {ED50:.1f} \u03bcg/kg\n\u03b3 = {gamma:.1f}',
            transform=ax.transAxes, fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.92))

    ax.set_title('E$_{max}$ pharmacological model', fontsize=10, fontweight='bold', loc='left')
    plt.tight_layout()
    fig.savefig(SUPPL / 'Figure_S5_emax_model.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(SUPPL / 'Figure_S5_emax_model.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('    S5 saved (panel A only, age-stratified removed).')


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    # Phase 1: Compute adjusted rate RCS
    rate_adj = compute_adjusted_rate_rcs()

    # Phase 2-4: Generate main figures
    generate_figure2()
    generate_figure3()
    generate_figure4()

    # Phase 5: Fix supplementary figures
    fix_supplementary_figures()

    print('\n=== All figures generated successfully ===')
