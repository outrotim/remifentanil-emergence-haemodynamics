#!/usr/bin/env python3
"""
=================================================================
OIH Study - Phase 4: Reviewer Response Supplementary Analyses
=================================================================
Supplementary analyses requested during peer review:

Analysis 1: Missingness Report (Major 8)
  - Variable-level missingness counts and percentages
  - Missingness rates by RFTN dose quartile (chi-squared)
  - Baseline comparison: included vs excluded (SMD)

Analysis 2: Complete-Case OWSI Sensitivity (Major 3)
  - OWSI_complete (all 3 components non-missing) vs OWSI_partial
  - RCS with 4 knots for both versions

Analysis 3: Taper Dynamics Expansion (Major 4)
  a) Taper slope quartile analysis
  b) DeltaCe_end metric
  c) Multivariable taper model

Analysis 4: Expanded Covariates (Major 5)
  - RCS with additional confounders (eph, phe, ebl)
  - IPTW with expanded PS model

Analysis 5: Binary Clinical Endpoints (Major 9)
  - HR_increase_20pct, MAP_increase_20pct, Any_vasopressor
  - Composite event
  - Logistic regression with dose and taper exposures
=================================================================
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from datetime import datetime
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
OIH_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper: JSON serializer for numpy types
# ============================================================
def _np_serializer(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    return str(obj)


def save_json(data, filename):
    """Save results dict to JSON file."""
    filepath = RESULTS_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=_np_serializer)
    print(f"  -> Saved to {filepath}")


# ============================================================
# Helper: Standardized Mean Difference
# ============================================================
def compute_smd(group1, group0):
    """
    Compute standardized mean difference (Cohen's d pooled).
    group1, group0: array-like of values for each group.
    Returns SMD (positive = group1 higher).
    """
    g1 = np.asarray(group1, dtype=float)
    g0 = np.asarray(group0, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g0 = g0[~np.isnan(g0)]
    if len(g1) < 2 or len(g0) < 2:
        return np.nan
    m1, m0 = g1.mean(), g0.mean()
    s1, s0 = g1.std(ddof=1), g0.std(ddof=1)
    pooled_sd = np.sqrt((s1**2 + s0**2) / 2)
    if pooled_sd < 1e-12:
        return 0.0
    return (m1 - m0) / pooled_sd


# ============================================================
# Helper: RCS fitting (restricted cubic splines via patsy)
# ============================================================
def fit_rcs(x, y, covariate_df=None, n_knots=4, knot_percentiles=None):
    """
    Fit restricted cubic splines model.

    Parameters
    ----------
    x : array-like, primary exposure
    y : array-like, outcome
    covariate_df : pd.DataFrame or None, additional covariates
    n_knots : int
    knot_percentiles : list of percentiles for knot placement

    Returns
    -------
    dict with r2, r2_linear, p_nonlinear, knots, n
    """
    try:
        from patsy import dmatrix

        if knot_percentiles is None:
            knot_percentiles = [5, 35, 65, 95]

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        knots = np.percentile(x, knot_percentiles)
        knot_str = ', '.join([f'{k:.6f}' for k in knots])

        df_data = pd.DataFrame({'x': x, 'y': y})
        cov_terms = ''
        if covariate_df is not None:
            for col in covariate_df.columns:
                df_data[col] = covariate_df[col].values
            cov_terms = ' + '.join(covariate_df.columns)

        if cov_terms:
            lin_formula = f'y ~ x + {cov_terms}'
            rcs_formula = f'y ~ cr(x, knots=[{knot_str}]) + {cov_terms}'
        else:
            lin_formula = 'y ~ x'
            rcs_formula = f'y ~ cr(x, knots=[{knot_str}])'

        lin_model = sm.OLS.from_formula(lin_formula, df_data).fit()
        rcs_model = sm.OLS.from_formula(rcs_formula, df_data).fit()

        df_diff = rcs_model.df_model - lin_model.df_model
        if df_diff > 0:
            lr_stat = -2 * (lin_model.llf - rcs_model.llf)
            p_nonlinear = 1 - stats.chi2.cdf(lr_stat, df_diff)
        else:
            p_nonlinear = 1.0

        return {
            'n': int(len(x)),
            'r2': float(rcs_model.rsquared),
            'r2_adj': float(rcs_model.rsquared_adj),
            'r2_linear': float(lin_model.rsquared),
            'p_nonlinear': float(p_nonlinear),
            'knots': [float(k) for k in knots],
            'aic_linear': float(lin_model.aic),
            'aic_rcs': float(rcs_model.aic),
        }
    except Exception as e:
        print(f"    [RCS ERROR] {e}")
        return None


# ============================================================
# Helper: IPTW with bootstrap CI
# ============================================================
def run_iptw(df, treatment_col, outcome_col, confounder_cols, n_boot=1000):
    """
    Run IPTW analysis with stabilized weights and bootstrap CI.

    Returns dict with ps_auc, max_smd_after, iptw_diff, ci_low, ci_high.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    needed = confounder_cols + [treatment_col, outcome_col]
    sub = df[needed].dropna()
    if len(sub) < 200:
        return {'error': f'Insufficient data (n={len(sub)})'}

    X = sub[confounder_cols].values
    T = sub[treatment_col].values.astype(int)
    Y = sub[outcome_col].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    ps_model = LogisticRegression(max_iter=5000, C=1.0)
    ps_model.fit(X_sc, T)
    ps = ps_model.predict_proba(X_sc)[:, 1]
    ps = np.clip(ps, 0.05, 0.95)

    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(T, ps)
    except Exception:
        auc = 0.5

    # Stabilized weights
    p_treat = T.mean()
    w_stable = np.where(T == 1, p_treat / ps, (1 - p_treat) / (1 - ps))

    # SMD after weighting
    smd_after_list = []
    for i in range(X.shape[1]):
        m1 = np.average(X[T == 1, i], weights=w_stable[T == 1])
        m0 = np.average(X[T == 0, i], weights=w_stable[T == 0])
        v1 = np.average((X[T == 1, i] - m1)**2, weights=w_stable[T == 1])
        v0 = np.average((X[T == 0, i] - m0)**2, weights=w_stable[T == 0])
        denom = np.sqrt((v1 + v0) / 2)
        smd_val = abs((m1 - m0) / denom) if denom > 1e-12 else 0.0
        smd_after_list.append(smd_val)

    max_smd_after = max(smd_after_list)

    # IPTW difference
    iptw_mean1 = np.average(Y[T == 1], weights=w_stable[T == 1])
    iptw_mean0 = np.average(Y[T == 0], weights=w_stable[T == 0])
    iptw_diff = iptw_mean1 - iptw_mean0

    # Bootstrap CI
    rng = np.random.RandomState(42)
    diffs = []
    n = len(Y)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yb, tb, wb = Y[idx], T[idx], w_stable[idx]
        if tb.sum() > 0 and (1 - tb).sum() > 0:
            m1b = np.average(yb[tb == 1], weights=wb[tb == 1])
            m0b = np.average(yb[tb == 0], weights=wb[tb == 0])
            diffs.append(m1b - m0b)
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))

    return {
        'n': int(len(sub)),
        'ps_auc': float(auc),
        'max_smd_after': float(max_smd_after),
        'mean_smd_after': float(np.mean(smd_after_list)),
        'iptw_diff': float(iptw_diff),
        'ci_low': ci_low,
        'ci_high': ci_high,
        'significant': bool(ci_low > 0 or ci_high < 0),
        'weight_range': [float(w_stable.min()), float(w_stable.max())],
    }


# ============================================================
# Data loading & cleaning (matching existing scripts)
# ============================================================
def load_clean_data():
    """Load and clean master dataset with standard outlier removal."""
    print("=" * 70)
    print("  Loading master dataset...")
    print("=" * 70)

    df = pd.read_csv(OIH_DIR / "oih_master_dataset.csv")
    print(f"  Loaded: {len(df)} cases x {len(df.columns)} columns")

    # Outlier cleanup: extreme RFTN values -> set RFTN cols to NaN
    outlier_mask = (
        (df['RFTN_total_mcg_kg'] > 200) |
        (df['RFTN_Ce_peak'] > 100)
    )
    rftn_cols = [c for c in df.columns if c.startswith('RFTN_') or c in ['rftn_conc']]
    outlier_ids = df.loc[outlier_mask, 'caseid'].tolist()
    df.loc[outlier_mask, rftn_cols] = np.nan
    print(f"  Outlier cleanup: {outlier_mask.sum()} cases (caseids: {outlier_ids}) "
          f"-> RFTN columns set to NaN")
    print(f"  Valid RFTN cases: {df['RFTN_total_mcg_kg'].notna().sum()}")

    # Sex encoding
    df['sex_num'] = (df['sex'] == 'F').astype(int)

    # RFTN dose quartile
    df['RFTN_quartile'] = pd.qcut(
        df['RFTN_total_mcg_kg'].dropna(), q=4, labels=['Q1', 'Q2', 'Q3', 'Q4']
    ).reindex(df.index)

    # Build OSI/OWSI from Z-standardized components
    oih_components = ['HR_rebound', 'MAP_rebound', 'FTN_rescue_mcg_kg']
    for comp in oih_components:
        if comp in df.columns:
            m, s = df[comp].mean(), df[comp].std()
            df[f'{comp}_Z'] = (df[comp] - m) / s if s > 0 else 0.0

    z_cols = [f'{c}_Z' for c in oih_components if c in df.columns]
    # OWSI_partial: mean of available Z-scores (allows partial missingness)
    df['OWSI'] = df[z_cols].mean(axis=1)

    # Age group
    df['age_group'] = pd.cut(df['age'], bins=[0, 65, 75, 120],
                              labels=['<65', '65-74', '>=75'], right=False)
    df['elderly'] = (df['age'] >= 65).astype(int)

    return df


# ============================================================
# ANALYSIS 1: Missingness Report (Major 8)
# ============================================================
def analysis_1_missingness(df):
    """
    Comprehensive missingness analysis:
    - Variable-level missing counts/percentages
    - Missing rates by RFTN dose quartile (chi-squared)
    - Baseline comparison: included vs excluded for HR_rebound analysis (SMD)
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Missingness Report (Major 8)")
    print("=" * 70)

    results = {}

    # --- 1a: Variable-level missingness ---
    key_vars = [
        'RFTN_total_mcg_kg', 'RFTN_rate_mean', 'RFTN_rate_peak',
        'RFTN_Ce_mean', 'RFTN_Ce_peak', 'RFTN_Ce_at_end',
        'RFTN_taper_slope', 'RFTN_Ce_SD', 'RFTN_Ce_CV', 'RFTN_Ce_ARV',
        'Time_Ce_above_4', 'Time_Ce_above_6', 'Time_Ce_above_8',
        'HR_rebound', 'MAP_rebound', 'HR_rebound_pct', 'MAP_rebound_pct',
        'FTN_rescue_mcg_kg', 'NHD_pct',
        'TWA_BIS', 'SD_BIS', 'CV_BIS', 'ARV_BIS',
        'HR_stable_mean', 'MAP_stable_mean',
        'HR_late_mean', 'MAP_late_mean',
        'HR_post30_mean', 'MAP_post30_mean',
        'age', 'sex', 'bmi', 'asa', 'opdur',
        'intraop_ppf', 'intraop_eph', 'intraop_phe', 'intraop_ebl',
    ]
    # Only include columns that exist
    key_vars = [v for v in key_vars if v in df.columns]

    n_total = len(df)
    var_missing = {}
    print(f"\n  {'Variable':<30s} {'Available':>10s} {'Missing':>10s} {'Missing%':>10s}")
    print("  " + "-" * 62)

    for var in key_vars:
        n_avail = int(df[var].notna().sum())
        n_miss = int(df[var].isna().sum())
        pct_miss = 100.0 * n_miss / n_total
        var_missing[var] = {
            'n_available': n_avail,
            'n_missing': n_miss,
            'pct_missing': round(pct_miss, 2),
        }
        print(f"  {var:<30s} {n_avail:>10d} {n_miss:>10d} {pct_miss:>9.1f}%")

    results['variable_missingness'] = var_missing

    # --- 1b: Missing rates by RFTN dose quartile ---
    print(f"\n  Missing rates by RFTN dose quartile (chi-squared test):")
    print(f"  {'Variable':<25s} {'Q1%':>8s} {'Q2%':>8s} {'Q3%':>8s} {'Q4%':>8s} {'chi2':>8s} {'P':>10s}")
    print("  " + "-" * 70)

    quartile_missing = {}
    q_labels = ['Q1', 'Q2', 'Q3', 'Q4']
    # Only among those with valid RFTN quartile
    df_q = df[df['RFTN_quartile'].notna()].copy()

    test_vars = ['HR_rebound', 'MAP_rebound', 'FTN_rescue_mcg_kg', 'NHD_pct',
                 'RFTN_Ce_at_end', 'RFTN_taper_slope', 'TWA_BIS', 'CV_BIS',
                 'intraop_ebl', 'HR_post30_mean', 'MAP_post30_mean']
    test_vars = [v for v in test_vars if v in df.columns]

    for var in test_vars:
        q_miss_pcts = []
        q_counts = []  # (missing, not_missing) for contingency table
        for q in q_labels:
            sub = df_q[df_q['RFTN_quartile'] == q]
            n_q = len(sub)
            n_miss_q = int(sub[var].isna().sum())
            pct = 100.0 * n_miss_q / n_q if n_q > 0 else 0.0
            q_miss_pcts.append(round(pct, 1))
            q_counts.append([n_miss_q, n_q - n_miss_q])

        # Chi-squared test on contingency table
        contingency = np.array(q_counts)
        if contingency.min() >= 0 and contingency.sum() > 0:
            try:
                chi2, p_chi2, _, _ = stats.chi2_contingency(contingency)
            except ValueError:
                chi2, p_chi2 = np.nan, np.nan
        else:
            chi2, p_chi2 = np.nan, np.nan

        p_str = f"{p_chi2:.4f}" if not np.isnan(p_chi2) and p_chi2 >= 0.001 else "<0.001"
        if np.isnan(p_chi2):
            p_str = "N/A"
        chi2_str = f"{chi2:.2f}" if not np.isnan(chi2) else "N/A"

        print(f"  {var:<25s} {q_miss_pcts[0]:>7.1f}% {q_miss_pcts[1]:>7.1f}% "
              f"{q_miss_pcts[2]:>7.1f}% {q_miss_pcts[3]:>7.1f}% {chi2_str:>8s} {p_str:>10s}")

        quartile_missing[var] = {
            'pct_by_quartile': {q: pct for q, pct in zip(q_labels, q_miss_pcts)},
            'chi2': float(chi2) if not np.isnan(chi2) else None,
            'p_value': float(p_chi2) if not np.isnan(p_chi2) else None,
        }

    results['quartile_missingness'] = quartile_missing

    # --- 1c: Included vs excluded comparison for HR_rebound analysis ---
    print(f"\n  Baseline comparison: included vs excluded for HR_rebound analysis (SMD):")

    df['_included_hr'] = df['HR_rebound'].notna().astype(int)
    n_incl = int(df['_included_hr'].sum())
    n_excl = int((1 - df['_included_hr']).sum())
    print(f"  Included: n={n_incl}, Excluded: n={n_excl}")

    comparison_vars = [
        ('age', 'Age', 'continuous'),
        ('sex_num', 'Female sex', 'binary'),
        ('bmi', 'BMI', 'continuous'),
        ('asa', 'ASA', 'continuous'),
        ('opdur', 'Surgery duration', 'continuous'),
        ('RFTN_total_mcg_kg', 'RFTN total dose', 'continuous'),
        ('intraop_ppf', 'Propofol', 'continuous'),
        ('intraop_ebl', 'Blood loss', 'continuous'),
    ]

    print(f"\n  {'Variable':<25s} {'Included':>15s} {'Excluded':>15s} {'SMD':>8s} {'Balanced':>10s}")
    print("  " + "-" * 75)

    baseline_comparison = {}
    for var, label, vtype in comparison_vars:
        if var not in df.columns:
            continue
        incl_vals = df.loc[df['_included_hr'] == 1, var].dropna()
        excl_vals = df.loc[df['_included_hr'] == 0, var].dropna()

        smd_val = compute_smd(incl_vals.values, excl_vals.values)
        balanced = "|SMD|<0.1" if abs(smd_val) < 0.1 else "|SMD|>=0.1"

        if vtype == 'continuous':
            incl_str = f"{incl_vals.mean():.1f} ({incl_vals.std():.1f})"
            excl_str = f"{excl_vals.mean():.1f} ({excl_vals.std():.1f})"
        else:
            incl_str = f"{incl_vals.mean()*100:.1f}%"
            excl_str = f"{excl_vals.mean()*100:.1f}%"

        print(f"  {label:<25s} {incl_str:>15s} {excl_str:>15s} {smd_val:>+8.3f} {balanced:>10s}")

        baseline_comparison[var] = {
            'label': label,
            'included_mean': float(incl_vals.mean()),
            'included_sd': float(incl_vals.std()),
            'excluded_mean': float(excl_vals.mean()),
            'excluded_sd': float(excl_vals.std()),
            'smd': float(smd_val),
            'balanced': abs(smd_val) < 0.1,
            'n_included': int(len(incl_vals)),
            'n_excluded': int(len(excl_vals)),
        }

    results['baseline_comparison'] = baseline_comparison
    results['_meta'] = {
        'n_total': n_total,
        'n_valid_rftn': int(df['RFTN_total_mcg_kg'].notna().sum()),
        'n_included_hr_rebound': n_incl,
        'n_excluded_hr_rebound': n_excl,
    }

    # Clean up temp column
    df.drop(columns=['_included_hr'], inplace=True)

    save_json(results, 'missingness_report.json')
    return results


# ============================================================
# ANALYSIS 2: Complete-Case OWSI Sensitivity (Major 3)
# ============================================================
def analysis_2_owsi_sensitivity(df):
    """
    Compare OWSI computed with partial data (mean of available Z-scores)
    vs OWSI_complete (only cases with ALL 3 components present).
    Run RCS for both.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Complete-Case OWSI Sensitivity (Major 3)")
    print("=" * 70)

    results = {}
    oih_components = ['HR_rebound', 'MAP_rebound', 'FTN_rescue_mcg_kg']
    z_cols = [f'{c}_Z' for c in oih_components]

    # OWSI_partial already computed in load_clean_data (df['OWSI'])
    # Create OWSI_complete: require ALL 3 Z-scores non-missing
    all_present_mask = df[z_cols].notna().all(axis=1)
    df['OWSI_complete'] = np.nan
    df.loc[all_present_mask, 'OWSI_complete'] = df.loc[all_present_mask, z_cols].mean(axis=1)

    n_partial = int(df['OWSI'].notna().sum())
    n_complete = int(df['OWSI_complete'].notna().sum())
    print(f"  OWSI_partial (>=1 component): n = {n_partial}")
    print(f"  OWSI_complete (all 3 components): n = {n_complete}")
    print(f"  Lost cases: {n_partial - n_complete}")

    # Descriptive comparison
    if n_partial > 0 and n_complete > 0:
        owsi_p = df['OWSI'].dropna()
        owsi_c = df['OWSI_complete'].dropna()
        print(f"\n  OWSI_partial:  mean={owsi_p.mean():.4f}, SD={owsi_p.std():.4f}, "
              f"median={owsi_p.median():.4f}")
        print(f"  OWSI_complete: mean={owsi_c.mean():.4f}, SD={owsi_c.std():.4f}, "
              f"median={owsi_c.median():.4f}")

        results['descriptive'] = {
            'owsi_partial': {
                'n': n_partial, 'mean': float(owsi_p.mean()),
                'sd': float(owsi_p.std()), 'median': float(owsi_p.median()),
            },
            'owsi_complete': {
                'n': n_complete, 'mean': float(owsi_c.mean()),
                'sd': float(owsi_c.std()), 'median': float(owsi_c.median()),
            },
        }

    # Correlation between partial and complete (on overlapping cases)
    overlap = df[['OWSI', 'OWSI_complete']].dropna()
    if len(overlap) > 10:
        r_corr, p_corr = stats.pearsonr(overlap['OWSI'], overlap['OWSI_complete'])
        print(f"  Correlation (partial vs complete on n={len(overlap)}): r = {r_corr:.4f}, P = {p_corr:.2e}")
        results['correlation_partial_complete'] = {
            'r': float(r_corr), 'p': float(p_corr), 'n': int(len(overlap))
        }

    # RCS analysis for both OWSI versions
    exposure = 'RFTN_total_mcg_kg'
    covariates_list = ['age', 'sex_num', 'bmi', 'asa', 'opdur',
                       'TWA_BIS', 'intraop_ppf']

    # Get top departments for department dummies
    dept_counts = df['department'].value_counts()
    top5_depts = dept_counts.head(5).index.tolist()
    for dept in top5_depts:
        safe_name = f"dept_{dept.replace(' ', '_').replace('/', '_')}"
        df[safe_name] = (df['department'] == dept).astype(int)
    dept_dummy_cols = [f"dept_{d.replace(' ', '_').replace('/', '_')}" for d in top5_depts]

    covariate_cols_full = covariates_list + dept_dummy_cols

    for owsi_name, owsi_col in [('OWSI_partial', 'OWSI'), ('OWSI_complete', 'OWSI_complete')]:
        needed = [exposure, owsi_col] + covariate_cols_full
        needed_existing = [c for c in needed if c in df.columns]
        sub = df[needed_existing].dropna()

        if len(sub) < 100:
            print(f"\n  [{owsi_name}] Insufficient data (n={len(sub)}), skipping RCS.")
            results[owsi_name] = {'error': f'Insufficient data (n={len(sub)})'}
            continue

        print(f"\n  --- RCS: {exposure} -> {owsi_name} (n={len(sub)}) ---")

        cov_df = sub[[c for c in covariate_cols_full if c in sub.columns]].copy()
        rcs_result = fit_rcs(
            sub[exposure].values, sub[owsi_col].values,
            covariate_df=cov_df,
            n_knots=4, knot_percentiles=[5, 35, 65, 95]
        )

        if rcs_result:
            print(f"    n = {rcs_result['n']}")
            print(f"    R² (RCS) = {rcs_result['r2']:.4f}")
            print(f"    R² (linear) = {rcs_result['r2_linear']:.4f}")
            print(f"    P_nonlinear = {rcs_result['p_nonlinear']:.4e}")
            print(f"    Knots = {rcs_result['knots']}")
            results[owsi_name] = rcs_result
        else:
            results[owsi_name] = {'error': 'RCS fitting failed'}

    # Comparison summary
    if 'OWSI_partial' in results and 'OWSI_complete' in results:
        if isinstance(results['OWSI_partial'], dict) and 'r2' in results['OWSI_partial']:
            if isinstance(results['OWSI_complete'], dict) and 'r2' in results['OWSI_complete']:
                r2_diff = results['OWSI_complete']['r2'] - results['OWSI_partial']['r2']
                pnl_partial = results['OWSI_partial']['p_nonlinear']
                pnl_complete = results['OWSI_complete']['p_nonlinear']
                print(f"\n  Summary:")
                print(f"    R² difference (complete - partial): {r2_diff:+.4f}")
                print(f"    P_nonlinear partial: {pnl_partial:.4e}")
                print(f"    P_nonlinear complete: {pnl_complete:.4e}")
                results['comparison'] = {
                    'r2_diff': float(r2_diff),
                    'both_nonlinear': bool(pnl_partial < 0.05 and pnl_complete < 0.05),
                    'conclusion': 'Consistent' if (pnl_partial < 0.05) == (pnl_complete < 0.05) else 'Divergent'
                }

    save_json(results, 'owsi_complete_case.json')
    return results


# ============================================================
# ANALYSIS 3: Taper Dynamics Expansion (Major 4)
# ============================================================
def analysis_3_taper_expansion(df):
    """
    Three sub-analyses expanding taper dynamics:
    a) Taper slope quartile analysis
    b) DeltaCe_end metric
    c) Multivariable taper model
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Taper Dynamics Expansion (Major 4)")
    print("=" * 70)

    results = {}

    # Build OWSI if not already present
    if 'OWSI' not in df.columns:
        oih_components = ['HR_rebound', 'MAP_rebound', 'FTN_rescue_mcg_kg']
        z_cols = [f'{c}_Z' for c in oih_components]
        df['OWSI'] = df[z_cols].mean(axis=1)

    # ---- 3a: Taper slope QUARTILE analysis ----
    print("\n  --- 3a: Taper slope quartile analysis ---")

    taper = df['RFTN_taper_slope'].dropna()
    print(f"  RFTN_taper_slope available: n = {len(taper)}")
    print(f"  Distribution: mean={taper.mean():.4f}, SD={taper.std():.4f}, "
          f"median={taper.median():.4f}")
    print(f"  Range: [{taper.min():.4f}, {taper.max():.4f}]")

    # Create taper quartiles
    df['taper_quartile'] = pd.qcut(
        df['RFTN_taper_slope'].dropna(), q=4,
        labels=['Q1_gradual', 'Q2', 'Q3', 'Q4_abrupt']
    ).reindex(df.index)

    q_labels = ['Q1_gradual', 'Q2', 'Q3', 'Q4_abrupt']
    outcomes = ['HR_rebound', 'MAP_rebound', 'OWSI']

    taper_quartile_results = {}
    print(f"\n  {'Outcome':<20s} {'Q1(gradual)':>14s} {'Q2':>12s} {'Q3':>12s} {'Q4(abrupt)':>14s} "
          f"{'KW_P':>10s} {'P_trend':>10s}")
    print("  " + "-" * 95)

    for outcome in outcomes:
        group_means = []
        group_data = []
        for q in q_labels:
            vals = df.loc[df['taper_quartile'] == q, outcome].dropna()
            group_means.append(float(vals.mean()) if len(vals) > 0 else np.nan)
            group_data.append(vals.values)

        # Kruskal-Wallis
        valid_groups = [g for g in group_data if len(g) > 5]
        if len(valid_groups) >= 2:
            kw_stat, kw_p = stats.kruskal(*valid_groups)
        else:
            kw_stat, kw_p = np.nan, np.nan

        # P for trend (Spearman rank correlation of quartile index with outcome)
        # Use individual-level data, not group means
        taper_sub = df[['taper_quartile', outcome]].dropna()
        if len(taper_sub) > 10:
            # Encode quartile as numeric 1-4
            q_map = {q: i + 1 for i, q in enumerate(q_labels)}
            taper_sub['q_num'] = taper_sub['taper_quartile'].map(q_map)
            r_trend, p_trend = stats.spearmanr(taper_sub['q_num'], taper_sub[outcome])
        else:
            r_trend, p_trend = np.nan, np.nan

        kw_p_str = f"{kw_p:.4f}" if not np.isnan(kw_p) and kw_p >= 0.001 else "<0.001"
        p_trend_str = f"{p_trend:.4f}" if not np.isnan(p_trend) and p_trend >= 0.001 else "<0.001"
        means_str = [f"{m:+.3f}" if not np.isnan(m) else "N/A" for m in group_means]

        print(f"  {outcome:<20s} {means_str[0]:>14s} {means_str[1]:>12s} "
              f"{means_str[2]:>12s} {means_str[3]:>14s} {kw_p_str:>10s} {p_trend_str:>10s}")

        taper_quartile_results[outcome] = {
            'quartile_means': {q: m for q, m in zip(q_labels, group_means)},
            'quartile_ns': {q: int(len(g)) for q, g in zip(q_labels, group_data)},
            'kruskal_wallis_stat': float(kw_stat) if not np.isnan(kw_stat) else None,
            'kruskal_wallis_p': float(kw_p) if not np.isnan(kw_p) else None,
            'spearman_r_trend': float(r_trend) if not np.isnan(r_trend) else None,
            'p_trend': float(p_trend) if not np.isnan(p_trend) else None,
        }

    results['taper_quartile_analysis'] = taper_quartile_results

    # Quartile range descriptions
    for q in q_labels:
        vals = df.loc[df['taper_quartile'] == q, 'RFTN_taper_slope'].dropna()
        if len(vals) > 0:
            print(f"    {q}: slope range [{vals.min():.4f}, {vals.max():.4f}], n={len(vals)}")

    # ---- 3b: DeltaCe_end metric ----
    print("\n  --- 3b: DeltaCe_end metric ---")

    df['DeltaCe_end'] = df['RFTN_Ce_at_end'] - df['RFTN_Ce_mean']
    n_delta = int(df['DeltaCe_end'].notna().sum())
    print(f"  DeltaCe_end = RFTN_Ce_at_end - RFTN_Ce_mean")
    print(f"  Available n = {n_delta} (RFTN_Ce_at_end has many missing)")

    delta_results = {}
    if n_delta > 20:
        delta_vals = df['DeltaCe_end'].dropna()
        print(f"  Distribution: mean={delta_vals.mean():.4f}, SD={delta_vals.std():.4f}, "
              f"median={delta_vals.median():.4f}")

        for outcome in ['HR_rebound', 'MAP_rebound']:
            sub = df[['DeltaCe_end', outcome]].dropna()
            if len(sub) > 20:
                r, p = stats.spearmanr(sub['DeltaCe_end'], sub[outcome])
                print(f"  DeltaCe_end -> {outcome}: Spearman rho={r:+.4f}, P={p:.4e} (n={len(sub)})")
                delta_results[outcome] = {
                    'spearman_rho': float(r),
                    'p_value': float(p),
                    'n': int(len(sub)),
                }
    else:
        print(f"  [SKIP] Insufficient DeltaCe_end data (n={n_delta})")

    delta_results['n_available'] = n_delta
    delta_results['descriptive'] = {
        'mean': float(df['DeltaCe_end'].mean()) if n_delta > 0 else None,
        'sd': float(df['DeltaCe_end'].std()) if n_delta > 0 else None,
        'median': float(df['DeltaCe_end'].median()) if n_delta > 0 else None,
    }
    results['delta_ce_end'] = delta_results

    # ---- 3c: Multivariable taper model ----
    print("\n  --- 3c: Multivariable taper model ---")

    # Department dummies: top 5, rest = reference
    dept_counts = df['department'].value_counts()
    top5_depts = dept_counts.head(5).index.tolist()
    dept_dummy_cols = []
    for dept in top5_depts:
        safe_name = f"dept_{dept.replace(' ', '_').replace('/', '_')}"
        df[safe_name] = (df['department'] == dept).astype(int)
        dept_dummy_cols.append(safe_name)

    print(f"  Top 5 departments: {top5_depts}")
    print(f"  Department dummies: {dept_dummy_cols}")

    # Model: HR_rebound ~ taper_slope + rate_mean + Ce_mean + opdur + dept_dummies
    #                      + age + sex_num + bmi + asa + intraop_ppf + eph + phe
    model_vars = (['RFTN_taper_slope', 'RFTN_rate_mean', 'RFTN_Ce_mean',
                    'opdur', 'age', 'sex_num', 'bmi', 'asa',
                    'intraop_ppf', 'intraop_eph', 'intraop_phe']
                   + dept_dummy_cols)
    outcome_col = 'HR_rebound'
    needed = [outcome_col] + model_vars
    needed_existing = [c for c in needed if c in df.columns]
    sub = df[needed_existing].dropna()

    print(f"  Complete cases for multivariable model: n = {len(sub)}")

    mv_results = {}
    if len(sub) >= 100:
        X_vars = [c for c in model_vars if c in sub.columns]
        X = sm.add_constant(sub[X_vars])
        y = sub[outcome_col]
        model = sm.OLS(y, X).fit()

        print(f"  Overall R² = {model.rsquared:.4f}, Adj R² = {model.rsquared_adj:.4f}")
        print(f"  F-statistic = {model.fvalue:.2f}, P = {model.f_pvalue:.2e}")

        # Focus on RFTN_taper_slope
        if 'RFTN_taper_slope' in model.params.index:
            beta = model.params['RFTN_taper_slope']
            se = model.bse['RFTN_taper_slope']
            p_val = model.pvalues['RFTN_taper_slope']
            ci = model.conf_int().loc['RFTN_taper_slope']
            print(f"\n  RFTN_taper_slope:")
            print(f"    beta = {beta:+.4f}")
            print(f"    SE   = {se:.4f}")
            print(f"    95% CI = [{ci[0]:+.4f}, {ci[1]:+.4f}]")
            print(f"    P    = {p_val:.4e}")

            mv_results['taper_slope'] = {
                'beta': float(beta),
                'se': float(se),
                'ci_low': float(ci[0]),
                'ci_high': float(ci[1]),
                'p_value': float(p_val),
            }

        # All coefficients summary
        all_coefs = {}
        for var_name in X_vars:
            if var_name in model.params.index:
                ci_row = model.conf_int().loc[var_name]
                all_coefs[var_name] = {
                    'beta': float(model.params[var_name]),
                    'se': float(model.bse[var_name]),
                    'ci_low': float(ci_row[0]),
                    'ci_high': float(ci_row[1]),
                    'p_value': float(model.pvalues[var_name]),
                }
        mv_results['all_coefficients'] = all_coefs
        mv_results['model_summary'] = {
            'n': int(len(sub)),
            'r2': float(model.rsquared),
            'r2_adj': float(model.rsquared_adj),
            'f_stat': float(model.fvalue),
            'f_pvalue': float(model.f_pvalue),
            'aic': float(model.aic),
            'bic': float(model.bic),
        }
    else:
        mv_results['error'] = f'Insufficient data (n={len(sub)})'
        print(f"  [SKIP] Insufficient data")

    results['multivariable_taper_model'] = mv_results

    save_json(results, 'taper_expansion.json')
    return results


# ============================================================
# ANALYSIS 4: Expanded Covariates (Major 5)
# ============================================================
def analysis_4_expanded_covariates(df):
    """
    Re-run main RCS and IPTW with expanded covariate set:
    Original: age, sex, bmi, asa, opdur, TWA_BIS, intraop_ppf
    Expanded: + intraop_eph, intraop_phe, intraop_ebl
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Expanded Covariates (Major 5)")
    print("=" * 70)

    results = {}

    exposure = 'RFTN_total_mcg_kg'
    outcome = 'HR_rebound'

    original_covs = ['age', 'sex_num', 'bmi', 'asa', 'opdur', 'TWA_BIS', 'intraop_ppf']
    expanded_covs = original_covs + ['intraop_eph', 'intraop_phe', 'intraop_ebl']

    # ---- 4a: RCS comparison ----
    print("\n  --- 4a: RCS with original vs expanded covariates ---")

    for cov_label, cov_list in [('original', original_covs), ('expanded', expanded_covs)]:
        needed = [exposure, outcome] + cov_list
        needed_existing = [c for c in needed if c in df.columns]
        sub = df[needed_existing].dropna()

        cov_existing = [c for c in cov_list if c in sub.columns]
        print(f"\n  [{cov_label}] covariates: {cov_existing}, n = {len(sub)}")

        if len(sub) < 100:
            results[f'rcs_{cov_label}'] = {'error': f'Insufficient data (n={len(sub)})'}
            continue

        cov_df = sub[cov_existing].copy()
        rcs_result = fit_rcs(
            sub[exposure].values, sub[outcome].values,
            covariate_df=cov_df,
            n_knots=4, knot_percentiles=[5, 35, 65, 95]
        )
        if rcs_result:
            print(f"    R² = {rcs_result['r2']:.4f}")
            print(f"    R²_linear = {rcs_result['r2_linear']:.4f}")
            print(f"    P_nonlinear = {rcs_result['p_nonlinear']:.4e}")
            results[f'rcs_{cov_label}'] = rcs_result
        else:
            results[f'rcs_{cov_label}'] = {'error': 'RCS fitting failed'}

    # Comparison
    if ('rcs_original' in results and 'rcs_expanded' in results and
            isinstance(results['rcs_original'], dict) and 'r2' in results['rcs_original'] and
            isinstance(results['rcs_expanded'], dict) and 'r2' in results['rcs_expanded']):
        r2_orig = results['rcs_original']['r2']
        r2_exp = results['rcs_expanded']['r2']
        pnl_orig = results['rcs_original']['p_nonlinear']
        pnl_exp = results['rcs_expanded']['p_nonlinear']
        print(f"\n  Comparison:")
        print(f"    R² change: {r2_orig:.4f} -> {r2_exp:.4f} (delta = {r2_exp - r2_orig:+.4f})")
        print(f"    P_nonlinear: {pnl_orig:.4e} -> {pnl_exp:.4e}")
        results['rcs_comparison'] = {
            'r2_original': float(r2_orig),
            'r2_expanded': float(r2_exp),
            'r2_delta': float(r2_exp - r2_orig),
            'p_nonlinear_original': float(pnl_orig),
            'p_nonlinear_expanded': float(pnl_exp),
            'conclusion': 'Robust' if (pnl_orig < 0.05) == (pnl_exp < 0.05) else 'Sensitive to covariates'
        }

    # ---- 4b: IPTW with expanded confounders ----
    print("\n  --- 4b: IPTW with original vs expanded confounders ---")

    # Binary treatment: RFTN_rate_mean > median
    rate_med = df['RFTN_rate_mean'].median()
    df['treat_high_rate'] = (df['RFTN_rate_mean'] > rate_med).astype(int)
    print(f"  Treatment: RFTN_rate_mean > {rate_med:.4f} μg/kg/min")

    for cov_label, cov_list in [('original', original_covs), ('expanded', expanded_covs)]:
        cov_existing = [c for c in cov_list if c in df.columns]
        print(f"\n  [{cov_label}] IPTW confounders: {cov_existing}")

        iptw_result = run_iptw(df, 'treat_high_rate', outcome, cov_existing, n_boot=1000)

        if 'error' not in iptw_result:
            print(f"    n = {iptw_result['n']}")
            print(f"    PS AUC = {iptw_result['ps_auc']:.3f}")
            print(f"    Max |SMD| after = {iptw_result['max_smd_after']:.3f}")
            print(f"    IPTW diff = {iptw_result['iptw_diff']:+.3f} "
                  f"[{iptw_result['ci_low']:+.3f}, {iptw_result['ci_high']:+.3f}]")
            sig_str = "Significant" if iptw_result['significant'] else "Not significant"
            print(f"    {sig_str}")
        else:
            print(f"    {iptw_result['error']}")

        results[f'iptw_{cov_label}'] = iptw_result

    # IPTW comparison
    if ('iptw_original' in results and 'iptw_expanded' in results and
            'iptw_diff' in results.get('iptw_original', {}) and
            'iptw_diff' in results.get('iptw_expanded', {})):
        diff_orig = results['iptw_original']['iptw_diff']
        diff_exp = results['iptw_expanded']['iptw_diff']
        print(f"\n  IPTW Comparison:")
        print(f"    Original IPTW diff: {diff_orig:+.3f}")
        print(f"    Expanded IPTW diff: {diff_exp:+.3f}")
        print(f"    Attenuation: {abs(diff_exp) - abs(diff_orig):+.3f}")
        results['iptw_comparison'] = {
            'original_diff': float(diff_orig),
            'expanded_diff': float(diff_exp),
            'change_pct': float((diff_exp - diff_orig) / abs(diff_orig) * 100)
            if abs(diff_orig) > 1e-10 else None,
        }

    save_json(results, 'expanded_covariates.json')
    return results


# ============================================================
# ANALYSIS 5: Binary Clinical Endpoints (Major 9)
# ============================================================
def analysis_5_binary_endpoints(df):
    """
    Create binary clinical events and run logistic regression:
    - HR_increase_20pct: HR_rebound_pct > 20
    - MAP_increase_20pct: MAP_rebound_pct > 20
    - Any_vasopressor: eph > 0 or phe > 0 or epi > 0
    - Composite_event: any of the above three
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Binary Clinical Endpoints (Major 9)")
    print("=" * 70)

    results = {}

    # Create binary endpoints
    df['HR_increase_20pct'] = (df['HR_rebound_pct'] > 20).astype(float)
    df.loc[df['HR_rebound_pct'].isna(), 'HR_increase_20pct'] = np.nan

    df['MAP_increase_20pct'] = (df['MAP_rebound_pct'] > 20).astype(float)
    df.loc[df['MAP_rebound_pct'].isna(), 'MAP_increase_20pct'] = np.nan

    # Any vasopressor
    eph = df['intraop_eph'].fillna(0)
    phe = df['intraop_phe'].fillna(0)
    epi = df['intraop_epi'].fillna(0)
    df['Any_vasopressor'] = ((eph > 0) | (phe > 0) | (epi > 0)).astype(float)

    # Composite: any of the three
    # Only compute where at least one of HR/MAP rebound_pct is available
    df['Composite_event'] = np.nan
    valid_mask = (df['HR_increase_20pct'].notna()) | (df['MAP_increase_20pct'].notna())
    df.loc[valid_mask, 'Composite_event'] = 0.0
    for evt_col in ['HR_increase_20pct', 'MAP_increase_20pct', 'Any_vasopressor']:
        mask = valid_mask & (df[evt_col] == 1.0)
        df.loc[mask, 'Composite_event'] = 1.0

    endpoints = [
        ('HR_increase_20pct', 'HR increase >20%'),
        ('MAP_increase_20pct', 'MAP increase >20%'),
        ('Any_vasopressor', 'Any vasopressor use'),
        ('Composite_event', 'Composite event'),
    ]

    # ---- Overall and quartile event rates ----
    print("\n  Event rates overall and by RFTN dose quartile:")
    print(f"  {'Endpoint':<25s} {'Overall':>10s} {'Q1':>10s} {'Q2':>10s} {'Q3':>10s} {'Q4':>10s}")
    print("  " + "-" * 68)

    q_labels = ['Q1', 'Q2', 'Q3', 'Q4']

    for ep_col, ep_label in endpoints:
        overall = df[ep_col].dropna()
        n_overall = len(overall)
        rate_overall = overall.mean() * 100 if n_overall > 0 else np.nan

        q_rates = []
        for q in q_labels:
            sub = df.loc[df['RFTN_quartile'] == q, ep_col].dropna()
            rate_q = sub.mean() * 100 if len(sub) > 0 else np.nan
            q_rates.append(rate_q)

        overall_str = f"{rate_overall:.1f}%" if not np.isnan(rate_overall) else "N/A"
        q_strs = [f"{r:.1f}%" if not np.isnan(r) else "N/A" for r in q_rates]
        print(f"  {ep_label:<25s} {overall_str:>10s} "
              f"{q_strs[0]:>10s} {q_strs[1]:>10s} {q_strs[2]:>10s} {q_strs[3]:>10s}")

        results[ep_col] = {
            'overall_rate_pct': float(rate_overall) if not np.isnan(rate_overall) else None,
            'overall_n': int(n_overall),
            'overall_events': int(overall.sum()) if n_overall > 0 else 0,
            'quartile_rates_pct': {q: float(r) if not np.isnan(r) else None
                                    for q, r in zip(q_labels, q_rates)},
        }

    # ---- Logistic regression ----
    print("\n  Logistic regression results:")

    # Covariates for logistic models
    log_covs = ['age', 'sex_num', 'bmi', 'asa', 'opdur', 'intraop_ppf']

    # Two exposures to test
    exposures = [
        ('RFTN_total_mcg_kg', 'per 10 mcg/kg', 10.0),
        ('RFTN_taper_slope', 'per unit', 1.0),
    ]

    for ep_col, ep_label in endpoints:
        print(f"\n  --- {ep_label} ({ep_col}) ---")
        results[ep_col]['logistic'] = {}

        for exp_col, exp_label, scale_factor in exposures:
            needed = [ep_col, exp_col] + log_covs
            needed_existing = [c for c in needed if c in df.columns]
            sub = df[needed_existing].dropna()

            # Drop if outcome has no variation
            if sub[ep_col].nunique() < 2:
                print(f"    [{exp_col}] No events or all events, skipping.")
                results[ep_col]['logistic'][exp_col] = {'error': 'No variation in outcome'}
                continue

            if len(sub) < 50:
                print(f"    [{exp_col}] Insufficient data (n={len(sub)}), skipping.")
                results[ep_col]['logistic'][exp_col] = {'error': f'Insufficient data (n={len(sub)})'}
                continue

            # Scale exposure
            sub[f'{exp_col}_scaled'] = sub[exp_col] / scale_factor

            X_vars = [f'{exp_col}_scaled'] + [c for c in log_covs if c in sub.columns]
            X = sm.add_constant(sub[X_vars])
            y = sub[ep_col]

            try:
                logit_model = sm.Logit(y, X).fit(disp=0, maxiter=200)

                exp_var = f'{exp_col}_scaled'
                if exp_var in logit_model.params.index:
                    beta = logit_model.params[exp_var]
                    se = logit_model.bse[exp_var]
                    p_val = logit_model.pvalues[exp_var]
                    ci = logit_model.conf_int().loc[exp_var]
                    or_val = np.exp(beta)
                    or_ci_low = np.exp(ci[0])
                    or_ci_high = np.exp(ci[1])

                    sig_str = "*" if p_val < 0.05 else ""
                    print(f"    {exp_col} ({exp_label}): OR={or_val:.3f} "
                          f"[{or_ci_low:.3f}, {or_ci_high:.3f}], P={p_val:.4f} "
                          f"(n={len(sub)}) {sig_str}")

                    results[ep_col]['logistic'][exp_col] = {
                        'exposure_label': exp_label,
                        'scale_factor': float(scale_factor),
                        'n': int(len(sub)),
                        'n_events': int(y.sum()),
                        'beta': float(beta),
                        'se': float(se),
                        'or': float(or_val),
                        'or_ci_low': float(or_ci_low),
                        'or_ci_high': float(or_ci_high),
                        'p_value': float(p_val),
                        'pseudo_r2': float(logit_model.prsquared),
                    }
                else:
                    results[ep_col]['logistic'][exp_col] = {'error': 'Exposure not in model'}
            except Exception as e:
                print(f"    [{exp_col}] Logistic regression failed: {e}")
                results[ep_col]['logistic'][exp_col] = {'error': str(e)}

    save_json(results, 'binary_endpoints.json')
    return results


# ============================================================
# Summary printer
# ============================================================
def print_summary(r1, r2, r3, r4, r5):
    """Print a compact summary table of all 5 analyses."""
    print("\n" + "=" * 70)
    print("  SUMMARY OF ALL REVIEWER RESPONSE ANALYSES")
    print("=" * 70)

    print(f"\n  {'Analysis':<45s} {'Key Finding':>25s}")
    print("  " + "-" * 72)

    # Analysis 1
    meta = r1.get('_meta', {})
    print(f"  {'1. Missingness Report':<45s} "
          f"{'n_valid=' + str(meta.get('n_valid_rftn', '?')):>25s}")
    if 'baseline_comparison' in r1:
        max_smd = max([abs(v.get('smd', 0)) for v in r1['baseline_comparison'].values()], default=0)
        balanced = "Yes" if max_smd < 0.1 else f"No (max|SMD|={max_smd:.3f})"
        print(f"  {'   Included vs Excluded balanced?':<45s} {balanced:>25s}")

    # Analysis 2
    for key_name in ['OWSI_partial', 'OWSI_complete']:
        if key_name in r2 and isinstance(r2[key_name], dict) and 'r2' in r2[key_name]:
            r2_val = r2[key_name]['r2']
            pnl = r2[key_name]['p_nonlinear']
            n_val = r2[key_name]['n']
            summary_str = "R2={:.4f} P_nl={:.3e}".format(r2_val, pnl)
            label = "2. " + key_name
            print(f"  {label:<45s} {summary_str:>25s}")

    # Analysis 3
    if 'multivariable_taper_model' in r3 and 'taper_slope' in r3['multivariable_taper_model']:
        ts = r3['multivariable_taper_model']['taper_slope']
        beta_val = ts['beta']
        p_val = ts['p_value']
        summary_str = "b={:+.4f} P={:.3e}".format(beta_val, p_val)
        print(f"  {'3. Taper slope (multivariable)':<45s} {summary_str:>25s}")

    # Analysis 4
    if 'rcs_comparison' in r4:
        comp = r4['rcs_comparison']
        delta_str = "dR2={:+.4f}".format(comp['r2_delta'])
        print(f"  {'4. RCS R2 change (expanded covariates)':<45s} {delta_str:>25s}")
    if 'iptw_comparison' in r4:
        ic = r4['iptw_comparison']
        iptw_str = "{:+.3f} -> {:+.3f}".format(ic['original_diff'], ic['expanded_diff'])
        print(f"  {'   IPTW diff change':<45s} {iptw_str:>25s}")

    # Analysis 5
    for ep in ['HR_increase_20pct', 'MAP_increase_20pct', 'Composite_event']:
        if ep in r5 and 'logistic' in r5[ep] and 'RFTN_total_mcg_kg' in r5[ep]['logistic']:
            lr = r5[ep]['logistic']['RFTN_total_mcg_kg']
            if 'or' in lr:
                or_str = "OR={:.3f} P={:.3e}".format(lr['or'], lr['p_value'])
                label = "5. " + ep
                print(f"  {label:<45s} {or_str:>25s}")

    print("\n  All results saved to: " + str(RESULTS_DIR))
    print("=" * 70)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  OIH Study - Phase 4: Reviewer Response Analyses")
    print(f"  Date: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    df = load_clean_data()

    r1 = analysis_1_missingness(df)
    r2 = analysis_2_owsi_sensitivity(df)
    r3 = analysis_3_taper_expansion(df)
    r4 = analysis_4_expanded_covariates(df)
    r5 = analysis_5_binary_endpoints(df)

    print_summary(r1, r2, r3, r4, r5)

    print(f"\n  Phase 4 (Reviewer Response) Complete!")
    print(f"  Output files:")
    print(f"    - {RESULTS_DIR / 'missingness_report.json'}")
    print(f"    - {RESULTS_DIR / 'owsi_complete_case.json'}")
    print(f"    - {RESULTS_DIR / 'taper_expansion.json'}")
    print(f"    - {RESULTS_DIR / 'expanded_covariates.json'}")
    print(f"    - {RESULTS_DIR / 'binary_endpoints.json'}")


if __name__ == '__main__':
    main()
