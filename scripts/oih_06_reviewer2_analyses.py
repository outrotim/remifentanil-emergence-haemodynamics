#!/usr/bin/env python3
"""
=================================================================
OIH Study - Phase 5: Reviewer Round 2 Supplementary Analyses
=================================================================
Addresses Major 1-4 from second round peer review:

Major 1: Informative Missingness — IPOW (Inverse Probability of
         Observation Weighting) to correct selection-on-observability
         bias for HR/MAP rebound endpoints.

Major 2: OWSI Composite Fix — Complete-case OWSI as primary;
         partial OWSI relegated to supplement.

Major 3: Taper De-collinearity — Partial correlations, residualized
         taper, elastic net to show taper's independent contribution
         beyond rate/Ce collinearity.

Major 4: Analysis Grid — Comprehensive table of all analyses with
         n, missing, covariates, exposure, endpoint, time window.

Also includes:
- IPTW weight distribution / trimming / ESS reporting (Minor 4)
- E_max ED50 bootstrap CI (Minor 5)
- ARD/NNT for binary endpoints (Minor 3)
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
# Helpers
# ============================================================
def _np_serializer(obj):
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
    filepath = RESULTS_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=_np_serializer)
    print(f"  -> Saved to {filepath}")


def load_clean_data():
    """Load and clean master dataset with standard outlier removal."""
    print("=" * 70)
    print("  Loading master dataset...")
    print("=" * 70)

    df = pd.read_csv(OIH_DIR / "oih_master_dataset.csv")
    print(f"  Loaded: {len(df)} cases x {len(df.columns)} columns")

    # Outlier cleanup
    outlier_mask = (
        (df['RFTN_total_mcg_kg'] > 200) |
        (df['RFTN_Ce_peak'] > 100)
    )
    rftn_cols = [c for c in df.columns if c.startswith('RFTN_') or c in ['rftn_conc']]
    df.loc[outlier_mask, rftn_cols] = np.nan
    print(f"  Outlier cleanup: {outlier_mask.sum()} cases -> RFTN NaN")

    # Sex encoding
    df['sex_num'] = (df['sex'] == 'F').astype(int)

    # RFTN dose quartile
    df['RFTN_quartile'] = pd.qcut(
        df['RFTN_total_mcg_kg'].dropna(), q=4, labels=['Q1', 'Q2', 'Q3', 'Q4']
    ).reindex(df.index)

    # Z-scores for OWSI
    oih_components = ['HR_rebound', 'MAP_rebound', 'FTN_rescue_mcg_kg']
    for comp in oih_components:
        if comp in df.columns:
            m, s = df[comp].mean(), df[comp].std()
            df[f'{comp}_Z'] = (df[comp] - m) / s if s > 0 else 0.0

    z_cols = [f'{c}_Z' for c in oih_components if c in df.columns]
    # OWSI_partial (existing): mean of available Z-scores
    df['OWSI_partial'] = df[z_cols].mean(axis=1)
    # OWSI_complete: only when ALL 3 components are non-missing
    all_avail = df[['HR_rebound', 'MAP_rebound', 'FTN_rescue_mcg_kg']].notna().all(axis=1)
    df['OWSI_complete'] = np.where(all_avail, df[z_cols].mean(axis=1), np.nan)
    df['has_complete_owsi'] = all_avail.astype(int)

    # Age/elderly
    df['elderly'] = (df['age'] >= 65).astype(int)

    # Observation indicator for HR/MAP rebound
    df['has_HR_rebound'] = df['HR_rebound'].notna().astype(int)
    df['has_MAP_rebound'] = df['MAP_rebound'].notna().astype(int)

    return df


# ============================================================
# MAJOR 1: IPOW — Inverse Probability of Observation Weighting
# ============================================================
def major1_ipow(df):
    """
    Address informative missingness for HR/MAP rebound:
    1. Model P(observed) using logistic regression
    2. Compute observation weights
    3. Re-run RCS and IPTW with observation weights
    4. Compare: crude, complete-case, IPOW results
    """
    print("\n" + "=" * 70)
    print("MAJOR 1: Inverse Probability of Observation Weighting (IPOW)")
    print("=" * 70)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    results = {}

    # --- 1a: Model P(HR_rebound observed) ---
    obs_predictors = ['opdur', 'age', 'sex_num', 'bmi', 'asa',
                      'RFTN_total_mcg_kg', 'RFTN_rate_mean']
    obs_predictors = [c for c in obs_predictors if c in df.columns]

    sub = df[obs_predictors + ['has_HR_rebound']].dropna()
    X_obs = sub[obs_predictors].values
    y_obs = sub['has_HR_rebound'].values

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_obs)

    obs_model = LogisticRegression(max_iter=5000, C=1.0)
    obs_model.fit(X_sc, y_obs)
    p_obs = obs_model.predict_proba(X_sc)[:, 1]
    auc_obs = roc_auc_score(y_obs, p_obs)

    # Feature importance
    coefs = dict(zip(obs_predictors, obs_model.coef_[0].tolist()))

    print(f"  Observation model AUC: {auc_obs:.3f}")
    print(f"  Observation rate: {y_obs.mean():.3f}")
    print(f"  Feature coefficients (logistic):")
    for k, v in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {k:<25s}: {v:+.4f}")

    results['observation_model'] = {
        'n': int(len(sub)),
        'auc': float(auc_obs),
        'obs_rate': float(y_obs.mean()),
        'feature_coefficients': coefs,
        'predictors': obs_predictors,
    }

    # --- 1b: Compute IPOW weights ---
    # Only for observed (HR_rebound available) cases
    # Weight = 1 / P(observed | X)
    p_obs_clipped = np.clip(p_obs, 0.1, 0.99)

    # Get IPOW weights for observed cases
    obs_idx = sub.index[y_obs == 1]
    p_obs_for_observed = p_obs[y_obs == 1]
    p_obs_for_observed_clipped = np.clip(p_obs_for_observed, 0.1, 0.99)
    ipow_weights = 1.0 / p_obs_for_observed_clipped

    # Stabilize: multiply by marginal P(observed)
    marginal_obs = y_obs.mean()
    ipow_weights_stable = ipow_weights * marginal_obs

    # Weight diagnostics
    results['ipow_diagnostics'] = {
        'n_observed': int(y_obs.sum()),
        'n_unobserved': int((1 - y_obs).sum()),
        'weight_mean': float(ipow_weights_stable.mean()),
        'weight_median': float(np.median(ipow_weights_stable)),
        'weight_min': float(ipow_weights_stable.min()),
        'weight_max': float(ipow_weights_stable.max()),
        'weight_p5': float(np.percentile(ipow_weights_stable, 5)),
        'weight_p95': float(np.percentile(ipow_weights_stable, 95)),
        'ESS': float((ipow_weights_stable.sum())**2 / (ipow_weights_stable**2).sum()),
        'trimming_threshold': '0.10 - 0.99',
    }

    print(f"\n  IPOW weight diagnostics:")
    print(f"    N observed: {y_obs.sum()}, N unobserved: {(1-y_obs).sum()}")
    print(f"    Weight range: [{ipow_weights_stable.min():.3f}, {ipow_weights_stable.max():.3f}]")
    print(f"    ESS: {results['ipow_diagnostics']['ESS']:.1f}")

    # --- 1c: IPOW-weighted regression: RFTN -> HR_rebound ---
    # Merge weights back to main df
    df_obs = df.loc[obs_idx].copy()
    df_obs['ipow_w'] = ipow_weights_stable

    # Need: RFTN_total_mcg_kg, HR_rebound, covariates
    covs = ['age', 'sex_num', 'bmi', 'asa', 'opdur', 'intraop_ppf']
    covs = [c for c in covs if c in df_obs.columns]
    needed = ['RFTN_total_mcg_kg', 'HR_rebound', 'ipow_w'] + covs
    df_reg = df_obs[needed].dropna()

    if len(df_reg) > 100:
        # Unweighted (complete-case)
        X_cc = sm.add_constant(df_reg[['RFTN_total_mcg_kg'] + covs])
        model_cc = sm.OLS(df_reg['HR_rebound'], X_cc).fit()
        beta_cc = model_cc.params['RFTN_total_mcg_kg']
        p_cc = model_cc.pvalues['RFTN_total_mcg_kg']

        # IPOW-weighted
        model_ipow = sm.WLS(df_reg['HR_rebound'], X_cc,
                            weights=df_reg['ipow_w']).fit()
        beta_ipow = model_ipow.params['RFTN_total_mcg_kg']
        p_ipow = model_ipow.pvalues['RFTN_total_mcg_kg']

        # Percent change
        pct_change = 100 * (beta_ipow - beta_cc) / abs(beta_cc) if abs(beta_cc) > 1e-12 else 0

        results['ipow_regression_HR'] = {
            'n': int(len(df_reg)),
            'complete_case_beta': float(beta_cc),
            'complete_case_p': float(p_cc),
            'ipow_beta': float(beta_ipow),
            'ipow_p': float(p_ipow),
            'pct_change': float(pct_change),
            'conclusion': 'consistent' if np.sign(beta_cc) == np.sign(beta_ipow) else 'direction_change',
        }

        print(f"\n  IPOW regression: RFTN -> HR_rebound (n={len(df_reg)})")
        print(f"    Complete-case: beta={beta_cc:.6f}, P={p_cc:.2e}")
        print(f"    IPOW-weighted: beta={beta_ipow:.6f}, P={p_ipow:.2e}")
        print(f"    Change: {pct_change:+.1f}%")

    # --- 1d: Same for MAP rebound ---
    df_obs_map = df.loc[df['MAP_rebound'].notna()].copy()
    # Re-compute obs weights for MAP
    sub_map = df[obs_predictors + ['has_MAP_rebound']].dropna()
    X_map = scaler.fit_transform(sub_map[obs_predictors].values)
    y_map = sub_map['has_MAP_rebound'].values
    obs_model_map = LogisticRegression(max_iter=5000, C=1.0)
    obs_model_map.fit(X_map, y_map)
    p_obs_map = obs_model_map.predict_proba(X_map)[:, 1]

    map_obs_idx = sub_map.index[y_map == 1]
    p_map_obs = np.clip(p_obs_map[y_map == 1], 0.1, 0.99)
    ipow_w_map = y_map.mean() / p_map_obs

    df_obs_map2 = df.loc[map_obs_idx].copy()
    df_obs_map2['ipow_w'] = ipow_w_map

    needed_map = ['RFTN_total_mcg_kg', 'MAP_rebound', 'ipow_w'] + covs
    df_reg_map = df_obs_map2[needed_map].dropna()

    if len(df_reg_map) > 100:
        X_cc_m = sm.add_constant(df_reg_map[['RFTN_total_mcg_kg'] + covs])
        model_cc_m = sm.OLS(df_reg_map['MAP_rebound'], X_cc_m).fit()
        beta_cc_m = model_cc_m.params['RFTN_total_mcg_kg']
        p_cc_m = model_cc_m.pvalues['RFTN_total_mcg_kg']

        model_ipow_m = sm.WLS(df_reg_map['MAP_rebound'], X_cc_m,
                               weights=df_reg_map['ipow_w']).fit()
        beta_ipow_m = model_ipow_m.params['RFTN_total_mcg_kg']
        p_ipow_m = model_ipow_m.pvalues['RFTN_total_mcg_kg']

        pct_m = 100 * (beta_ipow_m - beta_cc_m) / abs(beta_cc_m) if abs(beta_cc_m) > 1e-12 else 0

        results['ipow_regression_MAP'] = {
            'n': int(len(df_reg_map)),
            'complete_case_beta': float(beta_cc_m),
            'complete_case_p': float(p_cc_m),
            'ipow_beta': float(beta_ipow_m),
            'ipow_p': float(p_ipow_m),
            'pct_change': float(pct_m),
        }
        print(f"\n  IPOW regression: RFTN -> MAP_rebound (n={len(df_reg_map)})")
        print(f"    Complete-case: beta={beta_cc_m:.6f}, P={p_cc_m:.2e}")
        print(f"    IPOW-weighted: beta={beta_ipow_m:.6f}, P={p_ipow_m:.2e}")
        print(f"    Change: {pct_m:+.1f}%")

    # --- 1e: Explicit estimand statement ---
    results['estimand'] = {
        'primary': 'Average treatment effect among the observable subset '
                   '(patients with >=15min post-surgery monitoring data)',
        'assumption': 'Missing at random conditional on opdur, age, sex, bmi, asa, '
                      'RFTN_total_mcg_kg, RFTN_rate_mean',
        'ipow_purpose': 'Reweight observed cases to represent full cohort under MAR',
    }

    return results


# ============================================================
# MAJOR 2: OWSI Complete-Case as Primary
# ============================================================
def major2_owsi_fix(df):
    """
    Fix OWSI measurement model:
    - OWSI_complete (all 3 components) as primary
    - OWSI_partial (available mean) as sensitivity/supplement
    - Report n and composition for each
    """
    print("\n" + "=" * 70)
    print("MAJOR 2: OWSI Composite Fix -- Complete vs Partial")
    print("=" * 70)

    from patsy import dmatrix

    results = {}

    # Composition analysis
    hr_avail = df['HR_rebound'].notna()
    map_avail = df['MAP_rebound'].notna()
    ftn_avail = df['FTN_rescue_mcg_kg'].notna()

    composition = {
        'all_3': int((hr_avail & map_avail & ftn_avail).sum()),
        'hr_map_only': int((hr_avail & map_avail & ~ftn_avail).sum()),
        'ftn_only': int((~hr_avail & ~map_avail & ftn_avail).sum()),
        'hr_ftn_only': int((hr_avail & ~map_avail & ftn_avail).sum()),
        'map_ftn_only': int((~hr_avail & map_avail & ftn_avail).sum()),
        'none': int((~hr_avail & ~map_avail & ~ftn_avail).sum()),
        'any_1': int((hr_avail | map_avail | ftn_avail).sum()),
    }
    results['composition'] = composition
    print(f"  Component availability:")
    for k, v in composition.items():
        print(f"    {k}: {v}")

    # --- OWSI_complete RCS ---
    df_comp = df[df['OWSI_complete'].notna() & df['RFTN_total_mcg_kg'].notna()].copy()
    print(f"\n  OWSI_complete available: {len(df_comp)}")

    covs = ['age', 'sex_num', 'bmi', 'asa', 'opdur', 'intraop_ppf']
    covs = [c for c in covs if c in df_comp.columns]

    if len(df_comp) > 200:
        x = df_comp['RFTN_total_mcg_kg'].values
        y_comp = df_comp['OWSI_complete'].values
        knots = np.percentile(x, [5, 35, 65, 95])
        knot_str = ', '.join([f'{k:.6f}' for k in knots])

        df_fit = pd.DataFrame({'x': x, 'y': y_comp})
        for c in covs:
            df_fit[c] = df_comp[c].values

        cov_terms = ' + '.join(covs)
        lin_formula = f'y ~ x + {cov_terms}'
        rcs_formula = f'y ~ cr(x, knots=[{knot_str}]) + {cov_terms}'

        lin_m = sm.OLS.from_formula(lin_formula, df_fit).fit()
        rcs_m = sm.OLS.from_formula(rcs_formula, df_fit).fit()

        df_diff = rcs_m.df_model - lin_m.df_model
        if df_diff > 0:
            lr_stat = -2 * (lin_m.llf - rcs_m.llf)
            p_nl = 1 - stats.chi2.cdf(lr_stat, df_diff)
        else:
            p_nl = 1.0

        results['owsi_complete_rcs'] = {
            'n': int(len(df_comp)),
            'r2': float(rcs_m.rsquared),
            'r2_linear': float(lin_m.rsquared),
            'p_nonlinear': float(p_nl),
            'beta_linear': float(lin_m.params['x']),
            'p_linear': float(lin_m.pvalues['x']),
        }
        print(f"  OWSI_complete RCS: n={len(df_comp)}, R2={rcs_m.rsquared:.3f}, "
              f"P_nonlinear={p_nl:.4f}")

    # --- OWSI_partial RCS for comparison ---
    df_part = df[df['OWSI_partial'].notna() & df['RFTN_total_mcg_kg'].notna()].copy()
    if len(df_part) > 200:
        x2 = df_part['RFTN_total_mcg_kg'].values
        y_part = df_part['OWSI_partial'].values
        knots2 = np.percentile(x2, [5, 35, 65, 95])
        knot_str2 = ', '.join([f'{k:.6f}' for k in knots2])

        df_fit2 = pd.DataFrame({'x': x2, 'y': y_part})
        for c in covs:
            if c in df_part.columns:
                df_fit2[c] = df_part[c].values

        cov_terms2 = ' + '.join([c for c in covs if c in df_fit2.columns])
        lin_m2 = sm.OLS.from_formula(f'y ~ x + {cov_terms2}', df_fit2).fit()
        rcs_m2 = sm.OLS.from_formula(f'y ~ cr(x, knots=[{knot_str2}]) + {cov_terms2}', df_fit2).fit()

        df_diff2 = rcs_m2.df_model - lin_m2.df_model
        p_nl2 = 1 - stats.chi2.cdf(-2 * (lin_m2.llf - rcs_m2.llf), df_diff2) if df_diff2 > 0 else 1.0

        results['owsi_partial_rcs'] = {
            'n': int(len(df_part)),
            'r2': float(rcs_m2.rsquared),
            'p_nonlinear': float(p_nl2),
        }
        print(f"  OWSI_partial RCS: n={len(df_part)}, R2={rcs_m2.rsquared:.3f}, "
              f"P_nonlinear={p_nl2:.4f}")

    return results


# ============================================================
# MAJOR 3: Taper De-collinearity Analyses
# ============================================================
def major3_taper_decollinearity(df):
    """
    Show taper_slope's independent contribution beyond rate/Ce:
    (a) Partial correlations controlling rate + mean Ce
    (b) Residualized taper: regress taper on rate/Ce, use residual
    (c) Elastic net with all exposures to show stable taper contribution
    """
    print("\n" + "=" * 70)
    print("MAJOR 3: Taper De-collinearity Analyses")
    print("=" * 70)

    results = {}

    # Collinearity diagnostics first
    taper_vars = ['RFTN_taper_slope', 'RFTN_rate_mean', 'RFTN_Ce_mean',
                  'RFTN_total_mcg_kg']
    taper_vars = [v for v in taper_vars if v in df.columns]
    sub = df[taper_vars + ['HR_rebound']].dropna()
    print(f"  Complete cases for taper analysis: {len(sub)}")

    # VIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_vif = sm.add_constant(sub[taper_vars])
    vifs = {}
    for i, col in enumerate(X_vif.columns):
        if col == 'const':
            continue
        vifs[col] = float(variance_inflation_factor(X_vif.values, i))
    results['vif'] = vifs
    print(f"\n  VIF diagnostics:")
    for k, v in vifs.items():
        flag = " WARNING" if v > 5 else ""
        print(f"    {k:<25s}: {v:.2f}{flag}")

    # --- 3a: Partial correlations ---
    print(f"\n  Partial correlations (taper_slope <-> HR_rebound | controls):")
    outcomes = ['HR_rebound', 'MAP_rebound']
    outcomes = [o for o in outcomes if o in df.columns]
    controls_sets = {
        'unadjusted': [],
        'ctrl_rate': ['RFTN_rate_mean'],
        'ctrl_rate_Ce': ['RFTN_rate_mean', 'RFTN_Ce_mean'],
        'ctrl_rate_Ce_dose': ['RFTN_rate_mean', 'RFTN_Ce_mean', 'RFTN_total_mcg_kg'],
    }

    partial_results = {}
    for outcome in outcomes:
        partial_results[outcome] = {}
        needed_base = ['RFTN_taper_slope', outcome]

        for ctrl_name, ctrl_vars in controls_sets.items():
            needed = needed_base + ctrl_vars
            needed = list(set(needed))
            df_sub = df[needed].dropna()

            if len(df_sub) < 50:
                continue

            if len(ctrl_vars) == 0:
                r, p = stats.spearmanr(df_sub['RFTN_taper_slope'], df_sub[outcome])
            else:
                # Partial correlation via residualization
                X_ctrl = sm.add_constant(df_sub[ctrl_vars])
                resid_taper = sm.OLS(df_sub['RFTN_taper_slope'], X_ctrl).fit().resid
                resid_outcome = sm.OLS(df_sub[outcome], X_ctrl).fit().resid
                r, p = stats.spearmanr(resid_taper, resid_outcome)

            partial_results[outcome][ctrl_name] = {
                'n': int(len(df_sub)),
                'rho': float(r),
                'p': float(p),
            }
            print(f"    {outcome} | {ctrl_name}: rho={r:.4f}, P={p:.2e}, n={len(df_sub)}")

    results['partial_correlations'] = partial_results

    # --- 3b: Residualized taper ---
    print(f"\n  Residualized taper analysis:")
    resid_vars = ['RFTN_rate_mean', 'RFTN_Ce_mean']
    resid_vars = [v for v in resid_vars if v in df.columns]

    for outcome in outcomes:
        needed = ['RFTN_taper_slope'] + resid_vars + [outcome]
        df_sub = df[needed].dropna()

        if len(df_sub) < 100:
            continue

        # Residualize taper against rate + Ce
        X_resid = sm.add_constant(df_sub[resid_vars])
        resid_model = sm.OLS(df_sub['RFTN_taper_slope'], X_resid).fit()
        taper_residual = resid_model.resid
        r2_explained = resid_model.rsquared

        # Regression of outcome on residualized taper
        X_resid_out = sm.add_constant(taper_residual)
        out_model = sm.OLS(df_sub[outcome], X_resid_out).fit()
        beta_resid = out_model.params.iloc[1]
        p_resid = out_model.pvalues.iloc[1]

        results[f'residualized_taper_{outcome}'] = {
            'n': int(len(df_sub)),
            'taper_variance_explained_by_controls': float(r2_explained),
            'residual_taper_beta': float(beta_resid),
            'residual_taper_p': float(p_resid),
            'significant': bool(p_resid < 0.05),
        }
        print(f"    {outcome}: R2(taper~controls)={r2_explained:.3f}, "
              f"resid_beta={beta_resid:.4f}, P={p_resid:.2e}")

    # --- 3c: Elastic net ---
    print(f"\n  Elastic net (all exposure metrics -> HR_rebound):")
    try:
        from sklearn.linear_model import ElasticNetCV
        from sklearn.preprocessing import StandardScaler

        en_vars = ['RFTN_total_mcg_kg', 'RFTN_rate_mean', 'RFTN_Ce_mean',
                   'RFTN_Ce_peak', 'RFTN_Ce_SD', 'RFTN_Ce_CV',
                   'RFTN_taper_slope']
        en_vars = [v for v in en_vars if v in df.columns]

        covs = ['age', 'sex_num', 'bmi', 'asa', 'opdur', 'intraop_ppf']
        covs = [c for c in covs if c in df.columns]
        all_feats = en_vars + covs
        needed = all_feats + ['HR_rebound']
        df_en = df[needed].dropna()

        if len(df_en) > 200:
            X_en = df_en[all_feats].values
            y_en = df_en['HR_rebound'].values

            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X_en)

            en_model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95],
                                   cv=5, random_state=42, max_iter=5000)
            en_model.fit(X_sc, y_en)

            # Standardized coefficients
            en_coefs = dict(zip(all_feats, en_model.coef_.tolist()))

            results['elastic_net'] = {
                'n': int(len(df_en)),
                'best_l1_ratio': float(en_model.l1_ratio_),
                'best_alpha': float(en_model.alpha_),
                'r2': float(en_model.score(X_sc, y_en)),
                'standardized_coefficients': en_coefs,
                'taper_rank': sorted(
                    [(k, abs(v)) for k, v in en_coefs.items()],
                    key=lambda x: x[1], reverse=True
                ),
            }

            print(f"    n={len(df_en)}, R2={en_model.score(X_sc, y_en):.4f}")
            print(f"    L1 ratio: {en_model.l1_ratio_}, alpha: {en_model.alpha_:.4f}")
            print(f"    Standardized coefficients (exposure vars):")
            for v in en_vars:
                coef = en_coefs.get(v, 0)
                print(f"      {v:<25s}: {coef:+.5f}")

    except Exception as e:
        print(f"    [ELASTIC NET ERROR] {e}")
        results['elastic_net'] = {'error': str(e)}

    return results


# ============================================================
# MAJOR 4: Analysis Grid
# ============================================================
def major4_analysis_grid(df):
    """
    Create comprehensive analysis-level summary table:
    analysis_id, n, n_missing, covariates, exposure, endpoint, time_window
    """
    print("\n" + "=" * 70)
    print("MAJOR 4: Analysis Grid")
    print("=" * 70)

    grid = []

    # Helper to count
    def n_for(cols):
        return int(df[cols].dropna().shape[0]) if isinstance(cols, list) else int(df[cols].notna().sum())

    # --- Core analyses ---
    covs_base = 'age, sex, bmi, asa, opdur, intraop_ppf'
    covs_expanded = covs_base + ', eph, phe, ebl'

    grid.append({
        'analysis_id': 'RCS-1',
        'description': 'Total dose -> HR rebound (RCS 4 knots)',
        'n': n_for(['RFTN_total_mcg_kg', 'HR_rebound']),
        'exposure': 'RFTN_total_mcg_kg',
        'endpoint': 'HR_rebound',
        'covariates': covs_base,
        'time_window': 'Post 0-15 min (OR monitors)',
    })
    grid.append({
        'analysis_id': 'RCS-2',
        'description': 'Total dose -> MAP rebound',
        'n': n_for(['RFTN_total_mcg_kg', 'MAP_rebound']),
        'exposure': 'RFTN_total_mcg_kg',
        'endpoint': 'MAP_rebound',
        'covariates': covs_base,
        'time_window': 'Post 0-15 min',
    })
    grid.append({
        'analysis_id': 'RCS-3',
        'description': 'Total dose -> FTN rescue',
        'n': n_for(['RFTN_total_mcg_kg', 'FTN_rescue_mcg_kg']),
        'exposure': 'RFTN_total_mcg_kg',
        'endpoint': 'FTN_rescue_mcg_kg',
        'covariates': covs_base,
        'time_window': 'Intraop (bolus sum)',
    })
    grid.append({
        'analysis_id': 'RCS-4',
        'description': 'Total dose -> OWSI_complete',
        'n': n_for(['RFTN_total_mcg_kg', 'OWSI_complete']),
        'exposure': 'RFTN_total_mcg_kg',
        'endpoint': 'OWSI_complete',
        'covariates': covs_base,
        'time_window': 'Mixed (HR/MAP post + FTN intraop)',
    })
    grid.append({
        'analysis_id': 'RCS-5',
        'description': 'Total dose -> NHD index',
        'n': n_for(['RFTN_total_mcg_kg', 'NHD_pct']),
        'exposure': 'RFTN_total_mcg_kg',
        'endpoint': 'NHD_pct',
        'covariates': covs_base,
        'time_window': 'Intraop (% time)',
    })
    grid.append({
        'analysis_id': 'IPTW-1',
        'description': 'IPTW: High vs Low rate -> HR rebound',
        'n': 'see results',
        'exposure': 'RFTN_rate_mean (binary: >median)',
        'endpoint': 'HR_rebound',
        'covariates': covs_base + ' (PS model)',
        'time_window': 'Post 0-15 min',
    })
    grid.append({
        'analysis_id': 'TAPER-1',
        'description': 'Taper slope -> HR rebound (univariate)',
        'n': n_for(['RFTN_taper_slope', 'HR_rebound']),
        'exposure': 'RFTN_taper_slope',
        'endpoint': 'HR_rebound',
        'covariates': 'None',
        'time_window': 'Last 30 min -> Post 15 min',
    })
    grid.append({
        'analysis_id': 'TAPER-2',
        'description': 'Taper slope -> HR rebound (multivariable)',
        'n': n_for(['RFTN_taper_slope', 'HR_rebound', 'RFTN_rate_mean', 'RFTN_Ce_mean']),
        'exposure': 'RFTN_taper_slope',
        'endpoint': 'HR_rebound',
        'covariates': covs_base + ', rate_mean, Ce_mean',
        'time_window': 'Last 30 min -> Post 15 min',
    })
    grid.append({
        'analysis_id': 'IPOW-1',
        'description': 'IPOW-weighted: RFTN -> HR rebound',
        'n': 'see IPOW results',
        'exposure': 'RFTN_total_mcg_kg',
        'endpoint': 'HR_rebound',
        'covariates': covs_base + ' + obs weights',
        'time_window': 'Post 0-15 min',
    })
    grid.append({
        'analysis_id': 'BINARY-1',
        'description': 'Logistic: dose quartile -> HR>20% event',
        'n': n_for(['RFTN_total_mcg_kg', 'HR_rebound']),
        'exposure': 'RFTN_total_mcg_kg (quartile)',
        'endpoint': 'HR_rebound > 20% baseline',
        'covariates': covs_base,
        'time_window': 'Post 0-15 min',
    })
    grid.append({
        'analysis_id': 'MEDIATION-1',
        'description': 'Baron-Kenny: RFTN -> CV-BIS -> HR rebound',
        'n': n_for(['RFTN_total_mcg_kg', 'CV_BIS', 'HR_rebound']),
        'exposure': 'RFTN_total_mcg_kg',
        'endpoint': 'HR_rebound',
        'covariates': 'None (exploratory)',
        'time_window': 'Intraop -> Post',
    })
    grid.append({
        'analysis_id': 'VOLATILE-1',
        'description': 'Volatile agent sensitivity',
        'n': n_for(['RFTN_total_mcg_kg', 'HR_rebound']),
        'exposure': 'RFTN_total_mcg_kg + anes_type',
        'endpoint': 'HR_rebound',
        'covariates': covs_base + ', anes_type',
        'time_window': 'Post 0-15 min',
    })

    results = {'analysis_grid': grid}

    # Print table
    print(f"\n  {'ID':<12s} {'n':>6s} {'Exposure':<30s} {'Endpoint':<20s}")
    print("  " + "-" * 70)
    for row in grid:
        n_str = str(row['n'])[:6]
        print(f"  {row['analysis_id']:<12s} {n_str:>6s} {row['exposure'][:30]:<30s} {row['endpoint'][:20]:<20s}")

    return results


# ============================================================
# MINOR 3: ARD / NNT for binary endpoints
# ============================================================
def minor3_ard_nnt(df):
    """Compute absolute risk difference and NNT for binary endpoints."""
    print("\n" + "=" * 70)
    print("MINOR 3: ARD/NNT for Binary Endpoints")
    print("=" * 70)

    results = {}
    df_q = df[df['RFTN_quartile'].notna()].copy()

    binary_endpoints = {}
    if 'HR_rebound' in df.columns and 'HR_stable_mean' in df.columns:
        df_q['HR_event_20'] = (df_q['HR_rebound'] / df_q['HR_stable_mean'].clip(lower=1) > 0.20).astype(float)
        binary_endpoints['HR_event_20'] = 'HR > 20% increase'
    if 'MAP_rebound' in df.columns and 'MAP_stable_mean' in df.columns:
        df_q['MAP_event_20'] = (df_q['MAP_rebound'] / df_q['MAP_stable_mean'].clip(lower=1) > 0.20).astype(float)
        binary_endpoints['MAP_event_20'] = 'MAP > 20% increase'

    for ep_col, ep_label in binary_endpoints.items():
        sub = df_q[[ep_col, 'RFTN_quartile']].dropna()
        if len(sub) < 100:
            continue

        q1_risk = sub.loc[sub['RFTN_quartile'] == 'Q1', ep_col].mean()
        q4_risk = sub.loc[sub['RFTN_quartile'] == 'Q4', ep_col].mean()
        ard = q4_risk - q1_risk
        nnt = 1.0 / abs(ard) if abs(ard) > 1e-6 else float('inf')

        results[ep_col] = {
            'label': ep_label,
            'Q1_risk': float(q1_risk),
            'Q4_risk': float(q4_risk),
            'ARD': float(ard),
            'NNT': float(nnt) if nnt < 10000 else 'infinity',
            'n': int(len(sub)),
        }
        print(f"  {ep_label}: Q1={q1_risk:.3f}, Q4={q4_risk:.3f}, ARD={ard:+.3f}, NNT={nnt:.1f}")

    return results


# ============================================================
# MINOR 5: E_max ED50 bootstrap CI
# ============================================================
def minor5_emax_bootstrap(df):
    """Bootstrap CI for E_max model ED50 estimate."""
    print("\n" + "=" * 70)
    print("MINOR 5: E_max ED50 Bootstrap CI")
    print("=" * 70)

    from scipy.optimize import curve_fit

    results = {}
    sub = df[['RFTN_total_mcg_kg', 'HR_rebound']].dropna()
    x = sub['RFTN_total_mcg_kg'].values
    y = sub['HR_rebound'].values

    def emax_func(x, E0, Emax, ED50, n):
        return E0 + Emax * x**n / (ED50**n + x**n)

    # Fit on full data
    try:
        p0 = [4.0, -8.0, 25.0, 1.0]
        bounds = ([-20, -30, 1, 0.1], [30, 5, 200, 5])
        popt, pcov = curve_fit(emax_func, x, y, p0=p0, bounds=bounds, maxfev=10000)
        ed50_point = popt[2]
        print(f"  Point estimate: ED50 = {ed50_point:.1f} mcg/kg")
        print(f"  Full fit: E0={popt[0]:.2f}, Emax={popt[1]:.2f}, n={popt[3]:.2f}")

        # Bootstrap
        n_boot = 2000
        rng = np.random.RandomState(42)
        ed50_boots = []
        n_data = len(x)

        for b in range(n_boot):
            idx = rng.choice(n_data, n_data, replace=True)
            xb, yb = x[idx], y[idx]
            try:
                popt_b, _ = curve_fit(emax_func, xb, yb, p0=popt, bounds=bounds, maxfev=5000)
                ed50_boots.append(popt_b[2])
            except Exception:
                pass

        if len(ed50_boots) > 100:
            ci_low = float(np.percentile(ed50_boots, 2.5))
            ci_high = float(np.percentile(ed50_boots, 97.5))
            results['emax_bootstrap'] = {
                'n': int(n_data),
                'n_successful_boots': len(ed50_boots),
                'ED50_point': float(ed50_point),
                'ED50_ci_low': ci_low,
                'ED50_ci_high': ci_high,
                'E0': float(popt[0]),
                'Emax': float(popt[1]),
                'Hill_n': float(popt[3]),
            }
            print(f"  Bootstrap ({len(ed50_boots)}/{n_boot} successful):")
            print(f"  ED50 = {ed50_point:.1f} mcg/kg [95% CI: {ci_low:.1f} - {ci_high:.1f}]")
        else:
            print(f"  Bootstrap failed (only {len(ed50_boots)}/{n_boot} converged)")

    except Exception as e:
        print(f"  [E_MAX ERROR] {e}")
        results['emax_bootstrap'] = {'error': str(e)}

    return results


# ============================================================
# MINOR 4: IPTW reporting enhancement
# ============================================================
def minor4_iptw_reporting(df):
    """Enhanced IPTW reporting: weight distribution, ESS, overlap."""
    print("\n" + "=" * 70)
    print("MINOR 4: Enhanced IPTW Reporting")
    print("=" * 70)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    results = {}

    # Re-run IPTW with full reporting
    covs = ['age', 'sex_num', 'bmi', 'asa', 'opdur', 'intraop_ppf']
    covs = [c for c in covs if c in df.columns]

    # Treatment: above-median RFTN rate
    needed = covs + ['RFTN_rate_mean', 'HR_rebound']
    sub = df[needed].dropna()
    median_rate = sub['RFTN_rate_mean'].median()
    sub['T'] = (sub['RFTN_rate_mean'] > median_rate).astype(int)

    X = sub[covs].values
    T = sub['T'].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    ps_model = LogisticRegression(max_iter=5000, C=1.0)
    ps_model.fit(X_sc, T)
    ps = ps_model.predict_proba(X_sc)[:, 1]

    # Trimming: clip extreme values
    ps_raw = ps.copy()
    ps = np.clip(ps, 0.05, 0.95)
    n_trimmed = int(((ps_raw < 0.05) | (ps_raw > 0.95)).sum())

    # Stabilized weights
    p_treat = T.mean()
    w = np.where(T == 1, p_treat / ps, (1 - p_treat) / (1 - ps))
    ess_treated = (w[T == 1].sum())**2 / (w[T == 1]**2).sum()
    ess_control = (w[T == 0].sum())**2 / (w[T == 0]**2).sum()

    # Weight distribution by decile
    weight_deciles = np.percentile(w, np.arange(0, 101, 10)).tolist()

    results['iptw_enhanced'] = {
        'n': int(len(sub)),
        'n_treated': int(T.sum()),
        'n_control': int((1 - T).sum()),
        'ps_trimming': {'threshold': [0.05, 0.95], 'n_trimmed': n_trimmed},
        'weight_distribution': {
            'min': float(w.min()),
            'p5': float(np.percentile(w, 5)),
            'p25': float(np.percentile(w, 25)),
            'median': float(np.median(w)),
            'p75': float(np.percentile(w, 75)),
            'p95': float(np.percentile(w, 95)),
            'max': float(w.max()),
            'mean': float(w.mean()),
            'sd': float(w.std()),
            'deciles': weight_deciles,
        },
        'effective_sample_size': {
            'ESS_treated': float(ess_treated),
            'ESS_control': float(ess_control),
            'ESS_total': float(ess_treated + ess_control),
            'ESS_pct_of_actual': float((ess_treated + ess_control) / len(sub) * 100),
        },
        'overlap': {
            'ps_treated_mean': float(ps[T == 1].mean()),
            'ps_treated_sd': float(ps[T == 1].std()),
            'ps_control_mean': float(ps[T == 0].mean()),
            'ps_control_sd': float(ps[T == 0].std()),
        },
    }

    print(f"  n={len(sub)}, treated={T.sum()}, control={(1-T).sum()}")
    print(f"  PS trimmed: {n_trimmed} observations at [0.05, 0.95]")
    print(f"  Weight range: [{w.min():.3f}, {w.max():.3f}], median={np.median(w):.3f}")
    print(f"  ESS: treated={ess_treated:.0f}, control={ess_control:.0f}, "
          f"total={ess_treated+ess_control:.0f} ({(ess_treated+ess_control)/len(sub)*100:.1f}%)")

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("OIH Study Phase 5: Reviewer Round 2 Analyses")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    df = load_clean_data()

    all_results = {}

    # Major analyses
    all_results['major1_ipow'] = major1_ipow(df)
    all_results['major2_owsi'] = major2_owsi_fix(df)
    all_results['major3_taper'] = major3_taper_decollinearity(df)
    all_results['major4_grid'] = major4_analysis_grid(df)

    # Minor analyses
    all_results['minor3_ard_nnt'] = minor3_ard_nnt(df)
    all_results['minor4_iptw'] = minor4_iptw_reporting(df)
    all_results['minor5_emax'] = minor5_emax_bootstrap(df)

    # Metadata
    all_results['metadata'] = {
        'script': 'oih_06_reviewer2_analyses.py',
        'timestamp': datetime.now().isoformat(),
        'total_cases': int(len(df)),
        'valid_rftn': int(df['RFTN_total_mcg_kg'].notna().sum()),
    }

    save_json(all_results, 'reviewer2_analyses.json')

    print("\n" + "=" * 70)
    print("  ALL PHASE 5 ANALYSES COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
