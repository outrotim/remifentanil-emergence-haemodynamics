#!/usr/bin/env python3
"""
=================================================================
OIH Study - Phase 7: Supplementary Sensitivity Analyses
=================================================================
Based on external peer review optimization suggestions:
1. Fix S8 sensitivity analysis (opdur >= 90 min restriction)
2. FDR correction for RCS nonlinearity P-values
3. Pump cessation time proxy sensitivity (Ce_at_end adjustment)
4. Taper slope linear fit R² reporting
5. Alternative baseline definition sensitivity
=================================================================
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from datetime import datetime
from scipy import stats

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent.parent
OIH_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(OIH_DIR / "oih_master_dataset.csv")
    outlier_mask = (
        (df['RFTN_total_mcg_kg'] > 200) |
        (df['RFTN_Ce_peak'] > 100)
    )
    rftn_cols = [c for c in df.columns if c.startswith('RFTN_') or c in ['rftn_conc']]
    df.loc[outlier_mask, rftn_cols] = np.nan

    if 'sex' in df.columns:
        df['sex_num'] = (df['sex'] == 'F').astype(int)

    oih_components = ['HR_rebound', 'MAP_rebound', 'FTN_rescue_mcg_kg']
    available = [c for c in oih_components if c in df.columns]
    if available:
        for comp in available:
            col = f'{comp}_Z'
            m, s = df[comp].mean(), df[comp].std()
            df[col] = (df[comp] - m) / s if s > 0 else 0
        z_cols = [f'{c}_Z' for c in available]
        df['OSI'] = df[z_cols].mean(axis=1)

    print(f"Loaded {len(df)} cases, {df['RFTN_total_mcg_kg'].notna().sum()} valid RFTN")
    return df


def rcs_analysis(df, exposure, outcome, covariates=None, n_knots=4):
    """Compute RCS P_nonlinear (LRT) with specified covariates."""
    import statsmodels.api as sm

    if covariates is None:
        covariates = ['age', 'sex_num', 'bmi', 'asa', 'opdur', 'intraop_ppf']
    covariates = [c for c in covariates if c in df.columns]

    cols = [exposure, outcome] + covariates
    df_clean = df[cols].dropna().copy()
    if len(df_clean) < 100:
        return None

    cov_str = " + ".join(covariates)
    linear_formula = f"{outcome} ~ {exposure} + {cov_str}"
    spline_formula = f"{outcome} ~ cr({exposure}, df={n_knots-1}) + {cov_str}"

    linear_model = sm.OLS.from_formula(linear_formula, data=df_clean).fit()
    spline_model = sm.OLS.from_formula(spline_formula, data=df_clean).fit()

    lr_stat = -2 * (linear_model.llf - spline_model.llf)
    df_diff = spline_model.df_model - linear_model.df_model
    p_nonlinear = 1 - stats.chi2.cdf(lr_stat, df_diff)

    return {
        'n': len(df_clean),
        'linear_r2': linear_model.rsquared,
        'spline_r2': spline_model.rsquared,
        'p_nonlinear': p_nonlinear
    }


def analysis_1_fix_s8(df):
    """Fix S8: verify opdur filtering actually reduces sample."""
    print("\n" + "=" * 70)
    print("1. S8 Fix: Long Surgery Sensitivity (opdur >= 90 min)")
    print("=" * 70)

    exposure, outcome = 'RFTN_total_mcg_kg', 'HR_rebound'
    covariates = ['age', 'sex_num', 'bmi', 'asa', 'opdur', 'intraop_ppf']
    covariates = [c for c in covariates if c in df.columns]
    cols = [exposure, outcome] + covariates

    df_all = df[cols].dropna()
    df_long = df[df['opdur'] >= 90][cols].dropna()

    print(f"  Main analytic sample (all): n = {len(df_all)}")
    print(f"  Long surgery (opdur >= 90 min): n = {len(df_long)}")
    print(f"  Cases removed by filter: {len(df_all) - len(df_long)}")

    opdur_stats = df_all['opdur'].describe()
    print(f"  opdur in main sample: min={opdur_stats['min']:.0f}, "
          f"median={opdur_stats['50%']:.0f}, max={opdur_stats['max']:.0f}")
    n_below_90 = (df_all['opdur'] < 90).sum()
    print(f"  Cases with opdur < 90 in analytic sample: {n_below_90}")

    results = {}
    res_main = rcs_analysis(df, exposure, outcome)
    res_long = rcs_analysis(df[df['opdur'] >= 90], exposure, outcome)

    if res_main and res_long:
        results['main'] = res_main
        results['S8_long_surgery'] = res_long
        results['cases_removed'] = res_main['n'] - res_long['n']
        results['diagnosis'] = (
            "IDENTICAL" if res_main['p_nonlinear'] == res_long['p_nonlinear']
            else "DIFFERENT"
        )
        print(f"\n  Main P_nonlinear: {res_main['p_nonlinear']:.6f}")
        print(f"  S8   P_nonlinear: {res_long['p_nonlinear']:.6f}")
        print(f"  Diagnosis: {results['diagnosis']}")

    if n_below_90 == 0:
        print("\n  ROOT CAUSE: All cases in the HR_rebound analytic sample already "
              "have opdur >= 90 min (structural missingness removes short cases).")
        print("  → S8 is NOT a bug — the filter is vacuous because HR_rebound "
              "requires post-surgical data that short surgeries lack.")
        results['root_cause'] = (
            "Structural missingness: HR_rebound requires >=15 min post-surgical data, "
            "which effectively excludes nearly all cases with opdur < 90 min. "
            "The opdur >= 90 filter is therefore vacuous in the analytic sample."
        )

        print("\n  Alternative S8: Restricting to opdur >= 120 min...")
        res_120 = rcs_analysis(df[df['opdur'] >= 120], exposure, outcome)
        if res_120:
            results['S8_alt_120min'] = res_120
            print(f"  S8 (>=120 min) n={res_120['n']}, P={res_120['p_nonlinear']:.6f}")

        print("\n  Alternative S8b: Restricting to opdur >= 150 min...")
        res_150 = rcs_analysis(df[df['opdur'] >= 150], exposure, outcome)
        if res_150:
            results['S8_alt_150min'] = res_150
            print(f"  S8 (>=150 min) n={res_150['n']}, P={res_150['p_nonlinear']:.6f}")

    return results


def analysis_2_fdr_correction(df):
    """FDR correction for RCS nonlinearity P-values."""
    print("\n" + "=" * 70)
    print("2. FDR Correction for RCS P-values")
    print("=" * 70)

    pairs = [
        ('RFTN_total_mcg_kg', 'HR_rebound', 'Total dose → HR rebound'),
        ('RFTN_total_mcg_kg', 'MAP_rebound', 'Total dose → MAP rebound'),
        ('RFTN_total_mcg_kg', 'FTN_rescue_mcg_kg', 'Total dose → FTN rescue'),
        ('RFTN_total_mcg_kg', 'NHD_pct', 'Total dose → NHD index'),
        ('RFTN_total_mcg_kg', 'OSI', 'Total dose → OWSI'),
    ]

    p_values = []
    labels = []
    for exposure, outcome, label in pairs:
        if exposure in df.columns and outcome in df.columns:
            res = rcs_analysis(df, exposure, outcome)
            if res:
                p_values.append(res['p_nonlinear'])
                labels.append(label)
                print(f"  {label}: P = {res['p_nonlinear']:.6f}, n = {res['n']}")

    from statsmodels.stats.multitest import multipletests
    if len(p_values) >= 2:
        rejected_bh, q_values_bh, _, _ = multipletests(p_values, method='fdr_bh')
        rejected_bonf, p_bonf, _, _ = multipletests(p_values, method='bonferroni')

        results = {'tests': []}
        print(f"\n  {'Analysis':<35} {'P_raw':>10} {'q_BH':>10} {'P_Bonf':>10} {'Sig_BH':>8}")
        print("  " + "-" * 75)
        for i, label in enumerate(labels):
            entry = {
                'label': label,
                'p_raw': p_values[i],
                'q_BH': q_values_bh[i],
                'p_bonferroni': p_bonf[i],
                'sig_BH_005': bool(rejected_bh[i]),
                'sig_bonf_005': bool(rejected_bonf[i])
            }
            results['tests'].append(entry)
            print(f"  {label:<35} {p_values[i]:>10.4f} {q_values_bh[i]:>10.4f} "
                  f"{p_bonf[i]:>10.4f} {'*' if rejected_bh[i] else 'NS':>8}")

        n_sig_raw = sum(1 for p in p_values if p < 0.05)
        n_sig_bh = sum(rejected_bh)
        n_sig_bonf = sum(rejected_bonf)
        results['summary'] = {
            'n_tests': len(p_values),
            'n_significant_raw': n_sig_raw,
            'n_significant_BH': int(n_sig_bh),
            'n_significant_bonferroni': int(n_sig_bonf)
        }
        print(f"\n  Significant: raw={n_sig_raw}/5, BH={n_sig_bh}/5, Bonferroni={n_sig_bonf}/5")
        return results
    return {}


def analysis_3_cessation_proxy(df):
    """
    Use Ce_at_end as a proxy for pump cessation timing.
    If Ce_at_end ≈ 0: cessation well before surgery end → longer washout.
    If Ce_at_end > 0: cessation at/near surgery end → shorter washout.
    """
    print("\n" + "=" * 70)
    print("3. Pump Cessation Time Proxy (Ce_at_end)")
    print("=" * 70)

    results = {}
    exposure = 'RFTN_total_mcg_kg'
    outcome = 'HR_rebound'

    if 'RFTN_Ce_at_end' not in df.columns:
        print("  [SKIP] RFTN_Ce_at_end not available")
        return results

    covariates_base = ['age', 'sex_num', 'bmi', 'asa', 'opdur', 'intraop_ppf']
    covariates_ce = covariates_base + ['RFTN_Ce_at_end']

    print("  a) RCS with Ce_at_end as additional covariate...")
    res_base = rcs_analysis(df, exposure, outcome, covariates=covariates_base)
    res_ce = rcs_analysis(df, exposure, outcome, covariates=covariates_ce)

    if res_base and res_ce:
        results['without_Ce_at_end'] = res_base
        results['with_Ce_at_end'] = res_ce
        print(f"     Without Ce_at_end: n={res_base['n']}, P={res_base['p_nonlinear']:.6f}, R²={res_base['spline_r2']:.4f}")
        print(f"     With Ce_at_end:    n={res_ce['n']}, P={res_ce['p_nonlinear']:.6f}, R²={res_ce['spline_r2']:.4f}")

    print("\n  b) Ce_at_end distribution and correlation with outcomes...")
    ce_end = df['RFTN_Ce_at_end'].dropna()
    results['Ce_at_end_stats'] = {
        'n': int(len(ce_end)),
        'mean': float(ce_end.mean()),
        'median': float(ce_end.median()),
        'sd': float(ce_end.std()),
        'pct_zero': float((ce_end < 0.1).mean() * 100),
        'pct_above_2': float((ce_end > 2.0).mean() * 100)
    }
    print(f"     n={len(ce_end)}, mean={ce_end.mean():.2f}, median={ce_end.median():.2f}")
    print(f"     %<0.1 (pre-cessation): {(ce_end < 0.1).mean()*100:.1f}%")
    print(f"     %>2.0 (active at end): {(ce_end > 2.0).mean()*100:.1f}%")

    print("\n  c) Stratified analysis by Ce_at_end groups...")
    mask = df[[exposure, outcome, 'RFTN_Ce_at_end']].notna().all(axis=1)
    df_valid = df[mask].copy()
    if len(df_valid) > 100:
        df_valid['Ce_end_tertile'] = pd.cut(
            df_valid['RFTN_Ce_at_end'],
            bins=[-0.01, 0.1, 1.0, float('inf')],
            labels=['T1_early_stop', 'T2_mid', 'T3_late_stop']
        )
        strat_results = {}
        for t in ['T1_early_stop', 'T2_mid', 'T3_late_stop']:
            sub = df_valid[df_valid['Ce_end_tertile'] == t]
            if len(sub) > 50:
                rho, p = stats.spearmanr(
                    sub[exposure].values, sub[outcome].values
                )
                strat_results[t] = {
                    'n': int(len(sub)),
                    'rho': float(rho),
                    'p': float(p),
                    'mean_HR_rebound': float(sub[outcome].mean())
                }
                print(f"     {t}: n={len(sub)}, rho={rho:.3f}, P={p:.4f}, "
                      f"mean_HR={sub[outcome].mean():.2f}")
        results['stratified_by_Ce_end'] = strat_results

    return results


def analysis_4_taper_fit_quality(df):
    """Report taper slope linear fit R² and alternative definitions."""
    print("\n" + "=" * 70)
    print("4. Taper Slope Linear Fit Quality")
    print("=" * 70)

    results = {}

    if 'RFTN_taper_slope' not in df.columns:
        print("  [SKIP] Taper slope not available")
        return results

    taper = df['RFTN_taper_slope'].dropna()
    results['taper_stats'] = {
        'n': int(len(taper)),
        'mean': float(taper.mean()),
        'median': float(taper.median()),
        'sd': float(taper.std()),
        'pct_negative': float((taper < 0).mean() * 100)
    }
    print(f"  Taper slope: n={len(taper)}, mean={taper.mean():.4f}, "
          f"median={taper.median():.4f}, %negative={((taper<0).mean()*100):.1f}%")

    print("\n  Note: Taper slope R² from linear fit of Ce over last 30 min is not "
          "stored in the master dataset. To compute it, raw second-by-second Ce data "
          "would need to be re-extracted from VitalDB.")
    print("  Recommendation: Add R² computation to oih_01c_fast_download.py in "
          "the taper slope calculation block, then re-run data extraction.")

    results['recommendation'] = (
        "Add taper_r2 field to data extraction pipeline. "
        "Linear fit R² of Ce over last 30 min provides quality metric for the "
        "linearity assumption. Cases with poor fit (R² < 0.3) likely have "
        "step-wise or exponential Ce trajectories where linear slope is misleading."
    )

    return results


def analysis_5_alternative_baseline(df):
    """
    Sensitivity: use last 10 or 15 minutes as baseline instead of last 30.
    Requires recomputation from raw data — report feasibility assessment.
    """
    print("\n" + "=" * 70)
    print("5. Alternative Baseline Definition Assessment")
    print("=" * 70)

    results = {}

    print("  Current baseline: mean HR/MAP during final 30 min before surgery end.")
    print("  Alternative baselines (last 10, 15 min) would require re-extraction")
    print("  of raw hemodynamic data from VitalDB.")
    print()
    print("  Available proxy analysis: correlation between HR_rebound and opdur")
    print("  (if baseline artifact drives the finding, shorter surgeries should")
    print("  show larger rebound because of less low-stimulus time in baseline)")

    mask = df[['HR_rebound', 'opdur', 'RFTN_total_mcg_kg']].notna().all(axis=1)
    df_v = df[mask]
    if len(df_v) > 100:
        rho, p = stats.spearmanr(df_v['opdur'], df_v['HR_rebound'])
        results['opdur_HR_rebound_correlation'] = {'rho': float(rho), 'p': float(p), 'n': int(len(df_v))}
        print(f"  opdur ↔ HR_rebound: rho={rho:.3f}, P={p:.4g}, n={len(df_v)}")

        for threshold in [120, 150, 180]:
            sub = df_v[df_v['opdur'] >= threshold]
            if len(sub) > 100:
                rho_sub, p_sub = stats.spearmanr(
                    sub['RFTN_total_mcg_kg'], sub['HR_rebound']
                )
                results[f'dose_HR_in_opdur_gte_{threshold}'] = {
                    'n': int(len(sub)), 'rho': float(rho_sub), 'p': float(p_sub)
                }
                print(f"  In opdur >= {threshold} min: dose↔HR rho={rho_sub:.3f}, "
                      f"P={p_sub:.4g}, n={len(sub)}")

    results['recommendation'] = (
        "Re-extract HR/MAP with 10-min and 15-min baseline windows from VitalDB. "
        "This would require modifications to oih_01c_fast_download.py's "
        "compute_hemo() function."
    )
    return results


def main():
    print("=" * 70)
    print("  OIH Study - Phase 7: Supplementary Sensitivity Analyses")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    df = load_data()

    all_results = {}

    all_results['s8_fix'] = analysis_1_fix_s8(df)
    all_results['fdr_correction'] = analysis_2_fdr_correction(df)
    all_results['cessation_proxy'] = analysis_3_cessation_proxy(df)
    all_results['taper_fit_quality'] = analysis_4_taper_fit_quality(df)
    all_results['alternative_baseline'] = analysis_5_alternative_baseline(df)

    all_results['metadata'] = {
        'script': 'oih_07_sensitivity_supplement.py',
        'timestamp': datetime.now().isoformat(),
        'total_cases': len(df),
        'valid_rftn': int(df['RFTN_total_mcg_kg'].notna().sum()),
        'purpose': 'Supplementary sensitivity analyses based on external peer review'
    }

    out_path = RESULTS_DIR / "supplementary_sensitivity.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*70}")
    print(f"  All results saved to {out_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
