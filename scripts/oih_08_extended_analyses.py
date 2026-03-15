#!/usr/bin/env python3
"""
=================================================================
OIH Study - Phase 8: Extended Analyses
=================================================================
1. Taper slope linear fit R² (requires VitalDB re-download)
2. Alternative baseline definitions (last 10/15 min)
3. GPS (Generalized Propensity Score) continuous-exposure analysis
4. Figure 2 with rug plots
=================================================================
"""
import os, sys, io, json, time, asyncio, warnings
import numpy as np
import pandas as pd
import aiohttp
from pathlib import Path
from datetime import datetime
from scipy import stats

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent.parent
SHARED_DIR = PROJECT_DIR.parent / "shared_data"
OIH_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = PROJECT_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://api.vitaldb.net"
MAX_CONCURRENT = 30
BATCH_SIZE = 100
SUPPLEMENT_FILE = OIH_DIR / "oih_supplement_metrics.csv"

# ============================================================
# Part A: VitalDB Re-download for Taper R² + Alt Baselines
# ============================================================

async def download_track(session, tid, sem):
    """Download a single track from VitalDB API. Returns (times, values) tuple."""
    if tid is None:
        return None
    async with sem:
        try:
            async with session.get(f"{API_URL}/{tid}",
                                   timeout=aiohttp.ClientTimeout(total=90)) as resp:
                if resp.status != 200:
                    return None
                text = await resp.text()
        except Exception:
            return None
    try:
        arr = np.genfromtxt(io.StringIO(text), delimiter=',')
        if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) >= 30:
            return (arr[:, 0], arr[:, 1])
        elif arr.ndim == 1 and len(arr) >= 30:
            return (np.arange(len(arr)), arr)
        return None
    except Exception:
        return None


def _extract_window(times, values, t_start, t_end):
    """Extract values within a time window [t_start, t_end]."""
    mask = (times >= t_start) & (times <= t_end) & (~np.isnan(values))
    return values[mask]


def compute_supplement(ce_tv, hr_tv, map_tv, caseid, opstart_sec, opend_sec, weight):
    """Compute taper R², alternative baselines, and delta-HR-max.
    Each *_tv argument is a (times, values) tuple or None.
    """
    r = {'caseid': caseid}

    # --- Taper R² ---
    if ce_tv is not None:
        ce_t, ce_v = ce_tv
        for window_min in [10, 15, 20, 30]:
            t_start = opend_sec - window_min * 60
            wv = _extract_window(ce_t, ce_v, t_start, opend_sec)
            if len(wv) > 30:
                x = np.arange(len(wv))
                coeffs = np.polyfit(x, wv, 1)
                y_pred = np.polyval(coeffs, x)
                ss_res = np.sum((wv - y_pred) ** 2)
                ss_tot = np.sum((wv - np.mean(wv)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                suffix = '' if window_min == 30 else f'_{window_min}min'
                r[f'taper_r2{suffix}'] = float(r2)
                r[f'taper_slope{suffix}_check'] = float(coeffs[0] * 60)
            else:
                suffix = '' if window_min == 30 else f'_{window_min}min'
                r[f'taper_r2{suffix}'] = np.nan

    # --- Alternative baseline HR/MAP rebound ---
    if hr_tv is not None:
        hr_t, hr_v = hr_tv
        stable_hr = _extract_window(hr_t, hr_v, opstart_sec + 30*60, opend_sec - 60*60)
        hr_stable = float(np.mean(stable_hr)) if len(stable_hr) > 30 else np.nan

        for window_min in [10, 15, 30]:
            late_hr = _extract_window(hr_t, hr_v, opend_sec - window_min*60, opend_sec)
            if len(late_hr) > 10 and not np.isnan(hr_stable):
                r[f'HR_rebound_{window_min}min'] = float(np.mean(late_hr) - hr_stable)

        post_hr = _extract_window(hr_t, hr_v, opend_sec, opend_sec + 15*60)
        late30_hr = _extract_window(hr_t, hr_v, opend_sec - 30*60, opend_sec)
        if len(post_hr) > 5 and len(late30_hr) > 10:
            hr_late30_mean = float(np.mean(late30_hr))
            r['HR_post15_mean'] = float(np.mean(post_hr))
            r['HR_post15_max'] = float(np.max(post_hr))
            r['HR_delta_max'] = float(np.max(post_hr) - hr_late30_mean)
            r['HR_post_rebound_15min'] = float(np.mean(post_hr) - hr_late30_mean)

    if map_tv is not None:
        map_t, map_v = map_tv
        stable_map = _extract_window(map_t, map_v, opstart_sec + 30*60, opend_sec - 60*60)
        map_stable = float(np.mean(stable_map)) if len(stable_map) > 30 else np.nan
        for window_min in [10, 15, 30]:
            late_map = _extract_window(map_t, map_v, opend_sec - window_min*60, opend_sec)
            if len(late_map) > 10 and not np.isnan(map_stable):
                r[f'MAP_rebound_{window_min}min'] = float(np.mean(late_map) - map_stable)

    return r


async def process_case(session, sem, caseid, tid_map, clin_info):
    """Download and compute metrics for a single case."""
    tids = {}
    for tname_key, tname in [('ce', 'Orchestra/RFTN20_CE'),
                              ('hr', 'Solar8000/HR'),
                              ('map', 'Solar8000/ART_MBP')]:
        row = tid_map[(tid_map['caseid'] == caseid) & (tid_map['tname'] == tname)]
        tids[tname_key] = str(row['tid'].iloc[0]) if len(row) > 0 else None

    if tids['ce'] is None:
        row50 = tid_map[(tid_map['caseid'] == caseid) &
                        (tid_map['tname'] == 'Orchestra/RFTN50_CE')]
        tids['ce'] = str(row50['tid'].iloc[0]) if len(row50) > 0 else None

    tasks = [download_track(session, tids[k], sem) for k in ['ce', 'hr', 'map']]
    results = await asyncio.gather(*tasks)
    ce_tv, hr_tv, map_tv = results

    info = clin_info.loc[caseid] if caseid in clin_info.index else None
    if info is None:
        return None

    weight = float(info['weight']) if pd.notna(info.get('weight')) and info['weight'] > 0 else 60.0
    opstart_sec = float(info['opstart']) if pd.notna(info.get('opstart')) else 0
    opend_sec = float(info['opend']) if pd.notna(info.get('opend')) else opstart_sec + 120 * 60

    if map_tv is None:
        row_nibp = tid_map[(tid_map['caseid'] == caseid) &
                           (tid_map['tname'] == 'Solar8000/NIBP_MBP')]
        if len(row_nibp) > 0:
            map_tv = await download_track(session, str(row_nibp['tid'].iloc[0]), sem)

    return compute_supplement(ce_tv, hr_tv, map_tv, caseid, opstart_sec, opend_sec, weight)


async def download_all(caseids, tid_map, clin_info):
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    all_results = []
    done_set = set()

    if SUPPLEMENT_FILE.exists():
        existing = pd.read_csv(SUPPLEMENT_FILE)
        done_set = set(existing['caseid'].tolist())
        all_results = existing.to_dict('records')
        print(f"  Resuming: {len(done_set)} cases already done")

    remaining = [c for c in caseids if c not in done_set]
    print(f"  Downloading {len(remaining)} remaining cases...")

    import ssl as _ssl
    sslctx = _ssl.create_default_context()
    sslctx.check_hostname = False
    sslctx.verify_mode = _ssl.CERT_NONE
    conn = aiohttp.TCPConnector(limit=MAX_CONCURRENT, ttl_dns_cache=300, ssl=sslctx)
    async with aiohttp.ClientSession(connector=conn) as session:
        for i in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[i:i + BATCH_SIZE]
            tasks = [process_case(session, sem, c, tid_map, clin_info) for c in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if r is not None and not isinstance(r, Exception):
                    all_results.append(r)

            df_out = pd.DataFrame(all_results)
            df_out.to_csv(SUPPLEMENT_FILE, index=False)

            n_done = len(done_set) + i + len(batch)
            pct = n_done / len(caseids) * 100
            print(f"    Batch {i//BATCH_SIZE + 1}: {n_done}/{len(caseids)} ({pct:.0f}%)")

    print(f"  Download complete: {len(all_results)} cases with supplement metrics")
    return pd.DataFrame(all_results)


def run_download():
    """Main download orchestrator."""
    print("=" * 70)
    print("Part A: VitalDB Re-download for Taper R2 + Alt Baselines")
    print("=" * 70)

    eligible = pd.read_csv(OIH_DIR / "oih_eligible_caseids.csv")
    caseids = eligible['caseid'].tolist()

    tid_map = pd.read_csv(SHARED_DIR / "track_list.csv")
    clin_info = pd.read_csv(SHARED_DIR / "clinical_information.csv")
    clin_info = clin_info.set_index('caseid')

    df_supp = asyncio.run(download_all(caseids, tid_map, clin_info))
    return df_supp


# ============================================================
# Part B: GPS Analysis
# ============================================================
def run_gps_analysis():
    """Generalized Propensity Score for continuous remifentanil dose."""
    print("\n" + "=" * 70)
    print("Part B: Generalized Propensity Score (GPS) Analysis")
    print("=" * 70)

    import statsmodels.api as sm
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(OIH_DIR / "oih_master_dataset.csv")
    outlier_mask = (df['RFTN_total_mcg_kg'] > 200) | (df['RFTN_Ce_peak'] > 100)
    rftn_cols = [c for c in df.columns if c.startswith('RFTN_')]
    df.loc[outlier_mask, rftn_cols] = np.nan
    if 'sex' in df.columns:
        df['sex_num'] = (df['sex'] == 'F').astype(int)

    exposure = 'RFTN_total_mcg_kg'
    outcome = 'HR_rebound'
    covariates = ['age', 'sex_num', 'bmi', 'asa', 'opdur', 'intraop_ppf']
    covariates = [c for c in covariates if c in df.columns]

    cols = [exposure, outcome] + covariates
    df_gps = df[cols].dropna().copy()
    print(f"  GPS analytic sample: n = {len(df_gps)}")

    # Step 1: Model treatment (continuous dose) as function of covariates
    X = df_gps[covariates].values
    A = df_gps[exposure].values
    Y = df_gps[outcome].values

    scaler_X = StandardScaler()
    X_sc = scaler_X.fit_transform(X)

    # OLS model: E[A|X]
    X_with_const = sm.add_constant(X_sc)
    dose_model = sm.OLS(A, X_with_const).fit()
    A_pred = dose_model.predict(X_with_const)
    residuals = A - A_pred
    sigma = np.std(residuals, ddof=len(covariates) + 1)

    print(f"  Dose model R2: {dose_model.rsquared:.4f}")
    print(f"  Residual SD: {sigma:.2f} mcg/kg")

    # Step 2: Compute GPS = f(A | X) using normal density
    gps = stats.norm.pdf(A, loc=A_pred, scale=sigma)

    # Step 3: Outcome model: E[Y | A, GPS]
    # Hirano-Imbens approach: flexible model with A, A2, GPS, GPS2, A*GPS
    A_std = (A - A.mean()) / A.std()
    gps_std = (gps - gps.mean()) / gps.std()

    Z_gps = np.column_stack([
        np.ones(len(A)),
        A_std, A_std**2,
        gps_std, gps_std**2,
        A_std * gps_std
    ])

    outcome_model = sm.OLS(Y, Z_gps).fit()
    print(f"  Outcome model R2: {outcome_model.rsquared:.4f}")

    # Step 4: Estimate dose-response curve
    # For each dose level d, compute E[Y(d)] by averaging over GPS distribution
    dose_grid = np.linspace(np.percentile(A, 2.5), np.percentile(A, 97.5), 100)
    dr_curve = []
    dr_ci_low = []
    dr_ci_high = []

    for d in dose_grid:
        gps_at_d = stats.norm.pdf(d, loc=A_pred, scale=sigma)
        d_std = (d - A.mean()) / A.std()
        gps_d_std = (gps_at_d - gps.mean()) / gps.std()

        Z_d = np.column_stack([
            np.ones(len(gps_d_std)),
            np.full(len(gps_d_std), d_std),
            np.full(len(gps_d_std), d_std**2),
            gps_d_std,
            gps_d_std**2,
            d_std * gps_d_std
        ])

        y_pred = outcome_model.predict(Z_d)
        dr_curve.append(float(np.mean(y_pred)))

    # Bootstrap CI
    n_boot = 500
    boot_curves = np.zeros((n_boot, len(dose_grid)))
    print(f"  Bootstrap ({n_boot} iterations)...")

    for b in range(n_boot):
        idx = np.random.choice(len(df_gps), len(df_gps), replace=True)
        Ab, Yb, Xb = A[idx], Y[idx], X_sc[idx]
        Xb_c = sm.add_constant(Xb)

        try:
            dm_b = sm.OLS(Ab, Xb_c).fit()
            Ap_b = dm_b.predict(Xb_c)
            res_b = Ab - Ap_b
            sig_b = np.std(res_b, ddof=len(covariates)+1)
            gps_b = stats.norm.pdf(Ab, loc=Ap_b, scale=sig_b)

            As_b = (Ab - A.mean()) / A.std()
            gs_b = (gps_b - gps.mean()) / gps.std()
            Z_b = np.column_stack([np.ones(len(Ab)), As_b, As_b**2,
                                   gs_b, gs_b**2, As_b * gs_b])
            om_b = sm.OLS(Yb, Z_b).fit()

            for j, d in enumerate(dose_grid):
                gps_d_b = stats.norm.pdf(d, loc=Ap_b, scale=sig_b)
                d_s = (d - A.mean()) / A.std()
                g_s = (gps_d_b - gps.mean()) / gps.std()
                Z_d_b = np.column_stack([np.ones(len(g_s)), np.full(len(g_s), d_s),
                                         np.full(len(g_s), d_s**2), g_s, g_s**2,
                                         d_s * g_s])
                boot_curves[b, j] = np.mean(om_b.predict(Z_d_b))
        except Exception:
            boot_curves[b, :] = np.nan

    dr_ci_low = np.nanpercentile(boot_curves, 2.5, axis=0).tolist()
    dr_ci_high = np.nanpercentile(boot_curves, 97.5, axis=0).tolist()

    results = {
        'n': len(df_gps),
        'dose_model_r2': float(dose_model.rsquared),
        'outcome_model_r2': float(outcome_model.rsquared),
        'dose_grid': dose_grid.tolist(),
        'dr_curve': dr_curve,
        'dr_ci_low': dr_ci_low,
        'dr_ci_high': dr_ci_high,
        'method': 'Hirano-Imbens GPS with normal density, quadratic outcome model',
        'n_bootstrap': n_boot
    }

    # Comparison with binary IPTW
    print(f"\n  GPS dose-response curve computed ({len(dose_grid)} points)")
    y_at_q1 = dr_curve[0]
    y_at_q4 = dr_curve[-1]
    print(f"  HR rebound at dose={dose_grid[0]:.1f}: {y_at_q1:.2f} bpm")
    print(f"  HR rebound at dose={dose_grid[-1]:.1f}: {y_at_q4:.2f} bpm")
    print(f"  GPS-estimated dose-response range: {y_at_q4 - y_at_q1:.2f} bpm")

    with open(RESULTS_DIR / "gps_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {RESULTS_DIR / 'gps_analysis.json'}")

    return results


# ============================================================
# Part C: Taper R² and Alt Baseline Analysis
# ============================================================
def run_taper_altbaseline_analysis(df_supp):
    """Analyze taper R2 and alternative baseline results."""
    print("\n" + "=" * 70)
    print("Part C: Taper R2 and Alternative Baseline Analysis")
    print("=" * 70)

    if df_supp is None or len(df_supp) == 0:
        if SUPPLEMENT_FILE.exists():
            df_supp = pd.read_csv(SUPPLEMENT_FILE)
        else:
            print("  [SKIP] No supplement data available")
            return {}

    results = {}

    # Taper R2 statistics
    if 'taper_r2' in df_supp.columns:
        tr = df_supp['taper_r2'].dropna()
        results['taper_r2_stats'] = {
            'n': int(len(tr)),
            'mean': float(tr.mean()),
            'median': float(tr.median()),
            'sd': float(tr.std()),
            'q25': float(tr.quantile(0.25)),
            'q75': float(tr.quantile(0.75)),
            'pct_above_0.3': float((tr > 0.3).mean() * 100),
            'pct_above_0.5': float((tr > 0.5).mean() * 100),
            'pct_below_0.1': float((tr < 0.1).mean() * 100)
        }
        print(f"  Taper R2 (30 min): n={len(tr)}, median={tr.median():.3f}, "
              f"mean={tr.mean():.3f} +/- {tr.std():.3f}")
        print(f"    R2 > 0.3: {(tr > 0.3).mean()*100:.1f}%, "
              f"R2 > 0.5: {(tr > 0.5).mean()*100:.1f}%, "
              f"R2 < 0.1: {(tr < 0.1).mean()*100:.1f}%")

        for w in [10, 15, 20]:
            col = f'taper_r2_{w}min'
            if col in df_supp.columns:
                tw = df_supp[col].dropna()
                results[f'taper_r2_{w}min'] = {
                    'n': int(len(tw)),
                    'median': float(tw.median()),
                    'mean': float(tw.mean()),
                    'pct_above_0.3': float((tw > 0.3).mean() * 100)
                }
                print(f"  Taper R2 ({w} min): median={tw.median():.3f}, "
                      f"R2>0.3: {(tw > 0.3).mean()*100:.1f}%")

    # Alternative baseline analysis
    master = pd.read_csv(OIH_DIR / "oih_master_dataset.csv")
    outlier_mask = (master['RFTN_total_mcg_kg'] > 200) | (master['RFTN_Ce_peak'] > 100)
    rftn_cols = [c for c in master.columns if c.startswith('RFTN_')]
    master.loc[outlier_mask, rftn_cols] = np.nan

    df_merged = master.merge(df_supp[['caseid'] + [c for c in df_supp.columns
                                                     if 'rebound' in c and c != 'caseid']],
                              on='caseid', how='left')

    import statsmodels.api as sm
    if 'sex' in df_merged.columns:
        df_merged['sex_num'] = (df_merged['sex'] == 'F').astype(int)

    covariates = ['age', 'sex_num', 'bmi', 'asa', 'opdur', 'intraop_ppf']
    covariates = [c for c in covariates if c in df_merged.columns]
    exposure = 'RFTN_total_mcg_kg'

    alt_results = {}
    for window in [10, 15, 30]:
        hr_col = f'HR_rebound_{window}min'
        if hr_col in df_merged.columns:
            cols = [exposure, hr_col] + covariates
            df_clean = df_merged[cols].dropna()
            if len(df_clean) > 100:
                cov_str = " + ".join(covariates)
                linear_f = f"{hr_col} ~ {exposure} + {cov_str}"
                spline_f = f"{hr_col} ~ cr({exposure}, df=3) + {cov_str}"
                try:
                    lm = sm.OLS.from_formula(linear_f, data=df_clean).fit()
                    sm_ = sm.OLS.from_formula(spline_f, data=df_clean).fit()
                    lr = -2 * (lm.llf - sm_.llf)
                    df_diff = sm_.df_model - lm.df_model
                    p = 1 - stats.chi2.cdf(lr, df_diff)

                    rho, p_rho = stats.spearmanr(df_clean[exposure], df_clean[hr_col])

                    alt_results[f'{window}min'] = {
                        'n': int(len(df_clean)),
                        'spearman_rho': float(rho),
                        'spearman_p': float(p_rho),
                        'linear_r2': float(lm.rsquared),
                        'spline_r2': float(sm_.rsquared),
                        'p_nonlinear': float(p),
                        'mean_rebound': float(df_clean[hr_col].mean())
                    }
                    print(f"\n  HR rebound ({window} min baseline): n={len(df_clean)}, "
                          f"rho={rho:.3f}, P_nonlinear={p:.4f}")
                except Exception as e:
                    print(f"  [ERROR] {window} min baseline: {e}")

    results['alternative_baselines'] = alt_results

    # Delta HR max analysis
    if 'HR_delta_max' in df_supp.columns:
        dm = df_supp[['caseid', 'HR_delta_max', 'HR_post_rebound_15min']].dropna()
        merge_covs = [c for c in covariates if c in master.columns]
        dm_merged = dm.merge(master[['caseid', exposure] + merge_covs], on='caseid')
        dm_clean = dm_merged.dropna()
        if len(dm_clean) > 50:
            rho, p = stats.spearmanr(dm_clean[exposure], dm_clean['HR_delta_max'])
            results['delta_hr_max'] = {
                'n': int(len(dm_clean)),
                'rho': float(rho), 'p': float(p),
                'mean': float(dm_clean['HR_delta_max'].mean()),
                'sd': float(dm_clean['HR_delta_max'].std())
            }
            print(f"\n  Delta HR max (post 15 min peak - late 30 min mean): "
                  f"n={len(dm_clean)}, rho={rho:.3f}, P={p:.4f}")

            rho2, p2 = stats.spearmanr(dm_clean[exposure], dm_clean['HR_post_rebound_15min'])
            results['hr_post_rebound_15min'] = {
                'n': int(len(dm_clean)),
                'rho': float(rho2), 'p': float(p2),
                'mean': float(dm_clean['HR_post_rebound_15min'].mean())
            }
            print(f"  HR post-rebound (post 15 min mean - late 30 min mean): "
                  f"rho={rho2:.3f}, P={p2:.4f}")

    with open(RESULTS_DIR / "taper_altbaseline.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to {RESULTS_DIR / 'taper_altbaseline.json'}")

    return results


# ============================================================
# Part D: Figure 2 with Rug Plots
# ============================================================
def run_figure2_rug():
    """Regenerate Figure 2 RCS composite with rug plots."""
    print("\n" + "=" * 70)
    print("Part D: Figure 2 with Data Density Rug Plots")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 9,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    df = pd.read_csv(OIH_DIR / "oih_master_dataset.csv")
    outlier_mask = (df['RFTN_total_mcg_kg'] > 200) | (df['RFTN_Ce_peak'] > 100)
    rftn_cols = [c for c in df.columns if c.startswith('RFTN_')]
    df.loc[outlier_mask, rftn_cols] = np.nan
    if 'sex' in df.columns:
        df['sex_num'] = (df['sex'] == 'F').astype(int)

    import statsmodels.api as sm

    exposure = 'RFTN_total_mcg_kg'
    covariates = ['age', 'sex_num', 'bmi', 'asa', 'opdur', 'intraop_ppf']
    covariates = [c for c in covariates if c in df.columns]
    cov_str = " + ".join(covariates)

    panels = [
        ('HR_rebound', 'HR Rebound (bpm)', 'A'),
        ('MAP_rebound', 'MAP Rebound (mmHg)', 'B'),
        ('OSI', 'OWSI (composite Z-score)', 'C'),
    ]

    # Construct OSI
    oih_comps = ['HR_rebound', 'MAP_rebound', 'FTN_rescue_mcg_kg']
    for c in oih_comps:
        if c in df.columns:
            m, s = df[c].mean(), df[c].std()
            df[f'{c}_Z'] = (df[c] - m) / s if s > 0 else 0
    z_cols = [f'{c}_Z' for c in oih_comps if c in df.columns]
    df['OSI'] = df[z_cols].mean(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for idx, (outcome, ylabel, panel_label) in enumerate(panels):
        ax = axes[idx]
        cols = [exposure, outcome] + covariates
        dfc = df[cols].dropna().copy()

        spline_formula = f"{outcome} ~ cr({exposure}, df=3) + {cov_str}"
        model = sm.OLS.from_formula(spline_formula, data=dfc).fit()

        x_pred = np.linspace(dfc[exposure].quantile(0.025),
                             dfc[exposure].quantile(0.975), 200)
        pred_df = pd.DataFrame({exposure: x_pred})
        for cov in covariates:
            pred_df[cov] = dfc[cov].median()

        y_pred = model.predict(pred_df)

        n_boot = 500
        y_boot = np.zeros((n_boot, len(x_pred)))
        for b in range(n_boot):
            bid = np.random.choice(len(dfc), len(dfc), replace=True)
            bdf = dfc.iloc[bid]
            try:
                bm = sm.OLS.from_formula(spline_formula, data=bdf).fit()
                y_boot[b, :] = bm.predict(pred_df)
            except Exception:
                y_boot[b, :] = np.nan

        y_lo = np.nanpercentile(y_boot, 2.5, axis=0)
        y_hi = np.nanpercentile(y_boot, 97.5, axis=0)

        # P_nonlinear
        linear_f = f"{outcome} ~ {exposure} + {cov_str}"
        lm = sm.OLS.from_formula(linear_f, data=dfc).fit()
        lr_stat = -2 * (lm.llf - model.llf)
        df_diff = model.df_model - lm.df_model
        p_nl = 1 - stats.chi2.cdf(lr_stat, df_diff)

        # Plot
        ax.fill_between(x_pred, y_lo, y_hi, alpha=0.15, color='#2171B5')
        ax.plot(x_pred, y_pred, color='#2171B5', linewidth=2)
        ax.axhline(y=0, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)

        # Rug plot
        rug_data = dfc[exposure].values
        ax.plot(rug_data, np.full_like(rug_data, ax.get_ylim()[0]),
                '|', color='#2171B5', alpha=0.05, markersize=3, markeredgewidth=0.3)

        # Annotations
        p_str = f"P = {p_nl:.3f}" if p_nl >= 0.001 else f"P < 0.001"
        ax.text(0.97, 0.97, f"n = {len(dfc):,}\n{p_str}",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='white', alpha=0.8, edgecolor='grey'))

        ax.set_xlabel('Remifentanil Total Dose (mcg/kg)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f'({panel_label})', fontsize=10, fontweight='bold', loc='left')
        ax.tick_params(labelsize=8)

        if outcome == 'OSI' and p_nl > 0.05:
            ax.text(0.5, 0.85, 'NS', transform=ax.transAxes, ha='center',
                    fontsize=14, fontweight='bold', color='grey', alpha=0.5)

    # Rug plot needs to be re-drawn after ylim is set
    for idx, (outcome, ylabel, panel_label) in enumerate(panels):
        ax = axes[idx]
        cols = [exposure, outcome] + covariates
        dfc = df[cols].dropna()
        rug_data = dfc[exposure].values
        ymin = ax.get_ylim()[0]
        ax.plot(rug_data, np.full_like(rug_data, ymin),
                '|', color='#2171B5', alpha=0.05, markersize=4, markeredgewidth=0.3)

    plt.tight_layout(w_pad=3)

    for fmt, path in [('pdf', FIG_DIR / 'fig2_rcs_with_rug.pdf'),
                      ('png', FIG_DIR / 'fig2_rcs_with_rug.png')]:
        fig.savefig(path, format=fmt, bbox_inches='tight')
    plt.close()
    print(f"  Saved Figure 2 with rug plots to {FIG_DIR}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  OIH Study - Phase 8: Extended Analyses")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Part A: Download (can be skipped if data exists)
    if not SUPPLEMENT_FILE.exists():
        df_supp = run_download()
    else:
        print(f"\n  Supplement data exists: {SUPPLEMENT_FILE}")
        df_supp = pd.read_csv(SUPPLEMENT_FILE)
        remaining = 4443 - len(df_supp)
        if remaining > 100:
            print(f"  {remaining} cases remaining, continuing download...")
            df_supp = run_download()
        else:
            print(f"  {len(df_supp)} cases complete, skipping download")

    # Part B: GPS
    gps_results = run_gps_analysis()

    # Part C: Taper R2 + Alt baseline
    taper_results = run_taper_altbaseline_analysis(df_supp)

    # Part D: Figure 2
    run_figure2_rug()

    print("\n" + "=" * 70)
    print("  Phase 8 Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
