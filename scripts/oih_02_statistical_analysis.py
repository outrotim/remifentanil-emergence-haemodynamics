#!/usr/bin/env python3
"""
=================================================================
OIH Study - Phase 2: Statistical Analysis
=================================================================
围术期瑞芬太尼诱导痛觉过敏（OIH）数据挖掘
统计分析Pipeline

功能：
1. 描述性分析 + Table 1
2. RCS剂量-反应曲线
3. 分段回归阈值检测
4. E_max药理学模型拟合
5. 年龄交互效应分析（核心分析）
6. 输注模式聚类分析
7. 多模式镇痛保护效应
8. 预测模型 + SHAP
9. 敏感性分析
=================================================================
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
OIH_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = PROJECT_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. 数据加载与预处理
# ============================================================
def load_and_prepare():
    """加载合并数据集并进行预处理"""
    print("=" * 70)
    print("1. Loading master dataset...")
    print("=" * 70)

    df = pd.read_csv(OIH_DIR / "oih_master_dataset.csv")
    print(f"  Loaded: {len(df)} cases \u00d7 {len(df.columns)} columns")

    # ---- 异常值清理 ----
    # 临床合理范围：RFTN_total_mcg_kg < 200 \u03bcg/kg (P99\u224880)
    # RFTN_Ce_peak < 100 ng/mL (正常 2-20, 极端不超过 30)
    n_before = len(df)
    outlier_mask = (
        (df['RFTN_total_mcg_kg'] > 200) |
        (df['RFTN_Ce_peak'] > 100)
    )
    outlier_ids = df.loc[outlier_mask, 'caseid'].tolist()
    # 将异常值的 RFTN 指标全部设为 NaN（而非删除整行，保留血流动力学数据）
    rftn_cols = [c for c in df.columns if c.startswith('RFTN_') or c in ['rftn_conc']]
    df.loc[outlier_mask, rftn_cols] = np.nan
    print(f"  Outlier cleanup: {outlier_mask.sum()} cases with extreme RFTN values "
          f"(caseids: {outlier_ids}) \u2192 RFTN columns set to NaN")
    print(f"  Valid RFTN cases after cleanup: {df['RFTN_total_mcg_kg'].notna().sum()}")

    # 年龄分组
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 65, 75, 120],
        labels=['<65', '65-74', '>=75'],
        right=False
    )
    df['elderly'] = (df['age'] >= 65).astype(int)

    # 瑞芬太尼剂量四分位
    if 'RFTN_total_mcg_kg' in df.columns:
        df['RFTN_quartile'] = pd.qcut(
            df['RFTN_total_mcg_kg'].dropna(),
            q=4, labels=['Q1', 'Q2', 'Q3', 'Q4']
        ).reindex(df.index)

    # 复合OIH替代终点 (OSI) 构建
    oih_components = ['HR_rebound', 'MAP_rebound', 'FTN_rescue_mcg_kg']
    available_components = [c for c in oih_components if c in df.columns]

    if available_components:
        for comp in available_components:
            col = f'{comp}_Z'
            mean_val = df[comp].mean()
            std_val = df[comp].std()
            df[col] = (df[comp] - mean_val) / std_val if std_val > 0 else 0

        z_cols = [f'{c}_Z' for c in available_components]
        df['OSI'] = df[z_cols].mean(axis=1)
        print(f"  Constructed OSI from {len(available_components)} components: {available_components}")

    return df


# ============================================================
# 2. 描述性分析 + Table 1
# ============================================================
def descriptive_analysis(df):
    """生成按瑞芬太尼剂量分组的基线特征表"""
    print("\n" + "=" * 70)
    print("2. Descriptive Analysis & Table 1...")
    print("=" * 70)

    if 'RFTN_quartile' not in df.columns:
        print("  [SKIP] RFTN exposure data not available.")
        return

    groups = ['Q1', 'Q2', 'Q3', 'Q4']
    results = []

    # 连续变量
    cont_vars = [
        ('age', 'Age, years'),
        ('bmi', 'BMI, kg/m\u00b2'),
        ('opdur', 'Surgery duration, min'),
        ('anedur', 'Anesthesia duration, min'),
        ('intraop_ppf', 'Propofol total, mg'),
        ('intraop_ebl', 'Estimated blood loss, mL'),
        ('RFTN_total_mcg_kg', 'Remifentanil total, \u03bcg/kg'),
        ('RFTN_Ce_mean', 'Remifentanil Ce mean, ng/mL'),
        ('RFTN_Ce_peak', 'Remifentanil Ce peak, ng/mL'),
        ('RFTN_rate_mean', 'Remifentanil rate mean, \u03bcg/kg/min'),
    ]

    for var, label in cont_vars:
        if var not in df.columns:
            continue
        row = {'Variable': label}
        for g in groups:
            sub = df.loc[df['RFTN_quartile'] == g, var].dropna()
            row[g] = f"{sub.mean():.1f} ({sub.std():.1f})"
        # Kruskal-Wallis P
        group_data = [df.loc[df['RFTN_quartile'] == g, var].dropna() for g in groups]
        group_data = [g for g in group_data if len(g) > 0]
        if len(group_data) >= 2:
            _, p = stats.kruskal(*group_data)
            row['P'] = f"{p:.3f}" if p >= 0.001 else "<0.001"
        results.append(row)

    # 分类变量
    cat_vars = [
        ('sex', 'Female', lambda x: x == 'F'),
        ('elderly', 'Age >= 65', lambda x: x == 1),
        ('emop', 'Emergency', lambda x: x == 1),
        ('preop_htn', 'Hypertension', lambda x: x == 1),
        ('preop_dm', 'Diabetes', lambda x: x == 1),
    ]

    for var, label, cond in cat_vars:
        if var not in df.columns:
            continue
        row = {'Variable': label}
        for g in groups:
            sub = df.loc[df['RFTN_quartile'] == g, var].dropna()
            n_pos = cond(sub).sum()
            row[g] = f"{n_pos} ({n_pos/len(sub)*100:.1f}%)" if len(sub) > 0 else "N/A"
        results.append(row)

    # OIH替代终点
    oih_vars = [
        ('HR_rebound', 'HR rebound, bpm'),
        ('MAP_rebound', 'MAP rebound, mmHg'),
        ('FTN_rescue_mcg_kg', 'Fentanyl rescue, \u03bcg/kg'),
        ('NHD_pct', 'NHD index, %'),
        ('icu_days', 'ICU days'),
        ('OSI', 'OIH Surrogate Index'),
    ]

    for var, label in oih_vars:
        if var not in df.columns:
            continue
        row = {'Variable': f"[Outcome] {label}"}
        for g in groups:
            sub = df.loc[df['RFTN_quartile'] == g, var].dropna()
            row[g] = f"{sub.mean():.2f} ({sub.std():.2f})" if len(sub) > 0 else "N/A"
        group_data = [df.loc[df['RFTN_quartile'] == g, var].dropna() for g in groups]
        group_data = [g for g in group_data if len(g) > 0]
        if len(group_data) >= 2:
            _, p = stats.kruskal(*group_data)
            row['P'] = f"{p:.3f}" if p >= 0.001 else "<0.001"
        results.append(row)

    df_table1 = pd.DataFrame(results)
    df_table1.to_csv(RESULTS_DIR / "table1_by_rftn_quartile.csv", index=False)
    print(f"  Saved Table 1 to {RESULTS_DIR / 'table1_by_rftn_quartile.csv'}")
    print(df_table1.to_string())

    return df_table1


# ============================================================
# 3. RCS剂量-反应曲线（核心分析A）
# ============================================================
def rcs_dose_response(df, exposure='RFTN_total_mcg_kg', outcome='HR_rebound',
                      covariates=None, n_knots=4):
    """
    限制性立方样条（RCS）剂量-反应建模

    Parameters:
    -----------
    df : DataFrame
    exposure : str, 暴露变量
    outcome : str, 结局变量
    covariates : list, 协变量
    n_knots : int, 节点数（3-5）
    """
    print(f"\n  --- RCS: {exposure} \u2192 {outcome} (knots={n_knots}) ---")

    if exposure not in df.columns or outcome not in df.columns:
        print(f"  [SKIP] Variables not available.")
        return None

    if covariates is None:
        covariates = ['age', 'sex', 'bmi', 'asa', 'opdur', 'TWA_BIS', 'intraop_ppf']
    covariates = [c for c in covariates if c in df.columns]

    # 准备数据
    cols = [exposure, outcome] + covariates
    df_clean = df[cols].dropna().copy()
    print(f"  Complete cases: {len(df_clean)}")

    if len(df_clean) < 100:
        print("  [SKIP] Insufficient cases.")
        return None

    try:
        import statsmodels.api as sm
        from patsy import dmatrix

        # 节点位置
        knot_pcts = {3: [10, 50, 90], 4: [5, 35, 65, 95], 5: [5, 27.5, 50, 72.5, 95]}
        knots = np.percentile(df_clean[exposure], knot_pcts[n_knots])
        print(f"  Knots at: {[f'{k:.1f}' for k in knots]}")

        # 使用patsy构建自然样条
        # 编码分类变量
        if 'sex' in covariates:
            df_clean['sex_num'] = (df_clean['sex'] == 'F').astype(int)
            covariates = [c if c != 'sex' else 'sex_num' for c in covariates]

        # 线性模型（用于LRT比较）
        cov_str = " + ".join(covariates)
        linear_formula = f"{outcome} ~ {exposure} + {cov_str}"
        linear_model = sm.OLS.from_formula(linear_formula, data=df_clean).fit()

        # 样条模型
        spline_formula = f"{outcome} ~ cr({exposure}, df={n_knots-1}) + {cov_str}"
        spline_model = sm.OLS.from_formula(spline_formula, data=df_clean).fit()

        # LRT for nonlinearity
        lr_stat = -2 * (linear_model.llf - spline_model.llf)
        df_diff = spline_model.df_model - linear_model.df_model
        p_nonlinear = 1 - stats.chi2.cdf(lr_stat, df_diff)

        print(f"  Linear model R\u00b2: {linear_model.rsquared:.4f}")
        print(f"  Spline model R\u00b2: {spline_model.rsquared:.4f}")
        print(f"  LRT for nonlinearity: \u03c7\u00b2={lr_stat:.2f}, df={df_diff}, P={p_nonlinear:.4f}")
        print(f"  \u2192 {'Significant nonlinearity detected' if p_nonlinear < 0.05 else 'Linear relationship adequate'}")

        # 生成预测曲线数据
        x_pred = np.linspace(
            df_clean[exposure].quantile(0.025),
            df_clean[exposure].quantile(0.975),
            200
        )

        # 创建预测数据框（协变量设为中位数/众数）
        pred_df = pd.DataFrame({exposure: x_pred})
        for cov in covariates:
            if df_clean[cov].dtype in ['float64', 'int64']:
                pred_df[cov] = df_clean[cov].median()
            else:
                pred_df[cov] = df_clean[cov].mode().iloc[0]

        y_pred = spline_model.predict(pred_df)

        # Bootstrap CI
        n_boot = 500
        y_boot = np.zeros((n_boot, len(x_pred)))
        for b in range(n_boot):
            idx = np.random.choice(len(df_clean), len(df_clean), replace=True)
            boot_df = df_clean.iloc[idx]
            try:
                boot_model = sm.OLS.from_formula(spline_formula, data=boot_df).fit()
                y_boot[b, :] = boot_model.predict(pred_df)
            except Exception:
                y_boot[b, :] = np.nan

        y_lower = np.nanpercentile(y_boot, 2.5, axis=0)
        y_upper = np.nanpercentile(y_boot, 97.5, axis=0)

        # 保存结果
        rcs_results = {
            'exposure': exposure,
            'outcome': outcome,
            'n': len(df_clean),
            'n_knots': n_knots,
            'knots': knots.tolist(),
            'linear_r2': linear_model.rsquared,
            'spline_r2': spline_model.rsquared,
            'p_nonlinear': p_nonlinear,
            'curve_data': {
                'x': x_pred.tolist(),
                'y': y_pred.tolist(),
                'y_lower': y_lower.tolist(),
                'y_upper': y_upper.tolist()
            }
        }

        import json
        outfile = RESULTS_DIR / f"rcs_{exposure}_{outcome}.json"
        with open(outfile, 'w') as f:
            json.dump(rcs_results, f, indent=2)
        print(f"  Saved RCS results to {outfile}")

        return rcs_results

    except ImportError as e:
        print(f"  [ERROR] Missing package: {e}")
        return None


# ============================================================
# 4. 分段回归阈值检测（分析B）
# ============================================================
def segmented_regression(df, exposure='RFTN_total_mcg_kg', outcome='HR_rebound',
                         covariates=None):
    """分段回归检测剂量-反应曲线的拐点"""
    print(f"\n  --- Segmented Regression: {exposure} \u2192 {outcome} ---")

    if exposure not in df.columns or outcome not in df.columns:
        print(f"  [SKIP] Variables not available.")
        return None

    if covariates is None:
        covariates = ['age', 'bmi', 'asa', 'opdur']
    covariates = [c for c in covariates if c in df.columns]

    cols = [exposure, outcome] + covariates
    df_clean = df[cols].dropna()
    if len(df_clean) < 100:
        print("  [SKIP] Insufficient cases.")
        return None

    try:
        import statsmodels.api as sm

        # 先对协变量回归取残差
        if covariates:
            cov_formula = f"{outcome} ~ " + " + ".join(covariates)
            cov_model = sm.OLS.from_formula(cov_formula, data=df_clean).fit()
            y_resid = cov_model.resid.values
        else:
            y_resid = df_clean[outcome].values

        x = df_clean[exposure].values

        # 网格搜索最优断点
        search_lo = np.percentile(x, 15)
        search_hi = np.percentile(x, 85)
        search_points = np.linspace(search_lo, search_hi, 200)

        best_bic = np.inf
        best_bp = None
        best_model = None

        for bp in search_points:
            x1 = np.minimum(x, bp)
            x2 = np.maximum(0, x - bp)
            X = np.column_stack([np.ones_like(x), x1, x2])

            try:
                model = sm.OLS(y_resid, X).fit()
                if model.bic < best_bic:
                    best_bic = model.bic
                    best_bp = bp
                    best_model = model
            except Exception:
                continue

        # 对比无断点的线性模型
        X_linear = np.column_stack([np.ones_like(x), x])
        linear_model = sm.OLS(y_resid, X_linear).fit()

        # F-test for breakpoint
        lr_stat = -2 * (linear_model.llf - best_model.llf)
        p_breakpoint = 1 - stats.chi2.cdf(lr_stat, 1)

        print(f"  Best breakpoint: {best_bp:.1f} {exposure}")
        print(f"  Linear BIC: {linear_model.bic:.1f}")
        print(f"  Segmented BIC: {best_bic:.1f}")
        print(f"  LRT P-value: {p_breakpoint:.4f}")

        if best_model is not None:
            slope_below = best_model.params[1]
            slope_above = best_model.params[1] + best_model.params[2]
            print(f"  Slope below breakpoint: {slope_below:.4f}")
            print(f"  Slope above breakpoint: {slope_above:.4f}")

            # Bootstrap CI for breakpoint
            n_boot = 1000
            bp_boot = []
            for _ in range(n_boot):
                idx = np.random.choice(len(x), len(x), replace=True)
                x_b = x[idx]
                y_b = y_resid[idx]

                best_bic_b = np.inf
                best_bp_b = None
                for bp_try in search_points[::5]:  # 粗搜索
                    x1 = np.minimum(x_b, bp_try)
                    x2 = np.maximum(0, x_b - bp_try)
                    X_b = np.column_stack([np.ones_like(x_b), x1, x2])
                    try:
                        m = sm.OLS(y_b, X_b).fit()
                        if m.bic < best_bic_b:
                            best_bic_b = m.bic
                            best_bp_b = bp_try
                    except Exception:
                        continue
                if best_bp_b is not None:
                    bp_boot.append(best_bp_b)

            bp_ci = np.percentile(bp_boot, [2.5, 97.5]) if bp_boot else [np.nan, np.nan]
            print(f"  Breakpoint 95% CI: [{bp_ci[0]:.1f}, {bp_ci[1]:.1f}]")

            return {
                'breakpoint': best_bp,
                'breakpoint_ci': bp_ci.tolist(),
                'slope_below': slope_below,
                'slope_above': slope_above,
                'p_breakpoint': p_breakpoint,
                'n': len(df_clean)
            }

    except ImportError as e:
        print(f"  [ERROR] Missing package: {e}")
        return None


# ============================================================
# 5. E_max药理学模型（分析C）
# ============================================================
def emax_model_fit(df, exposure='RFTN_total_mcg_kg', outcome='HR_rebound'):
    """拟合S型E_max模型"""
    print(f"\n  --- E_max Model: {exposure} \u2192 {outcome} ---")

    if exposure not in df.columns or outcome not in df.columns:
        return None

    results = {}

    for group_name, sub_df in [
        ('Overall', df),
        ('Young (<65)', df[df['age'] < 65]),
        ('Elderly (>=65)', df[df['age'] >= 65]),
        ('Very elderly (>=75)', df[df['age'] >= 75])
    ]:
        x = sub_df[exposure].dropna().values
        y = sub_df[outcome].dropna().values

        # 对齐
        mask = sub_df[[exposure, outcome]].notna().all(axis=1)
        x = sub_df.loc[mask, exposure].values
        y = sub_df.loc[mask, outcome].values

        if len(x) < 30:
            print(f"  {group_name}: insufficient data (n={len(x)})")
            continue

        def emax_func(dose, e0, emax, ed50, n):
            return e0 + (emax * np.power(dose, n)) / (np.power(ed50, n) + np.power(dose, n))

        # 初始参数
        p0 = [np.percentile(y, 10), np.percentile(y, 90) - np.percentile(y, 10),
              np.median(x), 1.5]
        bounds = ([-np.inf, 0, 0.01, 0.3], [np.inf, np.inf, np.inf, 10])

        try:
            popt, pcov = curve_fit(emax_func, x, y, p0=p0, bounds=bounds,
                                    maxfev=20000)
            perr = np.sqrt(np.diag(pcov))

            results[group_name] = {
                'n': len(x),
                'E0': f"{popt[0]:.3f} (SE {perr[0]:.3f})",
                'Emax': f"{popt[1]:.3f} (SE {perr[1]:.3f})",
                'ED50': f"{popt[2]:.1f} (SE {perr[2]:.1f})",
                'Hill_n': f"{popt[3]:.2f} (SE {perr[3]:.2f})",
                'ED50_value': popt[2],
                'Hill_value': popt[3],
            }
            print(f"  {group_name} (n={len(x)}): ED50={popt[2]:.1f}, Hill n={popt[3]:.2f}")

        except (RuntimeError, ValueError) as e:
            print(f"  {group_name}: Fitting failed - {e}")
            results[group_name] = None

    # ED50差异
    if results.get('Young (<65)') and results.get('Elderly (>=65)'):
        ed50_y = results['Young (<65)']['ED50_value']
        ed50_e = results['Elderly (>=65)']['ED50_value']
        ratio = ed50_e / ed50_y if ed50_y > 0 else np.nan
        print(f"\n  ED50 ratio (Elderly/Young): {ratio:.2f}")
        print(f"  \u2192 Elderly threshold is {'lower' if ratio < 1 else 'higher'} "
              f"({'supports' if ratio < 1 else 'against'} left-shift hypothesis)")

    return results


# ============================================================
# 6. 年龄交互效应（核心分析D-F）
# ============================================================
def age_interaction_analysis(df, exposure='RFTN_total_mcg_kg', outcome='HR_rebound'):
    """年龄\u00d7瑞芬太尼剂量交互效应"""
    print(f"\n  --- Age Interaction: {exposure} \u00d7 Age \u2192 {outcome} ---")

    if exposure not in df.columns or outcome not in df.columns:
        return None

    covariates = ['bmi', 'asa', 'opdur', 'TWA_BIS', 'intraop_ppf']
    covariates = [c for c in covariates if c in df.columns]

    cols = [exposure, outcome, 'age', 'elderly'] + covariates
    if 'sex' in df.columns:
        cols.append('sex')
    df_clean = df[cols].dropna().copy()

    if 'sex' in df_clean.columns:
        df_clean['sex_num'] = (df_clean['sex'] == 'F').astype(int)

    if len(df_clean) < 100:
        print("  [SKIP] Insufficient data.")
        return None

    try:
        import statsmodels.api as sm

        cov_str = " + ".join(covariates)
        if 'sex' in df.columns:
            cov_str += " + sex_num"

        # 模型1：无交互项
        formula_main = f"{outcome} ~ {exposure} + age + {cov_str}"
        model_main = sm.OLS.from_formula(formula_main, data=df_clean).fit()

        # 模型2：线性交互项
        formula_interact = f"{outcome} ~ {exposure} * elderly + age + {cov_str}"
        model_interact = sm.OLS.from_formula(formula_interact, data=df_clean).fit()

        # 模型3：连续年龄交互
        formula_cont = f"{outcome} ~ {exposure} * age + {cov_str}"
        if 'sex' in df.columns:
            formula_cont += " + sex_num"
        model_cont = sm.OLS.from_formula(formula_cont, data=df_clean).fit()

        # LRT for interaction
        lr_stat = -2 * (model_main.llf - model_interact.llf)
        p_interact = 1 - stats.chi2.cdf(lr_stat, 1)

        print(f"\n  Main effects model R\u00b2: {model_main.rsquared:.4f}")
        print(f"  Interaction model R\u00b2:  {model_interact.rsquared:.4f}")
        print(f"  LRT for interaction:   P = {p_interact:.4f}")

        # 交互项系数
        interact_term = f"{exposure}:elderly"
        if interact_term in model_interact.params.index:
            coef = model_interact.params[interact_term]
            se = model_interact.bse[interact_term]
            p = model_interact.pvalues[interact_term]
            print(f"\n  Interaction coefficient ({interact_term}):")
            print(f"    \u03b2 = {coef:.4f} (SE = {se:.4f}), P = {p:.4f}")
            print(f"    Interpretation: In elderly, each unit increase in {exposure}")
            print(f"    is associated with {coef:.4f} additional units of {outcome}")

        # 分层效应
        print(f"\n  Stratified effects:")
        for group, label in [(0, 'Young (<65)'), (1, 'Elderly (>=65)')]:
            sub = df_clean[df_clean['elderly'] == group]
            if len(sub) < 30:
                continue
            sub_cov = " + ".join(covariates)
            if 'sex_num' in sub.columns:
                sub_cov += " + sex_num"
            sub_formula = f"{outcome} ~ {exposure} + {sub_cov}"
            sub_model = sm.OLS.from_formula(sub_formula, data=sub).fit()
            coef_sub = sub_model.params[exposure]
            se_sub = sub_model.bse[exposure]
            p_sub = sub_model.pvalues[exposure]
            print(f"    {label} (n={len(sub)}): \u03b2={coef_sub:.4f} (SE={se_sub:.4f}), P={p_sub:.4f}")

        return {
            'p_interaction': p_interact,
            'model_main_r2': model_main.rsquared,
            'model_interact_r2': model_interact.rsquared,
            'model_summary': model_interact.summary().as_text()
        }

    except ImportError as e:
        print(f"  [ERROR] Missing package: {e}")
        return None


# ============================================================
# 7. 输注模式聚类（分析G）
# ============================================================
def infusion_pattern_clustering(df, n_clusters=4):
    """基于瑞芬太尼暴露特征的聚类分析"""
    print(f"\n  --- Infusion Pattern Clustering (k={n_clusters}) ---")

    cluster_features = [
        'RFTN_Ce_mean', 'RFTN_Ce_peak', 'RFTN_Ce_CV', 'RFTN_Ce_ARV',
        'RFTN_rate_mean', 'RFTN_rate_peak', 'RFTN_Ct_changes',
        'RFTN_taper_slope', 'Time_Ce_above_4'
    ]
    available = [f for f in cluster_features if f in df.columns]

    if len(available) < 3:
        print("  [SKIP] Insufficient features for clustering.")
        return None

    df_feat = df[['caseid'] + available].dropna().copy()
    if len(df_feat) < 50:
        print("  [SKIP] Too few complete cases.")
        return None

    print(f"  Features: {available}")
    print(f"  Complete cases: {len(df_feat)}")

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score

        X = df_feat[available].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 最优K选择（肘部法 + 轮廓系数）
        inertias = []
        silhouettes = []
        K_range = range(2, 7)

        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X_scaled, labels))

        best_k = list(K_range)[np.argmax(silhouettes)]
        print(f"\n  Silhouette scores: {dict(zip(K_range, [f'{s:.3f}' for s in silhouettes]))}")
        print(f"  Best K by silhouette: {best_k}")

        # 用指定k聚类
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_feat['cluster'] = km.fit_predict(X_scaled)

        # 聚类特征描述
        print(f"\n  Cluster characteristics (k={n_clusters}):")
        for c in range(n_clusters):
            sub = df_feat[df_feat['cluster'] == c]
            print(f"\n  Cluster {c} (n={len(sub)}):")
            for feat in available[:5]:
                print(f"    {feat}: {sub[feat].mean():.2f} \u00b1 {sub[feat].std():.2f}")

        # 合并回主数据
        cluster_map = df_feat[['caseid', 'cluster']].set_index('caseid')['cluster']

        return df_feat, cluster_map

    except ImportError as e:
        print(f"  [ERROR] Missing package: {e}")
        return None


# ============================================================
# 8. 预测模型（分析K-L）
# ============================================================
def build_prediction_model(df, target='OSI'):
    """构建OIH风险预测模型"""
    print(f"\n  --- Prediction Model: target = {target} ---")

    if target not in df.columns:
        print("  [SKIP] Target variable not available.")
        return None

    feature_candidates = [
        'age', 'bmi', 'asa', 'opdur', 'emop',
        'preop_htn', 'preop_dm', 'preop_cr', 'preop_hb',
        'RFTN_total_mcg_kg', 'RFTN_Ce_mean', 'RFTN_Ce_peak',
        'RFTN_Ce_CV', 'RFTN_rate_mean', 'RFTN_taper_slope',
        'Time_Ce_above_4', 'RFTN_Ct_changes',
        'TWA_BIS', 'CV_BIS', 'intraop_ppf', 'intraop_ebl',
    ]
    features = [f for f in feature_candidates if f in df.columns]

    df_model = df[features + [target]].dropna()
    if len(df_model) < 100:
        print("  [SKIP] Insufficient complete cases.")
        return None

    print(f"  Features: {len(features)}")
    print(f"  Complete cases: {len(df_model)}")

    try:
        from sklearn.model_selection import cross_val_score, KFold
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        X = df_model[features].values
        y = df_model[target].values
        cv = KFold(n_splits=10, shuffle=True, random_state=42)

        models = {
            'LASSO': Pipeline([
                ('scaler', StandardScaler()),
                ('lasso', LassoCV(cv=5, random_state=42))
            ]),
            'RandomForest': RandomForestRegressor(
                n_estimators=500, max_depth=6, random_state=42
            ),
            'GBM': GradientBoostingRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42
            ),
        }

        print("\n  10-fold CV R\u00b2 scores:")
        best_model_name = None
        best_r2 = -np.inf

        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            mean_r2 = scores.mean()
            print(f"    {name:15s}: R\u00b2 = {mean_r2:.4f} \u00b1 {scores.std():.4f}")
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_model_name = name

        print(f"\n  Best model: {best_model_name} (R\u00b2 = {best_r2:.4f})")

        # SHAP分析
        try:
            import shap

            best_model = models[best_model_name]
            best_model.fit(X, y)

            if best_model_name in ['RandomForest', 'GBM']:
                explainer = shap.TreeExplainer(best_model)
            else:
                # For Pipeline with LASSO, use LinearExplainer on the inner model
                try:
                    inner = best_model.named_steps.get('lasso', best_model)
                    scaler = best_model.named_steps.get('scaler', None)
                    X_scaled = scaler.transform(X) if scaler else X
                    explainer = shap.LinearExplainer(inner, X_scaled)
                except Exception:
                    explainer = shap.Explainer(best_model.predict, X)

            if best_model_name in ['RandomForest', 'GBM']:
                shap_values = explainer.shap_values(X)
            else:
                # LinearExplainer: use scaled X
                scaler = best_model.named_steps.get('scaler', None)
                X_for_shap = scaler.transform(X) if scaler else X
                shap_values = explainer.shap_values(X_for_shap)

            # 特征重要性排序
            if hasattr(shap_values, 'values'):
                shap_values = shap_values.values
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feat_importance = sorted(zip(features, mean_abs_shap),
                                     key=lambda x: x[1], reverse=True)

            print(f"\n  SHAP Feature Importance (top 10):")
            for feat, imp in feat_importance[:10]:
                print(f"    {feat:30s}: {imp:.4f}")

            return {
                'best_model': best_model_name,
                'best_r2': best_r2,
                'features': features,
                'feature_importance': feat_importance,
                'shap_values': shap_values,
            }

        except ImportError:
            print("  [NOTE] shap package not available. Skipping SHAP analysis.")
            return {'best_model': best_model_name, 'best_r2': best_r2}

    except ImportError as e:
        print(f"  [ERROR] Missing package: {e}")
        return None


# ============================================================
# 9. 敏感性分析
# ============================================================
def sensitivity_analyses(df):
    """运行敏感性分析矩阵"""
    print("\n" + "=" * 70)
    print("9. Sensitivity Analyses...")
    print("=" * 70)

    results = {}

    # 主分析暴露/结局对
    main_pairs = [
        ('RFTN_total_mcg_kg', 'HR_rebound', 'Main: total dose \u2192 HR rebound'),
        ('RFTN_AUC_Ce', 'HR_rebound', 'S1: AUC Ce \u2192 HR rebound'),
        ('RFTN_rate_mean', 'HR_rebound', 'S2: mean rate \u2192 HR rebound'),
        ('RFTN_total_mcg_kg', 'MAP_rebound', 'S3: total dose \u2192 MAP rebound'),
        ('RFTN_total_mcg_kg', 'FTN_rescue_mcg_kg', 'S4: total dose \u2192 FTN rescue'),
    ]

    if 'NHD_pct' in df.columns:
        main_pairs.append(('RFTN_total_mcg_kg', 'NHD_pct', 'S5: total dose \u2192 NHD index'))
    if 'OSI' in df.columns:
        main_pairs.append(('RFTN_total_mcg_kg', 'OSI', 'S6: total dose \u2192 OSI'))

    for exposure, outcome, label in main_pairs:
        if exposure in df.columns and outcome in df.columns:
            print(f"\n  [{label}]")
            res = rcs_dose_response(df, exposure, outcome, n_knots=4)
            if res:
                results[label] = {
                    'p_nonlinear': res['p_nonlinear'],
                    'spline_r2': res['spline_r2']
                }

    # S7: 排除急诊手术
    if 'emop' in df.columns:
        print(f"\n  [S7: Excluding emergency cases]")
        df_elec = df[df['emop'] == 0]
        print(f"    Elective cases: {len(df_elec)}")
        res = rcs_dose_response(df_elec, 'RFTN_total_mcg_kg', 'HR_rebound', n_knots=4)
        if res:
            results['S7_elective_only'] = {'p_nonlinear': res['p_nonlinear']}

    # S8: 排除短手术
    if 'opdur' in df.columns:
        print(f"\n  [S8: Excluding short surgery (<90 min)]")
        df_long = df[df['opdur'] >= 90]
        print(f"    Cases with opdur >= 90 min: {len(df_long)}")
        res = rcs_dose_response(df_long, 'RFTN_total_mcg_kg', 'HR_rebound', n_knots=4)
        if res:
            results['S8_long_surgery'] = {'p_nonlinear': res['p_nonlinear']}

    # 保存
    import json
    with open(RESULTS_DIR / "sensitivity_analyses.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved sensitivity results to {RESULTS_DIR / 'sensitivity_analyses.json'}")

    return results


# ============================================================
# Main Analysis Pipeline
# ============================================================
def main():
    """​OIH统计分析主流程"""
    print("=" * 70)
    print("  OIH Study - Phase 2: Statistical Analysis")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 1. 加载数据
    df = load_and_prepare()

    # 2. 描述性分析
    table1 = descriptive_analysis(df)

    # 3. RCS剂量-反应（核心分析A）
    outcomes = ['HR_rebound', 'MAP_rebound', 'FTN_rescue_mcg_kg', 'OSI']
    for outcome in outcomes:
        rcs_dose_response(df, 'RFTN_total_mcg_kg', outcome, n_knots=4)

    # 4. 分段回归（分析B）
    for outcome in ['HR_rebound', 'OSI']:
        segmented_regression(df, 'RFTN_total_mcg_kg', outcome)

    # 5. E_max模型（分析C）
    for outcome in ['HR_rebound', 'OSI']:
        emax_model_fit(df, 'RFTN_total_mcg_kg', outcome)

    # 6. 年龄交互（核心分析D-F）
    for outcome in ['HR_rebound', 'MAP_rebound', 'FTN_rescue_mcg_kg', 'OSI']:
        age_interaction_analysis(df, 'RFTN_total_mcg_kg', outcome)

    # 7. 输注模式聚类（分析G）
    cluster_result = infusion_pattern_clustering(df, n_clusters=4)

    # 8. 预测模型（分析K-L）
    for target in ['OSI', 'HR_rebound']:
        build_prediction_model(df, target)

    # 9. 敏感性分析
    sensitivity_analyses(df)

    print("\n" + "=" * 70)
    print("  Phase 2 Complete! All analyses finished.")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"  Next: Run oih_03_visualization.py")
    print("=" * 70)


if __name__ == '__main__':
    main()
