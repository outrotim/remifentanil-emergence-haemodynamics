#!/usr/bin/env python3
"""
=================================================================
OIH Study - Phase 2b: Extended Causal & Mechanistic Analysis
=================================================================
扩展分析：
1. IPTW因果推断（控制混杂后的剂量-反应关系）
2. 输注速率替代总量分析（解耦手术时间混杂）
3. 高速率暴露亚群分析（>0.2 μg/kg/min）
4. 停药动力学分析（taper slope → 反弹关系）
5. 中介分析（NHD路径）
6. 增强预测模型（交互特征 + 时间特征）
=================================================================
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
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

# ============================================================
# Data loading
# ============================================================
def load_clean_data():
    """加载并清理数据"""
    df = pd.read_csv(OIH_DIR / "oih_master_dataset.csv")

    # 异常值清理
    outlier_mask = (df['RFTN_total_mcg_kg'] > 200) | (df['RFTN_Ce_peak'] > 100)
    rftn_cols = [c for c in df.columns if c.startswith('RFTN_') or c in ['rftn_conc']]
    df.loc[outlier_mask, rftn_cols] = np.nan

    # 年龄分组
    df['age_group'] = pd.cut(df['age'], bins=[0, 65, 75, 120],
                              labels=['<65', '65-74', '>=75'], right=False)
    df['elderly'] = (df['age'] >= 65).astype(int)

    # RFTN 四分位
    df['RFTN_quartile'] = pd.qcut(
        df['RFTN_total_mcg_kg'].dropna(), q=4, labels=['Q1','Q2','Q3','Q4']
    ).reindex(df.index)

    # 输注速率四分位
    df['rate_quartile'] = pd.qcut(
        df['RFTN_rate_mean'].dropna(), q=4, labels=['Q1','Q2','Q3','Q4']
    ).reindex(df.index)

    # OSI
    oih_components = ['HR_rebound', 'MAP_rebound', 'FTN_rescue_mcg_kg']
    for comp in oih_components:
        if comp in df.columns:
            m, s = df[comp].mean(), df[comp].std()
            df[f'{comp}_Z'] = (df[comp] - m) / s if s > 0 else 0
    z_cols = [f'{c}_Z' for c in oih_components if c in df.columns]
    df['OSI'] = df[z_cols].mean(axis=1)

    # 高速率标记
    df['high_rate'] = (df['RFTN_rate_mean'] > 0.2).astype(int)

    print(f"Loaded: {len(df)} cases, valid RFTN: {df['RFTN_total_mcg_kg'].notna().sum()}")
    return df


# ============================================================
# Analysis 1: IPTW因果推断
# ============================================================
def analysis_iptw(df):
    """IPTW加权分析 — 控制混杂因素后的因果效应估计"""
    print("\n" + "=" * 70)
    print("Analysis 1: IPTW Causal Inference")
    print("=" * 70)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    exposure = 'RFTN_rate_mean'

    # 创建二分类处理变量: 高速率 vs 低速率 (以中位数分割)
    rate_med = df[exposure].median()
    df['treat_high_rate'] = (df[exposure] > rate_med).astype(int)

    # 倾向性评分模型的协变量
    confounders = ['age', 'sex', 'bmi', 'asa', 'emop', 'opdur',
                   'preop_htn', 'preop_dm', 'intraop_ppf', 'intraop_ebl']

    # 编码性别
    df['sex_num'] = (df['sex'] == 'M').astype(int)
    confounder_cols = [c if c != 'sex' else 'sex_num' for c in confounders]

    # 完整数据
    needed = confounder_cols + ['treat_high_rate', 'HR_rebound', 'MAP_rebound',
                                 'FTN_rescue_mcg_kg', 'OSI', 'NHD_pct']
    sub = df[needed].dropna()
    print(f"  Complete cases for IPTW: {len(sub)}")

    if len(sub) < 200:
        print("  [SKIP] Insufficient data")
        return {}

    X = sub[confounder_cols].values
    T = sub['treat_high_rate'].values

    # 标准化
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # 倾向性评分
    ps_model = LogisticRegression(max_iter=5000, C=1.0)
    ps_model.fit(X_sc, T)
    ps = ps_model.predict_proba(X_sc)[:, 1]
    ps = np.clip(ps, 0.05, 0.95)  # 截断

    # IPTW 权重
    weights = np.where(T == 1, 1.0 / ps, 1.0 / (1.0 - ps))
    # 稳定化权重
    p_treat = T.mean()
    weights_stable = np.where(T == 1, p_treat / ps, (1 - p_treat) / (1 - ps))

    print(f"  Propensity score: AUC = {_auc(T, ps):.3f}")
    print(f"  Treatment prevalence: {T.mean():.3f}")
    print(f"  Weight range: [{weights_stable.min():.2f}, {weights_stable.max():.2f}]")

    # 协变量平衡检查 (SMD)
    print("\n  Covariate balance (|SMD|):")
    print(f"  {'Variable':<25s} {'Before':>8s} {'After':>8s}")
    smd_before_all = []
    smd_after_all = []
    for i, col in enumerate(confounder_cols):
        smd_before = _smd(X[:, i], T)
        smd_after = _smd_weighted(X[:, i], T, weights_stable)
        smd_before_all.append(abs(smd_before))
        smd_after_all.append(abs(smd_after))
        flag = " \u2713" if abs(smd_after) < 0.1 else " \u2717"
        print(f"  {col:<25s} {abs(smd_before):>8.3f} {abs(smd_after):>8.3f}{flag}")

    print(f"  {'Mean |SMD|':<25s} {np.mean(smd_before_all):>8.3f} {np.mean(smd_after_all):>8.3f}")

    # IPTW加权后的处理效应估计
    results = {}
    outcomes = ['HR_rebound', 'MAP_rebound', 'FTN_rescue_mcg_kg', 'OSI', 'NHD_pct']

    print(f"\n  IPTW-weighted treatment effects (high rate vs low rate):")
    print(f"  {'Outcome':<25s} {'Crude \u0394':>10s} {'IPTW \u0394':>10s} {'95% CI':>20s}")

    for outcome in outcomes:
        y = sub[outcome].values

        # 粗效应
        crude_diff = y[T==1].mean() - y[T==0].mean()

        # IPTW加权效应
        w1 = weights_stable[T==1]
        w0 = weights_stable[T==0]
        iptw_mean1 = np.average(y[T==1], weights=w1)
        iptw_mean0 = np.average(y[T==0], weights=w0)
        iptw_diff = iptw_mean1 - iptw_mean0

        # Bootstrap CI
        ci_low, ci_high = _bootstrap_ci_iptw(y, T, weights_stable, n_boot=1000)

        results[outcome] = {
            'crude_diff': float(crude_diff),
            'iptw_diff': float(iptw_diff),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
            'significant': bool(ci_low > 0 or ci_high < 0)
        }

        sig = " *" if results[outcome]['significant'] else ""
        print(f"  {outcome:<25s} {crude_diff:>+10.3f} {iptw_diff:>+10.3f} "
              f"[{ci_low:>+8.3f}, {ci_high:>+8.3f}]{sig}")

    results['_meta'] = {
        'n': int(len(sub)),
        'ps_auc': float(_auc(T, ps)),
        'mean_smd_after': float(np.mean(smd_after_all)),
        'rate_threshold': float(rate_med)
    }

    with open(RESULTS_DIR / "iptw_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {RESULTS_DIR / 'iptw_results.json'}")

    return results


def _auc(y_true, y_score):
    """简单AUC计算"""
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_score)
    except:
        return 0.5

def _smd(x, t):
    """标准化均值差"""
    m1, m0 = x[t==1].mean(), x[t==0].mean()
    s1, s0 = x[t==1].std(), x[t==0].std()
    denom = np.sqrt((s1**2 + s0**2) / 2)
    return (m1 - m0) / denom if denom > 0 else 0

def _smd_weighted(x, t, w):
    """加权SMD"""
    m1 = np.average(x[t==1], weights=w[t==1])
    m0 = np.average(x[t==0], weights=w[t==0])
    v1 = np.average((x[t==1] - m1)**2, weights=w[t==1])
    v0 = np.average((x[t==0] - m0)**2, weights=w[t==0])
    denom = np.sqrt((v1 + v0) / 2)
    return (m1 - m0) / denom if denom > 0 else 0

def _bootstrap_ci_iptw(y, t, w, n_boot=1000, alpha=0.05):
    """Bootstrap CI for IPTW"""
    rng = np.random.RandomState(42)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yb, tb, wb = y[idx], t[idx], w[idx]
        if tb.sum() > 0 and (1-tb).sum() > 0:
            m1 = np.average(yb[tb==1], weights=wb[tb==1])
            m0 = np.average(yb[tb==0], weights=wb[tb==0])
            diffs.append(m1 - m0)
    return np.percentile(diffs, 100*alpha/2), np.percentile(diffs, 100*(1-alpha/2))


# ============================================================
# Analysis 2: 输注速率RCS分析（替代总量）
# ============================================================
def analysis_rate_rcs(df):
    """用RFTN_rate_mean替代total_dose做RCS分析"""
    print("\n" + "=" * 70)
    print("Analysis 2: Infusion Rate as Primary Exposure")
    print("=" * 70)

    exposure = 'RFTN_rate_mean'
    outcomes = [
        ('HR_rebound', 'HR rebound'),
        ('MAP_rebound', 'MAP rebound'),
        ('FTN_rescue_mcg_kg', 'FTN rescue'),
        ('OSI', 'OIH Surrogate Index'),
        ('NHD_pct', 'NHD index'),
    ]

    # Table 1 by rate quartile
    print("\n  --- Table 1 by infusion rate quartile ---")
    rate_q = df['rate_quartile']
    for q in ['Q1','Q2','Q3','Q4']:
        sub = df[rate_q == q]
        print(f"  {q}: n={len(sub)}, rate_mean={sub[exposure].mean():.4f} \u00b1 {sub[exposure].std():.4f} "
              f"\u03bcg/kg/min, opdur={sub['opdur'].mean():.0f} min")

    results = {}
    for outcome, label in outcomes:
        data = df[[exposure, outcome, 'opdur', 'age', 'elderly']].dropna()
        if len(data) < 100:
            continue

        print(f"\n  --- Rate \u2192 {label} (n={len(data)}) ---")

        # 简单线性 + 调整手术时间
        from numpy.polynomial import polynomial as P

        # 未调整
        r_raw, p_raw = stats.spearmanr(data[exposure], data[outcome])

        # 偏相关（控制opdur）
        r_partial, p_partial = _partial_corr(data[exposure], data[outcome], data['opdur'])

        print(f"    Spearman (raw): \u03c1 = {r_raw:+.4f}, P = {p_raw:.4e}")
        print(f"    Partial (adj opdur): \u03c1 = {r_partial:+.4f}, P = {p_partial:.4e}")

        # 按速率四分位的结局均值
        q_means = []
        for q in ['Q1','Q2','Q3','Q4']:
            sub = df.loc[df['rate_quartile'] == q, outcome].dropna()
            q_means.append(sub.mean())

        # P for trend
        _, p_trend = stats.spearmanr(range(4), q_means)

        print(f"    Quartile means: {[f'{m:.3f}' for m in q_means]}")
        print(f"    P for trend: {p_trend:.4f}")

        # RCS with 4 knots
        rcs_result = _fit_rcs(data[exposure].values, data[outcome].values, n_knots=4,
                              covariates=data[['opdur', 'age']].values)

        if rcs_result:
            print(f"    RCS: R\u00b2={rcs_result['r2']:.4f}, P_nonlinear={rcs_result['p_nonlinear']:.4f}")

        results[f'rate\u2192{label}'] = {
            'n': len(data),
            'spearman_raw': float(r_raw),
            'p_raw': float(p_raw),
            'partial_r': float(r_partial),
            'p_partial': float(p_partial),
            'quartile_means': [float(m) for m in q_means],
            'p_trend': float(p_trend),
            'rcs': rcs_result
        }

    with open(RESULTS_DIR / "rate_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to {RESULTS_DIR / 'rate_analysis_results.json'}")

    return results


def _partial_corr(x, y, z):
    """偏相关系数（控制z）"""
    from scipy.stats import pearsonr
    # 残差法
    x_arr, y_arr, z_arr = np.array(x), np.array(y), np.array(z)
    # 回归x on z
    z_design = np.column_stack([np.ones(len(z_arr)), z_arr])
    bx = np.linalg.lstsq(z_design, x_arr, rcond=None)[0]
    by = np.linalg.lstsq(z_design, y_arr, rcond=None)[0]
    rx = x_arr - z_design @ bx
    ry = y_arr - z_design @ by
    return pearsonr(rx, ry)


def _fit_rcs(x, y, n_knots=4, covariates=None):
    """Restricted Cubic Splines拟合"""
    try:
        from patsy import dmatrix
        import statsmodels.api as sm

        # 节点位置
        percs = np.linspace(5, 95, n_knots)
        knots = np.percentile(x, percs)
        knot_str = ', '.join([f'{k:.4f}' for k in knots])

        df_data = pd.DataFrame({'x': x, 'y': y})
        if covariates is not None:
            for i in range(covariates.shape[1]):
                df_data[f'cov{i}'] = covariates[:, i]
            cov_terms = ' + '.join([f'cov{i}' for i in range(covariates.shape[1])])
        else:
            cov_terms = ''

        # 线性模型
        if cov_terms:
            linear_formula = f'y ~ x + {cov_terms}'
            spline_formula = f'y ~ cr(x, knots=[{knot_str}]) + {cov_terms}'
        else:
            linear_formula = 'y ~ x'
            spline_formula = f'y ~ cr(x, knots=[{knot_str}])'

        linear_model = sm.OLS.from_formula(linear_formula, df_data).fit()
        spline_model = sm.OLS.from_formula(spline_formula, df_data).fit()

        # LRT
        df_diff = spline_model.df_model - linear_model.df_model
        if df_diff > 0:
            lr_stat = -2 * (linear_model.llf - spline_model.llf)
            p_nonlinear = 1 - stats.chi2.cdf(lr_stat, df_diff)
        else:
            p_nonlinear = 1.0

        return {
            'r2': float(spline_model.rsquared),
            'r2_linear': float(linear_model.rsquared),
            'p_nonlinear': float(p_nonlinear),
            'knots': [float(k) for k in knots]
        }
    except Exception as e:
        print(f"    RCS failed: {e}")
        return None


# ============================================================
# Analysis 3: 高暴露亚群分析
# ============================================================
def analysis_high_rate_subgroup(df):
    """高速率暴露亚群分析（>0.2 \u03bcg/kg/min — 文献OIH高风险阈值）"""
    print("\n" + "=" * 70)
    print("Analysis 3: High-Rate Subgroup Analysis (>0.2 \u03bcg/kg/min)")
    print("=" * 70)

    rate = df['RFTN_rate_mean']
    outcomes = ['HR_rebound', 'MAP_rebound', 'FTN_rescue_mcg_kg', 'NHD_pct', 'OSI']

    # 三组比较: <0.1 vs 0.1-0.2 vs >0.2
    df['rate_group'] = pd.cut(rate, bins=[0, 0.1, 0.2, np.inf],
                               labels=['<0.1', '0.1-0.2', '>0.2'])

    results = {}

    print(f"\n  Group sizes:")
    for g in ['<0.1', '0.1-0.2', '>0.2']:
        n = (df['rate_group'] == g).sum()
        print(f"    {g}: n={n}")

    print(f"\n  {'Outcome':<25s} {'<0.1':>12s} {'0.1-0.2':>12s} {'>0.2':>12s} {'P(ANOVA)':>10s} {'P(trend)':>10s}")

    for outcome in outcomes:
        groups = []
        means = []
        for g in ['<0.1', '0.1-0.2', '>0.2']:
            vals = df.loc[df['rate_group'] == g, outcome].dropna()
            groups.append(vals.values)
            means.append(vals.mean())

        # ANOVA
        if all(len(g) > 10 for g in groups):
            f_stat, p_anova = stats.f_oneway(*groups)
            _, p_trend = stats.spearmanr([0, 1, 2], means)

            print(f"  {outcome:<25s} {means[0]:>+12.3f} {means[1]:>+12.3f} {means[2]:>+12.3f} "
                  f"{p_anova:>10.4f} {p_trend:>10.4f}")

            results[outcome] = {
                'means': [float(m) for m in means],
                'ns': [len(g) for g in groups],
                'p_anova': float(p_anova),
                'p_trend': float(p_trend)
            }

    # 高速率组内部的剂量-反应（如果样本量允许）
    high_rate = df[df['RFTN_rate_mean'] > 0.2].copy()
    print(f"\n  Within high-rate group (n={len(high_rate)}):")

    if len(high_rate) > 50:
        for outcome in ['HR_rebound', 'MAP_rebound', 'OSI']:
            sub = high_rate[['RFTN_rate_mean', outcome]].dropna()
            if len(sub) > 20:
                r, p = stats.spearmanr(sub['RFTN_rate_mean'], sub[outcome])
                print(f"    Rate \u2192 {outcome}: \u03c1={r:+.3f}, P={p:.4f} (n={len(sub)})")

    # 年龄分层的高速率效应
    print(f"\n  Age-stratified high-rate effects:")
    for age_label, age_mask in [('<65', df['age'] < 65), ('\u226565', df['age'] >= 65)]:
        sub = df[age_mask]
        for outcome in ['HR_rebound', 'OSI']:
            g_low = sub.loc[sub['rate_group'] == '<0.1', outcome].dropna()
            g_high = sub.loc[sub['rate_group'] == '>0.2', outcome].dropna()
            if len(g_low) > 10 and len(g_high) > 5:
                t_stat, p_val = stats.ttest_ind(g_low, g_high)
                diff = g_high.mean() - g_low.mean()
                print(f"    {age_label}, {outcome}: \u0394(>0.2 vs <0.1) = {diff:+.3f}, P={p_val:.4f}")

    results['_groups'] = ['<0.1', '0.1-0.2', '>0.2']
    with open(RESULTS_DIR / "high_rate_subgroup.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {RESULTS_DIR / 'high_rate_subgroup.json'}")

    return results


# ============================================================
# Analysis 4: 停药动力学分析
# ============================================================
def analysis_taper_dynamics(df):
    """停药动力学分析 — taper slope与反弹的关系"""
    print("\n" + "=" * 70)
    print("Analysis 4: Taper Dynamics Analysis")
    print("=" * 70)

    # taper_slope: 负值=逐渐减量, 正值=突然停药/增量
    # Ce_at_end: 停药时残余浓度

    taper_features = ['RFTN_taper_slope', 'RFTN_Ce_at_end', 'RFTN_Ce_SD', 'RFTN_Ce_CV']
    outcomes = ['HR_rebound', 'MAP_rebound', 'OSI']

    results = {}

    # taper_slope 与结局的相关性
    print(f"\n  Taper feature correlations with outcomes:")
    print(f"  {'Feature':<25s} {'Outcome':<20s} {'\u03c1':>8s} {'P':>12s}")

    for feat in taper_features:
        for outcome in outcomes:
            sub = df[[feat, outcome]].dropna()
            if len(sub) > 50:
                r, p = stats.spearmanr(sub[feat], sub[outcome])
                sig = " **" if p < 0.01 else " *" if p < 0.05 else ""
                print(f"  {feat:<25s} {outcome:<20s} {r:>+8.4f} {p:>12.4e}{sig}")
                results[f'{feat}\u2192{outcome}'] = {'rho': float(r), 'p': float(p), 'n': len(sub)}

    # 快速停药 vs 缓慢减量
    ts = df['RFTN_taper_slope'].dropna()
    ts_med = ts.median()
    df['taper_type'] = np.where(df['RFTN_taper_slope'] < ts_med, 'gradual', 'abrupt')

    print(f"\n  Taper type comparison (median split at {ts_med:.4f}):")
    print(f"  {'Outcome':<25s} {'Gradual':>12s} {'Abrupt':>12s} {'P':>10s}")

    for outcome in outcomes:
        g1 = df.loc[df['taper_type'] == 'gradual', outcome].dropna()
        g2 = df.loc[df['taper_type'] == 'abrupt', outcome].dropna()
        if len(g1) > 30 and len(g2) > 30:
            _, p = stats.ttest_ind(g1, g2)
            print(f"  {outcome:<25s} {g1.mean():>+12.3f} {g2.mean():>+12.3f} {p:>10.4f}")
            results[f'taper_type\u2192{outcome}'] = {
                'gradual_mean': float(g1.mean()), 'abrupt_mean': float(g2.mean()),
                'p': float(p), 'n_gradual': len(g1), 'n_abrupt': len(g2)
            }

    # 剂量\u00d7停药方式交互
    print(f"\n  Dose \u00d7 Taper interaction:")
    for outcome in ['HR_rebound', 'OSI']:
        sub = df[['RFTN_rate_mean', 'RFTN_taper_slope', outcome, 'opdur', 'age']].dropna()
        if len(sub) > 200:
            import statsmodels.api as sm
            sub['interaction'] = sub['RFTN_rate_mean'] * sub['RFTN_taper_slope']
            X = sm.add_constant(sub[['RFTN_rate_mean', 'RFTN_taper_slope', 'interaction', 'opdur', 'age']])
            model = sm.OLS(sub[outcome], X).fit()
            int_coef = model.params['interaction']
            int_p = model.pvalues['interaction']
            print(f"    Rate \u00d7 Taper \u2192 {outcome}: \u03b2={int_coef:+.4f}, P={int_p:.4f}")
            results[f'interaction_rate_taper\u2192{outcome}'] = {
                'beta': float(int_coef), 'p': float(int_p), 'r2': float(model.rsquared)
            }

    with open(RESULTS_DIR / "taper_dynamics.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {RESULTS_DIR / 'taper_dynamics.json'}")

    return results


# ============================================================
# Analysis 5: 中介分析（NHD路径）
# ============================================================
def analysis_mediation(df):
    """中介分析: RFTN \u2192 NHD \u2192 血流动力学反弹"""
    print("\n" + "=" * 70)
    print("Analysis 5: Mediation Analysis (NHD Pathway)")
    print("=" * 70)

    # 路径: RFTN_rate_mean \u2192 NHD_pct \u2192 HR_rebound
    # 以及: RFTN_rate_mean \u2192 BIS波动 \u2192 HR_rebound

    import statsmodels.api as sm

    mediators = [
        ('NHD_pct', 'NHD index'),
        ('CV_BIS', 'BIS variability'),
    ]

    exposure = 'RFTN_rate_mean'
    outcome = 'HR_rebound'
    covariates = ['opdur', 'age']

    results = {}

    for mediator, med_label in mediators:
        needed = [exposure, mediator, outcome] + covariates
        sub = df[needed].dropna()

        if len(sub) < 100:
            print(f"\n  [{med_label}] Insufficient data (n={len(sub)})")
            continue

        print(f"\n  --- Mediation via {med_label} (n={len(sub)}) ---")

        # Step 1: Total effect (c path)
        X_total = sm.add_constant(sub[[exposure] + covariates])
        total_model = sm.OLS(sub[outcome], X_total).fit()
        c_total = total_model.params[exposure]
        c_p = total_model.pvalues[exposure]

        # Step 2: a path (exposure \u2192 mediator)
        X_a = sm.add_constant(sub[[exposure] + covariates])
        a_model = sm.OLS(sub[mediator], X_a).fit()
        a_coef = a_model.params[exposure]
        a_p = a_model.pvalues[exposure]

        # Step 3: b path + c' (direct effect)
        X_med = sm.add_constant(sub[[exposure, mediator] + covariates])
        med_model = sm.OLS(sub[outcome], X_med).fit()
        b_coef = med_model.params[mediator]
        b_p = med_model.pvalues[mediator]
        c_direct = med_model.params[exposure]
        c_direct_p = med_model.pvalues[exposure]

        # Indirect effect (a \u00d7 b)
        indirect = a_coef * b_coef

        # Proportion mediated
        if abs(c_total) > 1e-10:
            prop_mediated = indirect / c_total
        else:
            prop_mediated = 0

        # Sobel test
        a_se = a_model.bse[exposure]
        b_se = med_model.bse[mediator]
        sobel_se = np.sqrt(a_coef**2 * b_se**2 + b_coef**2 * a_se**2)
        sobel_z = indirect / sobel_se if sobel_se > 0 else 0
        sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

        print(f"    c (total):    \u03b2 = {c_total:+.4f}, P = {c_p:.4f}")
        print(f"    a (X\u2192M):      \u03b2 = {a_coef:+.4f}, P = {a_p:.4f}")
        print(f"    b (M\u2192Y|X):    \u03b2 = {b_coef:+.4f}, P = {b_p:.4f}")
        print(f"    c' (direct):  \u03b2 = {c_direct:+.4f}, P = {c_direct_p:.4f}")
        print(f"    a\u00d7b (indirect): {indirect:+.4f}")
        print(f"    Sobel test:   Z = {sobel_z:.3f}, P = {sobel_p:.4f}")
        print(f"    % mediated:   {prop_mediated*100:.1f}%")

        results[med_label] = {
            'n': len(sub),
            'c_total': float(c_total), 'c_total_p': float(c_p),
            'a_path': float(a_coef), 'a_p': float(a_p),
            'b_path': float(b_coef), 'b_p': float(b_p),
            'c_direct': float(c_direct), 'c_direct_p': float(c_direct_p),
            'indirect_effect': float(indirect),
            'sobel_z': float(sobel_z), 'sobel_p': float(sobel_p),
            'proportion_mediated': float(prop_mediated)
        }

    with open(RESULTS_DIR / "mediation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {RESULTS_DIR / 'mediation_results.json'}")

    return results


# ============================================================
# Analysis 6: 增强预测模型
# ============================================================
def analysis_enhanced_prediction(df):
    """增强预测模型 — 加入交互特征和时间特征"""
    print("\n" + "=" * 70)
    print("Analysis 6: Enhanced Prediction Model")
    print("=" * 70)

    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LassoCV

    exposure_feats = ['RFTN_rate_mean', 'RFTN_rate_peak', 'RFTN_Ce_mean', 'RFTN_Ce_peak',
                      'RFTN_Ce_CV', 'RFTN_Ce_ARV', 'RFTN_taper_slope',
                      'Time_Ce_above_4', 'Time_Ce_above_6']

    clinical_feats = ['age', 'bmi', 'asa', 'emop', 'opdur', 'intraop_ppf']

    bis_feats = ['TWA_BIS', 'CV_BIS', 'ARV_BIS']

    base_feats = exposure_feats + clinical_feats + bis_feats

    # 构建交互特征
    print("  Building interaction features...")
    df_feat = df.copy()

    interaction_pairs = [
        ('RFTN_rate_mean', 'opdur'),
        ('RFTN_rate_mean', 'age'),
        ('RFTN_Ce_peak', 'RFTN_taper_slope'),
        ('RFTN_Ce_mean', 'CV_BIS'),
        ('RFTN_rate_mean', 'RFTN_Ce_CV'),
    ]

    interaction_feats = []
    for f1, f2 in interaction_pairs:
        if f1 in df.columns and f2 in df.columns:
            fname = f'{f1}_x_{f2}'
            df_feat[fname] = df[f1] * df[f2]
            interaction_feats.append(fname)

    # 非线性特征
    nonlinear_feats = []
    for f in ['RFTN_rate_mean', 'RFTN_Ce_mean', 'age']:
        if f in df.columns:
            fname = f'{f}_sq'
            df_feat[fname] = df[f] ** 2
            nonlinear_feats.append(fname)

    all_feats = base_feats + interaction_feats + nonlinear_feats
    all_feats = [f for f in all_feats if f in df_feat.columns]

    results = {}

    for target in ['HR_rebound', 'OSI']:
        needed = all_feats + [target]
        sub = df_feat[needed].dropna()

        if len(sub) < 200:
            continue

        X = sub[all_feats].values
        y = sub[target].values

        print(f"\n  Target: {target} (n={len(sub)}, features={len(all_feats)})")

        # Base model (原始特征)
        base_only = [f for f in base_feats if f in df_feat.columns]
        X_base = sub[base_only].values

        pipe_base = Pipeline([('scaler', StandardScaler()), ('lasso', LassoCV(cv=5, max_iter=10000))])
        scores_base = cross_val_score(pipe_base, X_base, y, cv=10, scoring='r2')

        # Enhanced model (全部特征)
        pipe_enh = Pipeline([('scaler', StandardScaler()), ('lasso', LassoCV(cv=5, max_iter=10000))])
        scores_enh = cross_val_score(pipe_enh, X, y, cv=10, scoring='r2')

        print(f"    Base model R\u00b2:     {scores_base.mean():.4f} \u00b1 {scores_base.std():.4f}")
        print(f"    Enhanced model R\u00b2: {scores_enh.mean():.4f} \u00b1 {scores_enh.std():.4f}")
        print(f"    Improvement: {(scores_enh.mean() - scores_base.mean()):.4f}")

        # 拟合最终模型获取特征重要性
        pipe_enh.fit(X, y)
        lasso = pipe_enh.named_steps['lasso']
        coefs = pd.Series(np.abs(lasso.coef_), index=all_feats)
        top_feats = coefs.nlargest(10)

        print(f"    Top 10 features:")
        for fname, importance in top_feats.items():
            if importance > 0:
                print(f"      {fname:<35s}: {importance:.4f}")

        # GBM with interactions
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            gbm = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                             learning_rate=0.05, subsample=0.8,
                                             random_state=42)
            scores_gbm = cross_val_score(gbm, X, y, cv=10, scoring='r2')
            print(f"    GBM model R\u00b2:      {scores_gbm.mean():.4f} \u00b1 {scores_gbm.std():.4f}")
        except:
            scores_gbm = scores_base

        results[target] = {
            'n': len(sub),
            'n_features': len(all_feats),
            'base_r2': float(scores_base.mean()),
            'enhanced_r2': float(scores_enh.mean()),
            'gbm_r2': float(scores_gbm.mean()),
            'top_features': {k: float(v) for k, v in top_feats.items() if v > 0}
        }

    with open(RESULTS_DIR / "enhanced_prediction.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {RESULTS_DIR / 'enhanced_prediction.json'}")

    return results


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  OIH Study - Phase 2b: Extended Analysis")
    print(f"  Date: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    df = load_clean_data()

    r1 = analysis_iptw(df)
    r2 = analysis_rate_rcs(df)
    r3 = analysis_high_rate_subgroup(df)
    r4 = analysis_taper_dynamics(df)
    r5 = analysis_mediation(df)
    r6 = analysis_enhanced_prediction(df)

    print("\n" + "=" * 70)
    print("  Phase 2b Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
