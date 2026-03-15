#!/usr/bin/env python3
"""
=================================================================
OIH Study - Phase 3: Visualization & Figure Generation
=================================================================
围术期瑞芬太尼诱导痛觉过敏（OIH）数据挖掘
出版级别可视化Pipeline

功能：
1. Figure 1: STROBE流程图
2. Figure 2: 瑞芬太尼暴露分布（多面板）
3. Figure 3: 剂量-反应曲线（核心图 ★）
4. Figure 4: E_max模型拟合
5. Figure 5: 输注模式聚类
6. Figure 6: 森林图（亚组分析）
7. Figure 7: SHAP特征重要性
8. Supplementary Figures
=================================================================
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
except ImportError:
    pass

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
OIH_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 配色方案
COLORS = {
    'young': '#2196F3',      # 蓝色 - <65
    'mid_elderly': '#FF9800', # 橙色 - 65-74
    'old_elderly': '#F44336', # 红色 - >=75
    'overall': '#424242',     # 深灰
    'ci': '#E3F2FD',          # 浅蓝 CI
    'threshold': '#E91E63',   # 粉红 - 阈值线
    'q1': '#4CAF50', 'q2': '#8BC34A', 'q3': '#FF9800', 'q4': '#F44336',
}

DPI = 300
FONT_FAMILY = 'DejaVu Sans'


# ============================================================
# Figure 2: 瑞芬太尼暴露分布
# ============================================================
def fig2_rftn_exposure_distribution(df):
    """Figure 2: 瑞芬太尼暴露分布（多面板）"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: 总剂量直方图
    ax = axes[0, 0]
    if 'RFTN_total_mcg_kg' in df.columns:
        data = df['RFTN_total_mcg_kg'].dropna()
        ax.hist(data, bins=50, color=COLORS['overall'], alpha=0.7, edgecolor='white')
        ax.axvline(data.median(), color=COLORS['threshold'], linestyle='--', linewidth=2,
                   label=f'Median: {data.median():.0f}')
        ax.set_xlabel('Remifentanil total dose (\u03bcg/kg)')
        ax.set_ylabel('Count')
        ax.set_title('A. Distribution of remifentanil exposure')
        ax.legend()

    # Panel B: 按年龄组的箱线图
    ax = axes[0, 1]
    if 'RFTN_total_mcg_kg' in df.columns and 'age_group' in df.columns:
        age_groups = ['<65', '65-74', '>=75']
        group_colors = [COLORS['young'], COLORS['mid_elderly'], COLORS['old_elderly']]

        box_data = [df.loc[df['age_group'] == g, 'RFTN_total_mcg_kg'].dropna()
                    for g in age_groups]
        box_data = [d for d in box_data if len(d) > 0]

        bp = ax.boxplot(box_data, patch_artist=True, labels=age_groups[:len(box_data)])
        for patch, color in zip(bp['boxes'], group_colors[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xlabel('Age group')
        ax.set_ylabel('Remifentanil total dose (\u03bcg/kg)')
        ax.set_title('B. Dose by age group')

    # Panel C: 剂量 vs 手术时间散点图
    ax = axes[1, 0]
    if 'RFTN_total_mcg_kg' in df.columns and 'opdur' in df.columns:
        scatter_data = df[['RFTN_total_mcg_kg', 'opdur', 'age']].dropna()
        if len(scatter_data) > 0:
            sc = ax.scatter(scatter_data['opdur'], scatter_data['RFTN_total_mcg_kg'],
                          c=scatter_data['age'], cmap='coolwarm', alpha=0.3, s=10)
            plt.colorbar(sc, ax=ax, label='Age (years)')
            ax.set_xlabel('Surgery duration (min)')
            ax.set_ylabel('Remifentanil total dose (\u03bcg/kg)')
            ax.set_title('C. Dose vs surgery duration')

    # Panel D: 暴露指标相关矩阵
    ax = axes[1, 1]
    corr_vars = ['RFTN_total_mcg_kg', 'RFTN_Ce_mean', 'RFTN_Ce_peak',
                 'RFTN_rate_mean', 'RFTN_Ce_CV', 'RFTN_AUC_Ce']
    available_corr = [v for v in corr_vars if v in df.columns]
    if len(available_corr) >= 3:
        corr_matrix = df[available_corr].corr(method='spearman')
        short_labels = [v.replace('RFTN_', '').replace('_mcg_kg', '') for v in available_corr]
        im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(short_labels)))
        ax.set_yticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(short_labels, fontsize=8)
        for i in range(len(short_labels)):
            for j in range(len(short_labels)):
                ax.text(j, i, f'{corr_matrix.values[i,j]:.2f}',
                       ha='center', va='center', fontsize=7)
        plt.colorbar(im, ax=ax, label='Spearman \u03c1')
        ax.set_title('D. Exposure metrics correlation')

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_rftn_exposure.png", dpi=DPI, bbox_inches='tight')
    plt.savefig(FIG_DIR / "fig2_rftn_exposure.pdf", bbox_inches='tight')
    plt.close()
    print("  Saved Figure 2: RFTN exposure distribution")


# ============================================================
# Figure 3: 剂量-反应曲线（核心图）
# ============================================================
def fig3_dose_response_curve(df):
    """Figure 3: RCS剂量-反应曲线（核心图 ★）"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    exposure = 'RFTN_total_mcg_kg'
    outcome = 'HR_rebound'
    x_max = df[exposure].dropna().quantile(0.99) * 1.05  # P99 上限

    # 尝试加载预计算的RCS结果
    rcs_file = RESULTS_DIR / f"rcs_{exposure}_{outcome}.json"

    # Panel A: 全人群RCS + 分箱均值叠加
    ax = axes[0]
    if rcs_file.exists():
        with open(rcs_file) as f:
            rcs = json.load(f)

        x = np.array(rcs['curve_data']['x'])
        y = np.array(rcs['curve_data']['y'])
        y_lo = np.array(rcs['curve_data']['y_lower'])
        y_hi = np.array(rcs['curve_data']['y_upper'])

        # 限制在X范围内
        mask_x = x <= x_max
        ax.fill_between(x[mask_x], y_lo[mask_x], y_hi[mask_x],
                        alpha=0.15, color=COLORS['overall'])
        ax.plot(x[mask_x], y[mask_x], color=COLORS['overall'], linewidth=2.5,
                label='RCS fit', zorder=5)

        # 分箱散点叠加
        if exposure in df.columns and outcome in df.columns:
            raw = df[[exposure, outcome]].dropna()
            raw = raw[raw[exposure] <= x_max]
            raw['bin'] = pd.qcut(raw[exposure], 20, duplicates='drop')
            binned = raw.groupby('bin').agg({exposure: 'mean', outcome: ['mean', 'sem']})
            binned.columns = ['bx', 'by', 'bse']
            ax.errorbar(binned['bx'], binned['by'], yerr=1.96*binned['bse'],
                        fmt='o', color='gray', markersize=3, capsize=2, alpha=0.5, zorder=3)

        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        # Rug plot
        if exposure in df.columns:
            data = df[exposure].dropna()
            data = data[data <= x_max]
            ax.plot(data.values, np.full(len(data), ax.get_ylim()[0]),
                   '|', color='gray', alpha=0.05, markersize=3)

        p_nl = rcs['p_nonlinear']
        ax.set_title(f'A. Overall dose-response\n(P$_{{nonlinear}}$ = {p_nl:.3f})')

    else:
        if exposure in df.columns and outcome in df.columns:
            data = df[[exposure, outcome]].dropna()
            data = data[data[exposure] <= x_max]
            if len(data) > 50:
                try:
                    import statsmodels.api as sm
                    lowess = sm.nonparametric.lowess(data[outcome], data[exposure], frac=0.3)
                    ax.plot(lowess[:, 0], lowess[:, 1], color=COLORS['overall'],
                           linewidth=2.5, label='LOWESS')
                except ImportError:
                    pass
                ax.scatter(data[exposure], data[outcome], alpha=0.03, s=3, color='gray')
                ax.set_title('A. Overall dose-response')

    ax.set_xlabel('Remifentanil total dose (\u03bcg/kg)')
    ax.set_ylabel('HR rebound (bpm)')
    ax.set_xlim(0, x_max)
    ax.legend(loc='lower left', fontsize=9)

    # Panel B: 年龄分层LOWESS（含CI带）
    ax = axes[1]
    if exposure in df.columns and outcome in df.columns and 'age_group' in df.columns:
        age_configs = [
            ('<65', COLORS['young'], 'solid'),
            ('65-74', COLORS['mid_elderly'], 'dashed'),
            ('>=75', COLORS['old_elderly'], 'dotted'),
        ]

        for group, color, ls in age_configs:
            sub = df.loc[df['age_group'] == group, [exposure, outcome]].dropna()
            sub = sub[sub[exposure] <= x_max]
            if len(sub) < 30:
                continue

            # 分箱均值 + SEM
            sub_copy = sub.copy()
            sub_copy['bin'] = pd.qcut(sub_copy[exposure], 12, duplicates='drop')
            binned = sub_copy.groupby('bin').agg({exposure: 'mean', outcome: ['mean', 'sem']})
            binned.columns = ['bx', 'by', 'bse']
            ax.errorbar(binned['bx'], binned['by'], yerr=1.96*binned['bse'],
                       fmt='o-', color=color, markersize=4, capsize=2,
                       linewidth=1.5, linestyle=ls, alpha=0.8,
                       label=f'{group} (n={len(sub)})')

        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Remifentanil total dose (\u03bcg/kg)')
        ax.set_ylabel('HR rebound (bpm)')
        ax.set_xlim(0, x_max)
        ax.set_title('B. Age-stratified dose-response')
        ax.legend(fontsize=9)

    # Panel C: 等高线图（年龄\u00d7剂量\u2192HR rebound）
    ax = axes[2]
    if exposure in df.columns and outcome in df.columns and 'age' in df.columns:
        data = df[[exposure, outcome, 'age']].dropna()
        data = data[data[exposure] <= x_max]
        if len(data) > 100:
            from scipy.interpolate import griddata

            # 先做分箱平滑减少噪声
            n_xbins, n_ybins = 20, 15
            data['xbin'] = pd.qcut(data[exposure], n_xbins, duplicates='drop')
            data['ybin'] = pd.cut(data['age'], n_ybins)
            smooth = data.groupby(['xbin', 'ybin']).agg(
                {exposure: 'mean', 'age': 'mean', outcome: 'mean'}
            ).dropna()

            xi = np.linspace(data[exposure].quantile(0.02), x_max, 60)
            yi = np.linspace(data['age'].quantile(0.02), data['age'].quantile(0.98), 60)
            xi, yi = np.meshgrid(xi, yi)

            try:
                zi = griddata(
                    (smooth[exposure].values, smooth['age'].values),
                    smooth[outcome].values,
                    (xi, yi), method='cubic'
                )

                contour = ax.contourf(xi, yi, zi, levels=12, cmap='RdYlBu_r', alpha=0.85)
                plt.colorbar(contour, ax=ax, label='HR rebound (bpm)', shrink=0.85)
                ax.set_xlabel('Remifentanil total dose (\u03bcg/kg)')
                ax.set_ylabel('Age (years)')
                ax.set_title('C. Dose \u00d7 Age interaction surface')

                # 标注零等高线
                try:
                    cs = ax.contour(xi, yi, zi, levels=[0], colors='black',
                                    linewidths=2, linestyles='--')
                    ax.clabel(cs, fmt='0', fontsize=9)
                except Exception:
                    pass

            except Exception:
                ax.text(0.5, 0.5, 'Insufficient data\nfor contour plot',
                       ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_dose_response.png", dpi=DPI, bbox_inches='tight')
    plt.savefig(FIG_DIR / "fig3_dose_response.pdf", bbox_inches='tight')
    plt.close()
    print("  Saved Figure 3: Dose-response curves (CORE FIGURE)")


# ============================================================
# Figure 4: E_max模型拟合
# ============================================================
def fig4_emax_model(df):
    """Figure 4: E_max模型拟合（改用分箱数据 + 限制X轴范围）"""
    from scipy.optimize import curve_fit

    def emax_func(dose, e0, emax, ed50, n):
        return e0 + (emax * np.power(dose, n)) / (np.power(ed50, n) + np.power(dose, n))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    exposure = 'RFTN_total_mcg_kg'
    outcome = 'HR_rebound'
    x_max = df[exposure].dropna().quantile(0.99) * 1.05

    if exposure not in df.columns or outcome not in df.columns:
        plt.close()
        return

    # Panel A: 全人群 — 用分箱均值拟合（更稳健）
    ax = axes[0]
    data = df[[exposure, outcome]].dropna()
    data = data[data[exposure] <= x_max]

    if len(data) > 50:
        # 分箱
        n_bins = 25
        data_s = data.sort_values(exposure).copy()
        data_s['bin'] = pd.qcut(data_s[exposure], n_bins, duplicates='drop')
        binned = data_s.groupby('bin').agg({
            exposure: 'mean', outcome: ['mean', 'sem', 'count']
        })
        binned.columns = ['x_mean', 'y_mean', 'y_sem', 'n']

        ax.errorbar(binned['x_mean'], binned['y_mean'],
                   yerr=1.96 * binned['y_sem'], fmt='o',
                   color=COLORS['overall'], markersize=5, capsize=3, alpha=0.7)

        # 用分箱均值加权拟合
        bx = binned['x_mean'].values
        by = binned['y_mean'].values
        bw = np.sqrt(binned['n'].values)

        try:
            p0 = [by.max(), by.min() - by.max(), np.median(bx), 1.5]
            bounds = ([-np.inf, -np.inf, 0.1, 0.3], [np.inf, 0, x_max*2, 10])
            popt, pcov = curve_fit(emax_func, bx, by, p0=p0, bounds=bounds,
                                   sigma=1/bw, maxfev=20000)

            x_fit = np.linspace(bx.min(), bx.max(), 200)
            y_fit = emax_func(x_fit, *popt)
            ax.plot(x_fit, y_fit, color=COLORS['threshold'], linewidth=2.5,
                   label=f'E$_{{max}}$ fit\nED$_{{50}}$={popt[2]:.1f} \u03bcg/kg\nHill n={popt[3]:.2f}')

            if popt[2] < x_max:
                ax.axvline(popt[2], color=COLORS['threshold'], linestyle=':', alpha=0.4)
            else:
                ax.text(0.95, 0.05, f'ED\u2085\u2080={popt[2]:.0f}\n(beyond range)',
                       transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
                       color=COLORS['threshold'], style='italic')
        except Exception as e:
            ax.text(0.5, 0.1, f'E_max fit failed:\n{str(e)[:50]}',
                   transform=ax.transAxes, fontsize=8, ha='center', color='red')

        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Remifentanil total dose (\u03bcg/kg)')
        ax.set_ylabel('HR rebound (bpm)')
        ax.set_xlim(0, x_max)
        ax.set_title('A. Overall E$_{max}$ model')
        ax.legend(fontsize=9, loc='lower left')

    # Panel B: 年龄分层
    ax = axes[1]
    configs = [
        ('<65', df[df['age'] < 65], COLORS['young']),
        ('65-74', df[(df['age'] >= 65) & (df['age'] < 75)], COLORS['mid_elderly']),
        ('\u226575', df[df['age'] >= 75], COLORS['old_elderly']),
    ]

    for label, sub_df, color in configs:
        sub = sub_df[[exposure, outcome]].dropna()
        sub = sub[sub[exposure] <= x_max]
        if len(sub) < 30:
            continue

        # 分箱
        sub_s = sub.sort_values(exposure).copy()
        sub_s['bin'] = pd.qcut(sub_s[exposure], min(15, len(sub_s)//10), duplicates='drop')
        binned = sub_s.groupby('bin').agg({
            exposure: 'mean', outcome: ['mean', 'sem', 'count']
        })
        binned.columns = ['bx', 'by', 'bse', 'bn']

        ax.errorbar(binned['bx'], binned['by'], yerr=1.96*binned['bse'],
                   fmt='o-', color=color, markersize=4, capsize=2, linewidth=1.5,
                   alpha=0.8, label=f'{label} (n={len(sub)})')

        # E_max拟合
        try:
            bx = binned['bx'].values
            by = binned['by'].values
            bw = np.sqrt(binned['bn'].values)
            p0 = [by.max(), by.min()-by.max(), np.median(bx), 1.5]
            bounds = ([-np.inf, -np.inf, 0.1, 0.3], [np.inf, 0, x_max*2, 10])
            popt, _ = curve_fit(emax_func, bx, by, p0=p0, bounds=bounds,
                               sigma=1/bw, maxfev=20000)
            x_fit = np.linspace(bx.min(), bx.max(), 200)
            y_fit = emax_func(x_fit, *popt)
            ax.plot(x_fit, y_fit, color=color, linewidth=2, linestyle='--', alpha=0.6)
        except Exception:
            pass

    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Remifentanil total dose (\u03bcg/kg)')
    ax.set_ylabel('HR rebound (bpm)')
    ax.set_xlim(0, x_max)
    ax.set_title('B. Age-stratified E$_{max}$ models')
    ax.legend(fontsize=9, loc='lower left')

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_emax_model.png", dpi=DPI, bbox_inches='tight')
    plt.savefig(FIG_DIR / "fig4_emax_model.pdf", bbox_inches='tight')
    plt.close()
    print("  Saved Figure 4: E_max model")


# ============================================================
# Figure 5: OIH替代终点与剂量的关系（多终点面板）
# ============================================================
def fig5_multi_outcome_panel(df):
    """Figure 5: 多个OIH替代终点 vs 瑞芬太尼剂量"""
    exposure = 'RFTN_total_mcg_kg'
    x_max = df[exposure].dropna().quantile(0.99) * 1.05
    outcomes = [
        ('HR_rebound', 'HR rebound (bpm)', '#1976D2'),
        ('MAP_rebound', 'MAP rebound (mmHg)', '#D32F2F'),
        ('FTN_rescue_mcg_kg', 'Fentanyl rescue (\u03bcg/kg)', '#388E3C'),
        ('NHD_pct', 'NHD index (%)', '#7B1FA2'),
    ]
    available = [(o, l, c) for o, l, c in outcomes if o in df.columns]

    if not available or exposure not in df.columns:
        return

    n_panels = len(available)
    fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, (outcome, label, color) in zip(axes, available):
        data = df[[exposure, outcome]].dropna()
        data = data[data[exposure] <= x_max]
        if len(data) < 30:
            continue

        # 分箱均值
        n_bins = 15
        data_c = data.copy()
        data_c['bin'] = pd.qcut(data_c[exposure], n_bins, duplicates='drop')
        binned = data_c.groupby('bin').agg({
            exposure: 'mean', outcome: ['mean', 'sem']
        })
        binned.columns = ['x', 'y', 'sem']

        ax.errorbar(binned['x'], binned['y'], yerr=1.96*binned['sem'],
                   fmt='o-', color=color, markersize=5, capsize=3, linewidth=1.5)

        # LOWESS 平滑线
        try:
            import statsmodels.api as sm
            lowess = sm.nonparametric.lowess(data[outcome], data[exposure], frac=0.3)
            lx, ly = lowess[:, 0], lowess[:, 1]
            mask_lx = lx <= x_max
            ax.plot(lx[mask_lx], ly[mask_lx], color=color, linewidth=2.5, alpha=0.6)
        except ImportError:
            pass

        if outcome in ['HR_rebound', 'MAP_rebound']:
            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Remifentanil dose (\u03bcg/kg)')
        ax.set_ylabel(label)
        ax.set_xlim(0, x_max)
        ax.set_title(label.split('(')[0].strip())

        # Spearman相关 + RCS P值
        from scipy.stats import spearmanr
        rho, p = spearmanr(data[exposure], data[outcome])

        # 加载对应RCS非线性P值
        rcs_file = RESULTS_DIR / f"rcs_{exposure}_{outcome}.json"
        p_nl_str = ''
        if rcs_file.exists():
            try:
                with open(rcs_file) as f:
                    rcs_data = json.load(f)
                p_nl = rcs_data.get('p_nonlinear', None)
                if p_nl is not None:
                    p_nl_str = f'\nP_nl={"<0.001" if p_nl<0.001 else f"{p_nl:.3f}"}'
            except Exception:
                pass

        ax.text(0.05, 0.95,
                f'\u03c1={rho:.3f}\nP={"<0.001" if p<0.001 else f"{p:.3f}"}{p_nl_str}',
               transform=ax.transAxes, fontsize=9, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_multi_outcome.png", dpi=DPI, bbox_inches='tight')
    plt.savefig(FIG_DIR / "fig5_multi_outcome.pdf", bbox_inches='tight')
    plt.close()
    print("  Saved Figure 5: Multi-outcome panel")


# ============================================================
# Figure 6: 年龄分层效应（森林图风格）
# ============================================================
def fig6_age_stratified_effects(df):
    """Figure 6: 年龄分层的剂量-反应效应大小"""
    exposure = 'RFTN_total_mcg_kg'
    outcome = 'HR_rebound'

    if exposure not in df.columns or outcome not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # 定义亚组
    subgroups = [
        ('Overall', df),
        ('', None),  # spacer
        ('Age <65', df[df['age'] < 65]),
        ('Age 65-74', df[(df['age'] >= 65) & (df['age'] < 75)]),
        ('Age >=75', df[df['age'] >= 75]),
        ('', None),
        ('Male', df[df['sex'] == 'M']),
        ('Female', df[df['sex'] == 'F']),
        ('', None),
        ('ASA 1-2', df[df['asa'] <= 2]),
        ('ASA 3-4', df[df['asa'] >= 3]),
        ('', None),
        ('Short surgery (<120min)', df[df['opdur'] < 120]),
        ('Long surgery (>=120min)', df[df['opdur'] >= 120]),
    ]

    y_pos = []
    labels = []
    effects = []
    cis_low = []
    cis_high = []
    ns = []

    pos = 0
    for label, sub_df in subgroups:
        if sub_df is None:
            pos -= 1
            continue

        data = sub_df[[exposure, outcome]].dropna()
        if len(data) < 20:
            continue

        # 简单线性回归斜率
        from scipy.stats import linregress
        slope, intercept, r, p, se = linregress(data[exposure], data[outcome])

        y_pos.append(pos)
        labels.append(f'{label} (n={len(data)})')
        effects.append(slope)
        cis_low.append(slope - 1.96 * se)
        cis_high.append(slope + 1.96 * se)
        ns.append(len(data))
        pos -= 1

    # 绘制森林图
    for i, (y, eff, lo, hi, label) in enumerate(zip(y_pos, effects, cis_low, cis_high, labels)):
        color = COLORS['overall'] if 'Overall' in label else '#2196F3'
        if 'Age' in label:
            if '<65' in label:
                color = COLORS['young']
            elif '65-74' in label:
                color = COLORS['mid_elderly']
            elif '>=75' in label:
                color = COLORS['old_elderly']

        size = 10 if 'Overall' in label else 8
        ax.plot(eff, y, 'D' if 'Overall' in label else 'o',
               color=color, markersize=size, zorder=5)
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, zorder=4)

    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f'Effect of {exposure} on {outcome}\n(\u03b2 per unit increase, 95% CI)')
    ax.set_title('Subgroup Analysis: Effect of Remifentanil Dose on HR Rebound')

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_forest_plot.png", dpi=DPI, bbox_inches='tight')
    plt.savefig(FIG_DIR / "fig6_forest_plot.pdf", bbox_inches='tight')
    plt.close()
    print("  Saved Figure 6: Forest plot")


# ============================================================
# Supplementary: 相关矩阵热图
# ============================================================
def sup_correlation_heatmap(df):
    """Supplementary: OIH指标与暴露指标的完整相关矩阵"""
    exposure_vars = ['RFTN_total_mcg_kg', 'RFTN_Ce_mean', 'RFTN_Ce_peak',
                     'RFTN_rate_mean', 'RFTN_Ce_CV', 'RFTN_taper_slope',
                     'Time_Ce_above_4']
    outcome_vars = ['HR_rebound', 'MAP_rebound', 'FTN_rescue_mcg_kg',
                    'NHD_pct', 'icu_days']

    all_vars = [v for v in exposure_vars + outcome_vars if v in df.columns]
    if len(all_vars) < 4:
        return

    corr = df[all_vars].corr(method='spearman')

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    try:
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, vmin=-1, vmax=1, ax=ax, square=True,
                   linewidths=0.5, cbar_kws={'shrink': 0.8})
    except NameError:
        im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(all_vars)))
        ax.set_yticks(range(len(all_vars)))
        short = [v.replace('RFTN_', '').replace('_mcg_kg', '') for v in all_vars]
        ax.set_xticklabels(short, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(short, fontsize=8)
        plt.colorbar(im, ax=ax)

    ax.set_title('Spearman Correlation Matrix: Exposure & Outcome Variables')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sup_correlation_heatmap.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved Supplementary: Correlation heatmap")


# ============================================================
# Supplementary: 瑞芬太尺Ce时间序列示例
# ============================================================
def sup_rftn_trajectory_examples(df):
    """Supplementary: 典型瑞芬太尼输注模式示例"""
    # 此函数需要原始轨迹数据（需vitaldb包）
    # 这里生成模拟示例的占位图

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    patterns = [
        ('Low-dose stable', lambda t: 2 + 0.2*np.random.randn(len(t))),
        ('Moderate-dose stable', lambda t: 4 + 0.3*np.random.randn(len(t))),
        ('High-dose escalating', lambda t: 3 + 0.02*t + 0.3*np.random.randn(len(t))),
        ('High-dose fluctuating', lambda t: 6 + 2*np.sin(t/30) + 0.5*np.random.randn(len(t))),
    ]

    for ax, (title, func) in zip(axes.flat, patterns):
        t = np.arange(0, 180, 0.5)  # 3 hours, 0.5 min intervals
        ce = func(t)
        ce = np.clip(ce, 0, 15)

        ax.plot(t, ce, linewidth=0.8, alpha=0.8)
        ax.fill_between(t, 0, ce, alpha=0.2)
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 12)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Ce (ng/mL)')
        ax.set_title(f'Pattern: {title}')
        ax.axhline(4, color='orange', linestyle='--', alpha=0.5, label='OIH threshold (~4 ng/mL)')
        ax.legend(fontsize=8)

    fig.suptitle('Typical Remifentanil Infusion Patterns\n(Schematic illustration)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sup_rftn_patterns.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved Supplementary: RFTN trajectory patterns")


# ============================================================
# Main Visualization Pipeline
# ============================================================
def main():
    """​OIH可视化主流程"""
    print("=" * 70)
    print("  OIH Study - Phase 3: Visualization")
    print("=" * 70)

    # 加载数据
    master_file = OIH_DIR / "oih_master_dataset.csv"
    if not master_file.exists():
        # 尝试加载基础临床数据作为占位
        clinical_file = OIH_DIR / "oih_eligible_clinical.csv"
        if clinical_file.exists():
            df = pd.read_csv(clinical_file)
            print(f"  Loaded clinical data: {len(df)} cases")
        else:
            print("  [ERROR] No data files found. Run oih_01_data_extraction.py first.")
            return
    else:
        df = pd.read_csv(master_file)
        print(f"  Loaded master dataset: {len(df)} cases \u00d7 {len(df.columns)} columns")

    # ---- 异常值清理 ----
    if 'RFTN_total_mcg_kg' in df.columns and 'RFTN_Ce_peak' in df.columns:
        outlier_mask = (
            (df['RFTN_total_mcg_kg'] > 200) |
            (df['RFTN_Ce_peak'] > 100)
        )
        rftn_cols = [c for c in df.columns if c.startswith('RFTN_') or c in ['rftn_conc']]
        df.loc[outlier_mask, rftn_cols] = np.nan
        print(f"  Outlier cleanup: {outlier_mask.sum()} cases with extreme RFTN values \u2192 set to NaN")

    # 添加年龄分组
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0, 65, 75, 120],
                                  labels=['<65', '65-74', '>=75'], right=False)
        df['elderly'] = (df['age'] >= 65).astype(int)

    if 'RFTN_total_mcg_kg' in df.columns:
        df['RFTN_quartile'] = pd.qcut(df['RFTN_total_mcg_kg'].dropna(),
                                       q=4, labels=['Q1','Q2','Q3','Q4']
                                       ).reindex(df.index)

    # 生成所有图
    print("\n  Generating figures...")

    fig2_rftn_exposure_distribution(df)
    fig3_dose_response_curve(df)
    fig4_emax_model(df)
    fig5_multi_outcome_panel(df)
    fig6_age_stratified_effects(df)
    sup_correlation_heatmap(df)
    sup_rftn_trajectory_examples(df)

    print(f"\n  All figures saved to: {FIG_DIR}")
    print("  Phase 3 Complete!")


if __name__ == '__main__':
    main()
