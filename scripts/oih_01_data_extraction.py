#!/usr/bin/env python3
"""
=================================================================
OIH Study - Phase 1: Data Extraction & Case Screening
=================================================================
围术期瑞芬太尼诱导痛觉过敏（OIH）数据挖掘
基于VitalDB公共数据库

功能：
1. 下载/加载临床信息和轨迹列表
2. 应用OIH研究纳入/排除标准
3. 识别符合条件的病例
4. 批量下载瑞芬太尼 + BIS + 血流动力学轨迹数据
5. 计算瑞芬太尼暴露指标
6. 计算OIH替代终点指标
=================================================================
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
SHARED_DIR = PROJECT_DIR.parent / "shared_data"
OIH_DIR = DATA_DIR
RESULTS_DIR = PROJECT_DIR / "results"

# 创建目录
OIH_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Step 1: 加载基础数据
# ============================================================
def load_base_data():
    """加载或下载VitalDB基础数据"""
    print("=" * 70)
    print("Step 1: Loading base data from VitalDB...")
    print("=" * 70)

    # 优先使用本地缓存
    cases_path = SHARED_DIR / "clinical_information.csv"
    trks_path = SHARED_DIR / "track_list.csv"

    if cases_path.exists():
        print(f"  Loading cached clinical_information from {cases_path}")
        df_cases = pd.read_csv(cases_path)
    else:
        print("  Downloading clinical information from VitalDB API...")
        df_cases = pd.read_csv("https://api.vitaldb.net/cases")
        df_cases.to_csv(cases_path, index=False)

    if trks_path.exists():
        print(f"  Loading cached track_list from {trks_path}")
        df_trks = pd.read_csv(trks_path)
    else:
        print("  Downloading track list from VitalDB API...")
        df_trks = pd.read_csv("https://api.vitaldb.net/trks")
        df_trks.to_csv(trks_path, index=False)

    print(f"  Total cases: {len(df_cases)}")
    print(f"  Total tracks: {len(df_trks)}")
    print(f"  Unique track names: {df_trks['tname'].nunique()}")

    return df_cases, df_trks


# ============================================================
# Step 2: 识别各轨迹的病例集合
# ============================================================
def identify_track_availability(df_trks):
    """识别各关键轨迹的可用病例"""
    print("\n" + "=" * 70)
    print("Step 2: Identifying track availability...")
    print("=" * 70)

    track_sets = {}

    # 瑞芬太尼相关轨迹
    rftn_tracks = {
        'RFTN20_CE':   'Orchestra/RFTN20_CE',
        'RFTN20_CP':   'Orchestra/RFTN20_CP',
        'RFTN20_RATE': 'Orchestra/RFTN20_RATE',
        'RFTN20_VOL':  'Orchestra/RFTN20_VOL',
        'RFTN20_CT':   'Orchestra/RFTN20_CT',
        'RFTN50_CE':   'Orchestra/RFTN50_CE',
        'RFTN50_RATE': 'Orchestra/RFTN50_RATE',
        'RFTN50_VOL':  'Orchestra/RFTN50_VOL',
    }

    # BIS相关轨迹
    bis_tracks = {
        'BIS':  'BIS/BIS',
        'SQI':  'BIS/SQI',
        'EMG':  'BIS/EMG',
        'SR':   'BIS/SR',
    }

    # 血流动力学轨迹
    hemo_tracks = {
        'HR':      'Solar8000/HR',
        'ART_MBP': 'Solar8000/ART_MBP',
        'ART_SBP': 'Solar8000/ART_SBP',
        'ART_DBP': 'Solar8000/ART_DBP',
        'NIBP_MBP': 'Solar8000/NIBP_MBP',
    }

    # 丙泊酚TCI
    ppf_tracks = {
        'PPF20_CE':   'Orchestra/PPF20_CE',
        'PPF20_RATE': 'Orchestra/PPF20_RATE',
        'PPF20_VOL':  'Orchestra/PPF20_VOL',
    }

    all_tracks = {**rftn_tracks, **bis_tracks, **hemo_tracks, **ppf_tracks}

    for key, tname in all_tracks.items():
        cases = set(df_trks.loc[df_trks['tname'] == tname, 'caseid'])
        track_sets[key] = cases
        print(f"  {key:20s} ({tname:30s}): {len(cases):5d} cases")

    # 瑞芬太尼合并集（RFTN20 \u222a RFTN50）
    rftn_any = track_sets['RFTN20_CE'] | track_sets.get('RFTN50_CE', set())
    track_sets['RFTN_any'] = rftn_any
    print(f"\n  {'RFTN_any (20+50)':20s} {'':30s}: {len(rftn_any):5d} cases")

    # MBP合并集（ART \u222a NIBP）
    mbp_any = track_sets['ART_MBP'] | track_sets['NIBP_MBP']
    track_sets['MBP_any'] = mbp_any
    print(f"  {'MBP_any (ART+NIBP)':20s} {'':30s}: {len(mbp_any):5d} cases")

    return track_sets


# ============================================================
# Step 3: 应用纳入/排除标准
# ============================================================
def apply_criteria(df_cases, track_sets):
    """应用OIH研究的纳入/排除标准"""
    print("\n" + "=" * 70)
    print("Step 3: Applying inclusion/exclusion criteria...")
    print("=" * 70)

    screening = {}

    # --- 总数 ---
    all_cases = set(df_cases['caseid'])
    n_total = len(all_cases)
    screening['S0_total'] = n_total
    print(f"\n  [S0] Total VitalDB cases: {n_total}")

    # --- I1: 全身麻醉 ---
    ga = set(df_cases.loc[df_cases['ane_type'] == 'General', 'caseid'])
    eligible = all_cases & ga
    screening['I1_general_anesthesia'] = len(eligible)
    print(f"  [I1] General anesthesia: {len(eligible)}")

    # --- I2: 有瑞芬太尼TCI数据 ---
    eligible = eligible & track_sets['RFTN_any']
    screening['I2_has_rftn'] = len(eligible)
    print(f"  [I2] + Has remifentanil TCI: {len(eligible)}")

    # --- I3: 有BIS监测 ---
    has_bis = track_sets['BIS'] & track_sets['SQI']
    eligible = eligible & has_bis
    screening['I3_has_bis'] = len(eligible)
    print(f"  [I3] + Has BIS + SQI: {len(eligible)}")

    # --- I4: 年龄 \u2265 18 ---
    adult = set(df_cases.loc[df_cases['age'] >= 18, 'caseid'])
    eligible = eligible & adult
    screening['I4_adult'] = len(eligible)
    print(f"  [I4] + Age >= 18: {len(eligible)}")

    # --- I5: 麻醉时间 \u2265 60 min ---
    long_enough = set(df_cases.loc[
        (df_cases['anedur'] >= 60) & df_cases['anedur'].notna(), 'caseid'
    ])
    eligible = eligible & long_enough
    screening['I5_anedur_60'] = len(eligible)
    print(f"  [I5] + Anesthesia duration >= 60min: {len(eligible)}")

    # --- E1: 排除神经外科 ---
    neuro = set(df_cases.loc[
        df_cases['department'].str.contains(
            'neuro|NS|brain|cranio', case=False, na=False
        ), 'caseid'
    ])
    n_neuro = len(eligible & neuro)
    eligible = eligible - neuro
    screening['E1_exclude_neuro'] = len(eligible)
    print(f"  [E1] - Exclude neurosurgery ({n_neuro} removed): {len(eligible)}")

    # --- E2: 排除 ASA \u2265 V ---
    asa5 = set(df_cases.loc[df_cases['asa'] >= 5, 'caseid'])
    n_asa5 = len(eligible & asa5)
    eligible = eligible - asa5
    screening['E2_exclude_asa5'] = len(eligible)
    print(f"  [E2] - Exclude ASA >= V ({n_asa5} removed): {len(eligible)}")

    # --- 有血流动力学数据（HR + MBP） ---
    has_hemo = track_sets['HR'] & track_sets['MBP_any']
    eligible_with_hemo = eligible & has_hemo
    screening['has_hemodynamics'] = len(eligible_with_hemo)
    print(f"\n  [QC] Cases with hemodynamic data (HR + MBP): {len(eligible_with_hemo)}")

    # 最终使用有血流动力学数据的子集（OIH替代终点需要）
    final_eligible = sorted(list(eligible_with_hemo))
    screening['final_eligible'] = len(final_eligible)
    print(f"\n  >>> FINAL ELIGIBLE (with hemodynamics): {len(final_eligible)}")

    return final_eligible, screening


# ============================================================
# Step 4: 人群特征描述
# ============================================================
def describe_population(df_cases, eligible_ids, track_sets):
    """描述纳入人群特征"""
    print("\n" + "=" * 70)
    print("Step 4: Population characteristics...")
    print("=" * 70)

    df = df_cases[df_cases['caseid'].isin(eligible_ids)].copy()

    # 人口学
    print(f"\n  --- Demographics ---")
    print(f"  N = {len(df)}")
    print(f"  Age: {df['age'].mean():.1f} +/- {df['age'].std():.1f} years "
          f"(range: {df['age'].min():.0f}-{df['age'].max():.0f})")

    age_groups = {
        '<65': (df['age'] < 65).sum(),
        '65-74': ((df['age'] >= 65) & (df['age'] < 75)).sum(),
        '>=75': (df['age'] >= 75).sum(),
    }
    print(f"  Age groups:")
    for g, n in age_groups.items():
        print(f"    {g}: {n} ({n/len(df)*100:.1f}%)")

    print(f"  Female: {(df['sex'] == 'F').sum()} ({(df['sex'] == 'F').mean()*100:.1f}%)")
    print(f"  BMI: {df['bmi'].mean():.1f} +/- {df['bmi'].std():.1f}")

    # ASA
    print(f"\n  --- ASA Distribution ---")
    for asa in sorted(df['asa'].dropna().unique()):
        n = (df['asa'] == asa).sum()
        print(f"    ASA {int(asa)}: {n} ({n/len(df)*100:.1f}%)")

    # 手术特征
    print(f"\n  --- Surgical Characteristics ---")
    print(f"  Anesthesia duration: {df['anedur'].mean():.0f} +/- {df['anedur'].std():.0f} min")
    print(f"  Surgery duration: {df['opdur'].mean():.0f} +/- {df['opdur'].std():.0f} min")
    print(f"  Emergency: {df['emop'].sum()} ({df['emop'].mean()*100:.1f}%)")

    print(f"\n  --- Top 10 Departments ---")
    for dept, n in df['department'].value_counts().head(10).items():
        print(f"    {dept}: {n} ({n/len(df)*100:.1f}%)")

    # 阿片类药物数据
    print(f"\n  --- Opioid Data ---")
    elig_set = set(eligible_ids)
    has_rftn20 = len(elig_set & track_sets['RFTN20_CE'])
    has_rftn50 = len(elig_set & track_sets.get('RFTN50_CE', set()))
    has_ftn = df['intraop_ftn'].notna().sum()
    has_ftn_gt0 = (df['intraop_ftn'] > 0).sum()

    print(f"    RFTN20 TCI: {has_rftn20} ({has_rftn20/len(df)*100:.1f}%)")
    print(f"    RFTN50 TCI: {has_rftn50} ({has_rftn50/len(df)*100:.1f}%)")
    print(f"    Fentanyl bolus recorded: {has_ftn}")
    print(f"    Fentanyl bolus > 0: {has_ftn_gt0} ({has_ftn_gt0/len(df)*100:.1f}%)")
    print(f"    Fentanyl dose (when >0): {df.loc[df['intraop_ftn']>0, 'intraop_ftn'].mean():.1f} "
          f"+/- {df.loc[df['intraop_ftn']>0, 'intraop_ftn'].std():.1f} mcg")

    # 结局数据
    print(f"\n  --- Outcome Data Availability ---")
    print(f"    ICU days > 0: {(df['icu_days'] > 0).sum()} ({(df['icu_days'] > 0).mean()*100:.1f}%)")
    print(f"    In-hospital death: {df['death_inhosp'].sum()} ({df['death_inhosp'].mean()*100:.1f}%)")

    # 丙泊酚
    print(f"\n  --- Propofol Data ---")
    has_ppf = len(elig_set & track_sets['PPF20_CE'])
    print(f"    PPF20 TCI: {has_ppf} ({has_ppf/len(df)*100:.1f}%)")
    print(f"    Propofol total (mg): {df['intraop_ppf'].mean():.0f} +/- {df['intraop_ppf'].std():.0f}")

    return df


# ============================================================
# Step 5: 计算瑞芬太尼暴露指标
# ============================================================
def compute_rftn_exposure_single(caseid, weight, opdur_min, track_sets):
    """
    计算单个病例的瑞芬太尼暴露指标

    Returns: dict of exposure metrics, or None if data unavailable
    """
    try:
        import vitaldb
    except ImportError:
        print("  [WARNING] vitaldb package not installed. Using API fallback.")
        return _compute_rftn_api(caseid, weight, opdur_min, track_sets)

    # 确定使用RFTN20还是RFTN50
    if caseid in track_sets.get('RFTN20_CE', set()):
        tracks = ['Orchestra/RFTN20_CE', 'Orchestra/RFTN20_RATE',
                  'Orchestra/RFTN20_VOL', 'Orchestra/RFTN20_CT']
        conc = 20  # \u03bcg/mL
    elif caseid in track_sets.get('RFTN50_CE', set()):
        tracks = ['Orchestra/RFTN50_CE', 'Orchestra/RFTN50_RATE',
                  'Orchestra/RFTN50_VOL', 'Orchestra/RFTN50_CT']
        conc = 50
    else:
        return None

    try:
        data = vitaldb.load_case(caseid, tracks, interval=1)
        if data is None or data.shape[0] < 60:
            return None
    except Exception as e:
        print(f"    [ERROR] caseid={caseid}: {e}")
        return None

    ce   = data[:, 0]  # 效应部位浓度 (ng/mL)
    rate = data[:, 1]  # 输注速率 (mL/h)
    vol  = data[:, 2]  # 累积体积 (mL)
    ct   = data[:, 3]  # 目标浓度 (ng/mL)

    results = {'caseid': caseid, 'rftn_conc': conc}

    # ---- A. 总量指标 ----
    vol_valid = vol[~np.isnan(vol)]
    if len(vol_valid) > 1:
        total_vol_ml = np.nanmax(vol) - np.nanmin(vol_valid[vol_valid >= 0])
    else:
        total_vol_ml = 0

    results['RFTN_total_mcg'] = total_vol_ml * conc
    results['RFTN_total_mcg_kg'] = results['RFTN_total_mcg'] / weight if weight > 0 else np.nan
    results['RFTN_mcg_kg_hr'] = (
        results['RFTN_total_mcg_kg'] / (opdur_min / 60)
        if opdur_min > 0 else np.nan
    )

    # Ce AUC（梯形积分）
    ce_clean = np.where(np.isnan(ce), 0, ce)
    results['RFTN_AUC_Ce'] = np.trapz(ce_clean, dx=1/60)  # ng\u00b7min/mL

    # ---- B. 速率指标 ----
    if weight > 0:
        rate_mcg_kg_min = rate * conc / 60 / weight
        results['RFTN_rate_mean'] = np.nanmean(rate_mcg_kg_min)
        results['RFTN_rate_peak'] = np.nanmax(rate_mcg_kg_min)
    else:
        results['RFTN_rate_mean'] = np.nan
        results['RFTN_rate_peak'] = np.nan

    ce_valid = ce[~np.isnan(ce)]
    if len(ce_valid) > 0:
        results['RFTN_Ce_mean'] = np.nanmean(ce_valid)
        results['RFTN_Ce_peak'] = np.nanmax(ce_valid)
        results['RFTN_Ce_median'] = np.nanmedian(ce_valid)
    else:
        results['RFTN_Ce_mean'] = np.nan
        results['RFTN_Ce_peak'] = np.nan
        results['RFTN_Ce_median'] = np.nan

    # 术末Ce（最后60秒均值）
    ce_end = ce[-60:] if len(ce) >= 60 else ce
    results['RFTN_Ce_at_end'] = np.nanmean(ce_end)

    # ---- C. 波动/模式指标 ----
    if len(ce_valid) > 60:
        results['RFTN_Ce_SD'] = np.nanstd(ce_valid, ddof=1)
        results['RFTN_Ce_CV'] = (results['RFTN_Ce_SD'] / results['RFTN_Ce_mean'] * 100
                                  if results['RFTN_Ce_mean'] > 0 else np.nan)
        results['RFTN_Ce_ARV'] = np.nanmean(np.abs(np.diff(ce_valid)))
    else:
        results['RFTN_Ce_SD'] = np.nan
        results['RFTN_Ce_CV'] = np.nan
        results['RFTN_Ce_ARV'] = np.nan

    # 目标浓度调整次数
    ct_valid = ct[~np.isnan(ct)]
    if len(ct_valid) > 1:
        ct_diff = np.abs(np.diff(ct_valid))
        results['RFTN_Ct_changes'] = int(np.sum(ct_diff > 0.05))
    else:
        results['RFTN_Ct_changes'] = 0

    # 术末30min Ce斜率
    n_last = min(1800, len(ce))
    last_ce = ce[-n_last:]
    last_valid = last_ce[~np.isnan(last_ce)]
    if len(last_valid) > 60:
        x = np.arange(len(last_valid))
        slope = np.polyfit(x, last_valid, 1)[0]
        results['RFTN_taper_slope'] = slope * 60  # ng/mL per minute
    else:
        results['RFTN_taper_slope'] = np.nan

    # 高浓度暴露时间（分钟）
    results['Time_Ce_above_4'] = np.sum(ce > 4) / 60
    results['Time_Ce_above_6'] = np.sum(ce > 6) / 60
    results['Time_Ce_above_8'] = np.sum(ce > 8) / 60

    # 数据质量指标
    results['RFTN_data_duration_sec'] = len(ce)
    results['RFTN_nan_pct'] = np.sum(np.isnan(ce)) / len(ce) * 100

    return results


def _compute_rftn_api(caseid, weight, opdur_min, track_sets):
    """​API回退方案（不使用vitaldb包时）"""
    # 此处留作备用，实际建议安装vitaldb包
    return None


# ============================================================
# Step 6: 计算血流动力学反弹指标（OIH替代终点）
# ============================================================
def compute_hemodynamic_rebound(caseid, opstart_sec, opend_sec, track_sets):
    """
    计算术中晚期血流动力学反弹指标

    时间窗定义：
    - 稳定期(baseline): 手术开始后30min \u2192 术末前60min
    - 晚期(late):       术末前30min \u2192 术末
    - 早期(early):      手术开始前30min（如有PACU数据）

    Returns: dict of rebound metrics
    """
    try:
        import vitaldb
    except ImportError:
        return None

    try:
        # 加载HR和MBP
        hr_track = 'Solar8000/HR'
        mbp_track = 'Solar8000/ART_MBP'

        data = vitaldb.load_case(caseid, [hr_track, mbp_track], interval=1)
        if data is None or data.shape[0] < 60:
            # 尝试NIBP
            data = vitaldb.load_case(caseid, [hr_track, 'Solar8000/NIBP_MBP'], interval=1)
            if data is None:
                return None

        hr = data[:, 0]
        mbp = data[:, 1]

    except Exception as e:
        print(f"    [ERROR] hemodynamic caseid={caseid}: {e}")
        return None

    results = {'caseid': caseid}

    # 时间窗（秒为单位的索引）
    total_len = len(hr)
    op_dur = opend_sec - opstart_sec

    if op_dur < 90 * 60:  # 手术时间至少90分钟才能计算
        return None

    # 绝对索引（从记录开始）
    stable_start = int(opstart_sec + 30 * 60)
    stable_end = int(opend_sec - 60 * 60)
    late_start = int(opend_sec - 30 * 60)
    late_end = int(opend_sec)

    # 边界检查
    if stable_start >= stable_end or stable_end < 0 or late_start < 0:
        return None
    stable_start = max(0, min(stable_start, total_len - 1))
    stable_end = max(0, min(stable_end, total_len - 1))
    late_start = max(0, min(late_start, total_len - 1))
    late_end = max(0, min(late_end, total_len - 1))

    # 稳定期指标
    hr_stable = hr[stable_start:stable_end]
    mbp_stable = mbp[stable_start:stable_end]
    hr_stable_mean = np.nanmean(hr_stable)
    mbp_stable_mean = np.nanmean(mbp_stable)

    # 晚期指标
    hr_late = hr[late_start:late_end]
    mbp_late = mbp[late_start:late_end]
    hr_late_mean = np.nanmean(hr_late)
    mbp_late_mean = np.nanmean(mbp_late)

    # 反弹幅度
    results['HR_stable_mean'] = hr_stable_mean
    results['MAP_stable_mean'] = mbp_stable_mean
    results['HR_late_mean'] = hr_late_mean
    results['MAP_late_mean'] = mbp_late_mean
    results['HR_rebound'] = hr_late_mean - hr_stable_mean
    results['MAP_rebound'] = mbp_late_mean - mbp_stable_mean

    # 反弹百分比
    if hr_stable_mean > 0:
        results['HR_rebound_pct'] = results['HR_rebound'] / hr_stable_mean * 100
    else:
        results['HR_rebound_pct'] = np.nan

    if mbp_stable_mean > 0:
        results['MAP_rebound_pct'] = results['MAP_rebound'] / mbp_stable_mean * 100
    else:
        results['MAP_rebound_pct'] = np.nan

    # 晚期变异度
    hr_late_valid = hr_late[~np.isnan(hr_late)]
    if len(hr_late_valid) > 30:
        results['HR_late_SD'] = np.nanstd(hr_late_valid, ddof=1)
        results['HR_late_CV'] = results['HR_late_SD'] / np.nanmean(hr_late_valid) * 100
    else:
        results['HR_late_SD'] = np.nan
        results['HR_late_CV'] = np.nan

    mbp_late_valid = mbp_late[~np.isnan(mbp_late)]
    if len(mbp_late_valid) > 30:
        results['MAP_late_SD'] = np.nanstd(mbp_late_valid, ddof=1)
        results['MAP_late_CV'] = results['MAP_late_SD'] / np.nanmean(mbp_late_valid) * 100
    else:
        results['MAP_late_SD'] = np.nan
        results['MAP_late_CV'] = np.nan

    # 术后30min指标（如果数据延续到手术结束后）
    post_start = int(opend_sec)
    post_end = int(opend_sec + 30 * 60)
    if post_end <= total_len:
        hr_post = hr[post_start:post_end]
        mbp_post = mbp[post_start:post_end]
        results['HR_post30_mean'] = np.nanmean(hr_post)
        results['MAP_post30_mean'] = np.nanmean(mbp_post)
        results['HR_post_rebound'] = np.nanmean(hr_post) - hr_stable_mean
        results['MAP_post_rebound'] = np.nanmean(mbp_post) - mbp_stable_mean
    else:
        results['HR_post30_mean'] = np.nan
        results['MAP_post30_mean'] = np.nan
        results['HR_post_rebound'] = np.nan
        results['MAP_post_rebound'] = np.nan

    return results


# ============================================================
# Step 7: 计算BIS相关指标
# ============================================================
def compute_bis_metrics(caseid, opstart_sec, opend_sec):
    """计算BIS波动指标（与OIH分析相关）"""
    try:
        import vitaldb
    except ImportError:
        return None

    try:
        data = vitaldb.load_case(
            caseid,
            ['BIS/BIS', 'BIS/SQI', 'BIS/EMG'],
            interval=1
        )
        if data is None:
            return None
    except Exception:
        return None

    bis = data[:, 0]
    sqi = data[:, 1]
    emg = data[:, 2]

    # 截取手术期间
    start = max(0, int(opstart_sec))
    end = min(len(bis), int(opend_sec))
    bis_op = bis[start:end].copy()
    sqi_op = sqi[start:end] if sqi is not None else None
    emg_op = emg[start:end] if emg is not None else None

    # 质量过滤
    if sqi_op is not None:
        bis_op[sqi_op < 50] = np.nan
    if emg_op is not None:
        bis_op[emg_op > 55] = np.nan
    bis_op[(bis_op < 0) | (bis_op > 100)] = np.nan

    # 有效数据
    bis_valid = bis_op[~np.isnan(bis_op)]
    if len(bis_valid) < 300:  # 至少5分钟有效数据
        return None

    results = {'caseid': caseid}
    results['TWA_BIS'] = np.nanmean(bis_valid)
    results['SD_BIS'] = np.nanstd(bis_valid, ddof=1)
    results['CV_BIS'] = results['SD_BIS'] / results['TWA_BIS'] * 100
    results['BIS_pct_in_range'] = np.sum((bis_valid >= 40) & (bis_valid <= 60)) / len(bis_valid) * 100
    results['BIS_nan_pct'] = np.sum(np.isnan(bis_op)) / len(bis_op) * 100

    return results


# ============================================================
# Step 8: NHD指数（Nociception-Hypnosis Dissociation）
# ============================================================
def compute_nhd_index(caseid, opstart_sec, opend_sec):
    """
    计算伤害感受-催眠分离指数

    当BIS在40-60（催眠充分）但HR/MAP升高时，
    提示伤害感受通路异常激活——可能与中枢敏化/OIH相关
    """
    try:
        import vitaldb
    except ImportError:
        return None

    try:
        data = vitaldb.load_case(
            caseid,
            ['BIS/BIS', 'BIS/SQI', 'Solar8000/HR', 'Solar8000/ART_MBP'],
            interval=1
        )
        if data is None:
            return None
    except Exception:
        return None

    bis = data[:, 0]
    sqi = data[:, 1]
    hr  = data[:, 2]
    mbp = data[:, 3]

    start = max(0, int(opstart_sec))
    end = min(len(bis), int(opend_sec))

    bis_op = bis[start:end].copy()
    sqi_op = sqi[start:end]
    hr_op  = hr[start:end]
    mbp_op = mbp[start:end]

    # BIS质量过滤
    if sqi_op is not None:
        bis_op[sqi_op < 50] = np.nan

    # 计算HR/MAP基线（手术开始后30min的稳定期）
    stable_end = min(30 * 60, len(hr_op))
    hr_baseline = np.nanmean(hr_op[:stable_end])
    mbp_baseline = np.nanmean(mbp_op[:stable_end])

    if np.isnan(hr_baseline) or np.isnan(mbp_baseline):
        return None

    # 识别分离时段
    # BIS在适当范围（40-60）但HR或MAP升高（>基线\u00d71.2）
    bis_ok = (bis_op >= 40) & (bis_op <= 60)
    hr_high = hr_op > hr_baseline * 1.2
    mbp_high = mbp_op > mbp_baseline * 1.2

    dissociation = bis_ok & (hr_high | mbp_high)
    total_valid = (~np.isnan(bis_op)).sum()

    results = {'caseid': caseid}
    results['NHD_seconds'] = int(np.nansum(dissociation))
    results['NHD_pct'] = results['NHD_seconds'] / total_valid * 100 if total_valid > 0 else np.nan
    results['HR_baseline'] = hr_baseline
    results['MAP_baseline'] = mbp_baseline

    return results


# ============================================================
# Main Pipeline
# ============================================================
def main():
    """​OIH数据提取主流程"""
    print("=" * 70)
    print("  OIH Study - Phase 1: Data Extraction & Screening")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Step 1: 加载基础数据
    df_cases, df_trks = load_base_data()

    # Step 2: 轨迹可用性
    track_sets = identify_track_availability(df_trks)

    # Step 3: 筛选
    eligible_ids, screening = apply_criteria(df_cases, track_sets)

    # Step 4: 人群描述
    df_eligible = describe_population(df_cases, eligible_ids, track_sets)

    # 保存中间结果
    pd.Series(eligible_ids, name='caseid').to_csv(OIH_DIR / "oih_eligible_caseids.csv", index=False)
    df_eligible.to_csv(OIH_DIR / "oih_eligible_clinical.csv", index=False)
    with open(RESULTS_DIR / "oih_screening_log.json", 'w') as f:
        json.dump(screening, f, indent=2)

    print(f"\n  Saved {len(eligible_ids)} eligible case IDs to {OIH_DIR / 'oih_eligible_caseids.csv'}")
    print(f"  Saved screening log to {RESULTS_DIR / 'oih_screening_log.json'}")

    # ============================================================
    # Step 5-8: 轨迹数据处理（需要vitaldb包，耗时较长）
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 5-8: Computing exposure and outcome metrics...")
    print("  NOTE: This requires the 'vitaldb' package and may take hours.")
    print("  To run: pip install vitaldb && python oih_01_data_extraction.py")
    print("=" * 70)

    try:
        import vitaldb
        VITALDB_AVAILABLE = True
        print("  vitaldb package found. Proceeding with track data extraction...")
    except ImportError:
        VITALDB_AVAILABLE = False
        print("  vitaldb package NOT found. Skipping track data extraction.")
        print("  Install with: pip install vitaldb")
        print("  Phase 1 (screening) complete. Run again after installing vitaldb.")
        return

    # 计算手术时间参数
    df_eligible['opstart_dt'] = pd.to_datetime(df_eligible['opstart'])
    df_eligible['opend_dt'] = pd.to_datetime(df_eligible['opend'])
    df_eligible['anestart_dt'] = pd.to_datetime(df_eligible['anestart'])

    # 瑞芬太尼暴露指标
    rftn_results = []
    hemo_results = []
    bis_results = []
    nhd_results = []

    total = len(eligible_ids)
    for i, caseid in enumerate(eligible_ids):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"\n  Processing case {i+1}/{total} (caseid={caseid})...")

        row = df_eligible[df_eligible['caseid'] == caseid].iloc[0]
        weight = row['weight'] if pd.notna(row['weight']) and row['weight'] > 0 else 60
        opdur_min = row['opdur'] if pd.notna(row['opdur']) else 0

        # 计算秒数偏移（从麻醉开始）
        if pd.notna(row['opstart']) and pd.notna(row['anestart']):
            opstart_sec = (row['opstart_dt'] - row['anestart_dt']).total_seconds()
            opend_sec = (row['opend_dt'] - row['anestart_dt']).total_seconds()
        else:
            opstart_sec = 0
            opend_sec = opdur_min * 60

        # 5. 瑞芬太尼暴露
        rftn = compute_rftn_exposure_single(caseid, weight, opdur_min, track_sets)
        if rftn:
            rftn_results.append(rftn)

        # 6. 血流动力学反弹
        hemo = compute_hemodynamic_rebound(caseid, opstart_sec, opend_sec, track_sets)
        if hemo:
            hemo_results.append(hemo)

        # 7. BIS指标
        bis = compute_bis_metrics(caseid, opstart_sec, opend_sec)
        if bis:
            bis_results.append(bis)

        # 8. NHD指数
        nhd = compute_nhd_index(caseid, opstart_sec, opend_sec)
        if nhd:
            nhd_results.append(nhd)

    # 保存结果
    if rftn_results:
        df_rftn = pd.DataFrame(rftn_results)
        df_rftn.to_csv(OIH_DIR / "oih_rftn_exposure.csv", index=False)
        print(f"\n  Saved RFTN exposure data: {len(df_rftn)} cases")

    if hemo_results:
        df_hemo = pd.DataFrame(hemo_results)
        df_hemo.to_csv(OIH_DIR / "oih_hemodynamic_rebound.csv", index=False)
        print(f"  Saved hemodynamic rebound data: {len(df_hemo)} cases")

    if bis_results:
        df_bis = pd.DataFrame(bis_results)
        df_bis.to_csv(OIH_DIR / "oih_bis_metrics.csv", index=False)
        print(f"  Saved BIS metrics: {len(df_bis)} cases")

    if nhd_results:
        df_nhd = pd.DataFrame(nhd_results)
        df_nhd.to_csv(OIH_DIR / "oih_nhd_index.csv", index=False)
        print(f"  Saved NHD index: {len(df_nhd)} cases")

    # 合并主数据集
    df_main = df_eligible.copy()
    if rftn_results:
        df_main = df_main.merge(df_rftn, on='caseid', how='left')
    if hemo_results:
        df_main = df_main.merge(df_hemo, on='caseid', how='left')
    if bis_results:
        df_main = df_main.merge(df_bis, on='caseid', how='left')
    if nhd_results:
        df_main = df_main.merge(df_nhd, on='caseid', how='left')

    # 计算附加指标
    # 芬太尼追加量标准化
    df_main['FTN_rescue_mcg_kg'] = df_main['intraop_ftn'] / df_main['weight']

    # LOS计算
    df_main['LOS_total'] = df_main['icu_days']  # 近似（VitalDB的adm/dis均为同日期）

    # 保存最终合并数据集
    df_main.to_csv(OIH_DIR / "oih_master_dataset.csv", index=False)
    print(f"\n  Saved master dataset: {len(df_main)} cases \u00d7 {len(df_main.columns)} variables")

    # 最终摘要
    print("\n" + "=" * 70)
    print("  Phase 1 Complete!")
    print("=" * 70)
    print(f"  Final eligible cases: {len(eligible_ids)}")
    print(f"  With RFTN exposure data: {len(rftn_results)}")
    print(f"  With hemodynamic rebound: {len(hemo_results)}")
    print(f"  With BIS metrics: {len(bis_results)}")
    print(f"  With NHD index: {len(nhd_results)}")
    print(f"\n  Output directory: {OIH_DIR}")
    print(f"  Next: Run oih_02_statistical_analysis.py")


if __name__ == '__main__':
    main()
