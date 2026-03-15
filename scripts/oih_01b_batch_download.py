#!/usr/bin/env python3
"""
=================================================================
OIH Study - Phase 1b: Batch Track Data Download & Processing
=================================================================
支持断点续传的批量轨迹数据下载与指标计算

特性：
- 每处理50例自动保存进度（断点续传）
- 跳过已处理的病例
- 错误容忍（单例失败不影响全局）
- 实时进度与速度估算
- 分别保存：RFTN暴露、血流动力学反弹、BIS指标

用法：
  python oih_01b_batch_download.py          # 从头或续传
  python oih_01b_batch_download.py --start 0 --end 100  # 仅处理前100例
=================================================================
"""

import pandas as pd
import numpy as np
import vitaldb
import json
import time
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================
# Configuration
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
OIH_DIR = PROJECT_DIR / "data"
CHECKPOINT_INTERVAL = 50  # 每N例保存一次

# 输出文件
RFTN_FILE = OIH_DIR / "oih_rftn_exposure.csv"
HEMO_FILE = OIH_DIR / "oih_hemodynamic_rebound.csv"
BIS_FILE  = OIH_DIR / "oih_bis_metrics.csv"
NHD_FILE  = OIH_DIR / "oih_nhd_index.csv"
PROGRESS_FILE = OIH_DIR / "oih_download_progress.json"


# ============================================================
# 加载已完成的进度
# ============================================================
def load_progress():
    """加载已处理的caseid集合"""
    done = set()
    for f in [RFTN_FILE, HEMO_FILE, BIS_FILE, NHD_FILE]:
        if f.exists():
            try:
                df = pd.read_csv(f)
                if 'caseid' in df.columns:
                    done |= set(df['caseid'].tolist())
            except Exception:
                pass
    return done


def load_existing_results():
    """加载已有的结果列表（用于追加）"""
    results = {'rftn': [], 'hemo': [], 'bis': [], 'nhd': []}
    for key, f in [('rftn', RFTN_FILE), ('hemo', HEMO_FILE),
                   ('bis', BIS_FILE), ('nhd', NHD_FILE)]:
        if f.exists():
            try:
                df = pd.read_csv(f)
                results[key] = df.to_dict('records')
            except Exception:
                results[key] = []
    return results


def save_results(results):
    """保存当前结果到CSV"""
    for key, f in [('rftn', RFTN_FILE), ('hemo', HEMO_FILE),
                   ('bis', BIS_FILE), ('nhd', NHD_FILE)]:
        if results[key]:
            df = pd.DataFrame(results[key])
            df.to_csv(f, index=False)


# ============================================================
# 核心计算函数
# ============================================================
def process_rftn(data, caseid, weight, opdur_min, conc=20):
    """计算瑞芬太尼暴露指标"""
    ce   = data[:, 0]
    rate = data[:, 1]
    vol  = data[:, 2]
    ct   = data[:, 3]

    r = {'caseid': caseid, 'rftn_conc': conc}

    # 总量
    vol_valid = vol[~np.isnan(vol)]
    if len(vol_valid) > 1 and np.any(vol_valid >= 0):
        total_vol = np.nanmax(vol) - np.nanmin(vol_valid[vol_valid >= 0])
    else:
        total_vol = 0

    r['RFTN_total_mcg'] = total_vol * conc
    r['RFTN_total_mcg_kg'] = r['RFTN_total_mcg'] / weight if weight > 0 else np.nan
    r['RFTN_mcg_kg_hr'] = r['RFTN_total_mcg_kg'] / (opdur_min / 60) if opdur_min > 0 else np.nan

    # Ce AUC
    ce_c = np.where(np.isnan(ce), 0, ce)
    r['RFTN_AUC_Ce'] = float(np.trapezoid(ce_c, dx=1/60))

    # 速率
    if weight > 0:
        rate_conv = rate * conc / 60 / weight  # μg/kg/min
        r['RFTN_rate_mean'] = float(np.nanmean(rate_conv))
        r['RFTN_rate_peak'] = float(np.nanmax(rate_conv)) if np.any(~np.isnan(rate_conv)) else np.nan
    else:
        r['RFTN_rate_mean'] = np.nan
        r['RFTN_rate_peak'] = np.nan

    ce_v = ce[~np.isnan(ce)]
    if len(ce_v) > 0:
        r['RFTN_Ce_mean'] = float(np.mean(ce_v))
        r['RFTN_Ce_peak'] = float(np.max(ce_v))
        r['RFTN_Ce_median'] = float(np.median(ce_v))
    else:
        r['RFTN_Ce_mean'] = r['RFTN_Ce_peak'] = r['RFTN_Ce_median'] = np.nan

    # 术末Ce
    tail = ce[-60:] if len(ce) >= 60 else ce
    tail_v = tail[~np.isnan(tail)]
    r['RFTN_Ce_at_end'] = float(np.mean(tail_v)) if len(tail_v) > 0 else np.nan

    # 波动
    if len(ce_v) > 60:
        r['RFTN_Ce_SD'] = float(np.std(ce_v, ddof=1))
        r['RFTN_Ce_CV'] = r['RFTN_Ce_SD'] / r['RFTN_Ce_mean'] * 100 if r['RFTN_Ce_mean'] > 0 else np.nan
        r['RFTN_Ce_ARV'] = float(np.mean(np.abs(np.diff(ce_v))))
    else:
        r['RFTN_Ce_SD'] = r['RFTN_Ce_CV'] = r['RFTN_Ce_ARV'] = np.nan

    # Ct调整次数
    ct_v = ct[~np.isnan(ct)]
    if len(ct_v) > 1:
        r['RFTN_Ct_changes'] = int(np.sum(np.abs(np.diff(ct_v)) > 0.05))
    else:
        r['RFTN_Ct_changes'] = 0

    # 术末30min斜率
    n_last = min(1800, len(ce))
    last_ce = ce[-n_last:]
    lv = last_ce[~np.isnan(last_ce)]
    if len(lv) > 60:
        slope = np.polyfit(np.arange(len(lv)), lv, 1)[0]
        r['RFTN_taper_slope'] = float(slope * 60)
    else:
        r['RFTN_taper_slope'] = np.nan

    # 高浓度暴露时间
    r['Time_Ce_above_4'] = float(np.sum(ce > 4) / 60)
    r['Time_Ce_above_6'] = float(np.sum(ce > 6) / 60)
    r['Time_Ce_above_8'] = float(np.sum(ce > 8) / 60)

    r['RFTN_data_len'] = int(len(ce))
    r['RFTN_nan_pct'] = float(np.sum(np.isnan(ce)) / len(ce) * 100)

    return r


def process_hemo(data, caseid, opstart_sec, opend_sec):
    """计算血流动力学反弹指标"""
    hr  = data[:, 4]  # Solar8000/HR
    mbp_art = data[:, 5]  # ART_MBP
    mbp_nibp = data[:, 6]  # NIBP_MBP

    # 优先使用ART_MBP，回退到NIBP_MBP
    art_valid = np.sum(~np.isnan(mbp_art))
    nibp_valid = np.sum(~np.isnan(mbp_nibp))
    mbp = mbp_art if art_valid > nibp_valid else mbp_nibp
    mbp_source = 'ART' if art_valid > nibp_valid else 'NIBP'

    total_len = len(hr)
    op_dur_sec = opend_sec - opstart_sec

    if op_dur_sec < 90 * 60:
        return None

    # 时间窗索引
    stable_s = max(0, int(opstart_sec + 30 * 60))
    stable_e = max(0, int(opend_sec - 60 * 60))
    late_s = max(0, int(opend_sec - 30 * 60))
    late_e = max(0, int(opend_sec))

    if stable_s >= stable_e:
        return None

    stable_s = min(stable_s, total_len - 1)
    stable_e = min(stable_e, total_len - 1)
    late_s = min(late_s, total_len - 1)
    late_e = min(late_e, total_len - 1)

    hr_sm = np.nanmean(hr[stable_s:stable_e])
    mbp_sm = np.nanmean(mbp[stable_s:stable_e])
    hr_lm = np.nanmean(hr[late_s:late_e])
    mbp_lm = np.nanmean(mbp[late_s:late_e])

    if np.isnan(hr_sm) or np.isnan(hr_lm):
        return None

    r = {'caseid': caseid, 'mbp_source': mbp_source}
    r['HR_stable_mean'] = float(hr_sm)
    r['MAP_stable_mean'] = float(mbp_sm)
    r['HR_late_mean'] = float(hr_lm)
    r['MAP_late_mean'] = float(mbp_lm)
    r['HR_rebound'] = float(hr_lm - hr_sm)
    r['MAP_rebound'] = float(mbp_lm - mbp_sm)
    r['HR_rebound_pct'] = float(r['HR_rebound'] / hr_sm * 100) if hr_sm > 0 else np.nan
    r['MAP_rebound_pct'] = float(r['MAP_rebound'] / mbp_sm * 100) if mbp_sm > 0 else np.nan

    # 晚期变异度
    hr_late = hr[late_s:late_e]
    hr_lv = hr_late[~np.isnan(hr_late)]
    if len(hr_lv) > 30:
        r['HR_late_SD'] = float(np.std(hr_lv, ddof=1))
        r['HR_late_CV'] = float(r['HR_late_SD'] / np.mean(hr_lv) * 100)
    else:
        r['HR_late_SD'] = r['HR_late_CV'] = np.nan

    mbp_late = mbp[late_s:late_e]
    mbp_lv = mbp_late[~np.isnan(mbp_late)]
    if len(mbp_lv) > 30:
        r['MAP_late_SD'] = float(np.std(mbp_lv, ddof=1))
        r['MAP_late_CV'] = float(r['MAP_late_SD'] / np.mean(mbp_lv) * 100)
    else:
        r['MAP_late_SD'] = r['MAP_late_CV'] = np.nan

    # 术后30min（如果数据延续到手术后）
    post_s = int(opend_sec)
    post_e = int(opend_sec + 30 * 60)
    if post_e <= total_len:
        r['HR_post30_mean'] = float(np.nanmean(hr[post_s:post_e]))
        r['MAP_post30_mean'] = float(np.nanmean(mbp[post_s:post_e]))
        r['HR_post_rebound'] = float(r['HR_post30_mean'] - hr_sm)
        r['MAP_post_rebound'] = float(r['MAP_post30_mean'] - mbp_sm)
    else:
        r['HR_post30_mean'] = r['MAP_post30_mean'] = np.nan
        r['HR_post_rebound'] = r['MAP_post_rebound'] = np.nan

    return r


def process_bis(data, caseid, opstart_sec, opend_sec):
    """计算BIS指标"""
    bis = data[:, 4]  # BIS/BIS (index 4 in full track list)
    sqi = data[:, 5]  # BIS/SQI

    start = max(0, int(opstart_sec))
    end = min(len(bis), int(opend_sec))

    bis_op = bis[start:end].copy()
    sqi_op = sqi[start:end]

    # 过滤
    bis_op[sqi_op < 50] = np.nan
    bis_op[(bis_op < 0) | (bis_op > 100)] = np.nan

    bv = bis_op[~np.isnan(bis_op)]
    if len(bv) < 300:
        return None

    r = {'caseid': caseid}
    r['TWA_BIS'] = float(np.mean(bv))
    r['SD_BIS'] = float(np.std(bv, ddof=1))
    r['CV_BIS'] = float(r['SD_BIS'] / r['TWA_BIS'] * 100)
    r['ARV_BIS'] = float(np.mean(np.abs(np.diff(bv))))
    r['BIS_pct_in_range'] = float(np.sum((bv >= 40) & (bv <= 60)) / len(bv) * 100)
    r['BIS_nan_pct'] = float(np.sum(np.isnan(bis_op)) / len(bis_op) * 100)

    # AUT
    r['AUT_below40'] = float(np.sum(np.maximum(0, 40 - bv)) / 60)
    r['AUT_above60'] = float(np.sum(np.maximum(0, bv - 60)) / 60)

    return r


def process_nhd(data, caseid, opstart_sec, opend_sec):
    """计算NHD指数"""
    bis = data[:, 4]
    sqi = data[:, 5]
    hr  = data[:, 6]  # mapped to index 6 in nhd track list
    mbp = data[:, 7]  # mapped to index 7

    start = max(0, int(opstart_sec))
    end = min(len(bis), int(opend_sec))

    bis_op = bis[start:end].copy()
    sqi_op = sqi[start:end]
    hr_op = hr[start:end]
    mbp_op = mbp[start:end]

    bis_op[sqi_op < 50] = np.nan

    # 基线：手术开始后前30min
    bl_end = min(30 * 60, len(hr_op))
    hr_bl = np.nanmean(hr_op[:bl_end])
    mbp_bl = np.nanmean(mbp_op[:bl_end])

    if np.isnan(hr_bl) or np.isnan(mbp_bl) or hr_bl == 0:
        return None

    bis_ok = (bis_op >= 40) & (bis_op <= 60)
    hr_high = hr_op > hr_bl * 1.2
    mbp_high = mbp_op > mbp_bl * 1.2
    dissoc = bis_ok & (hr_high | mbp_high)

    total_valid = np.sum(~np.isnan(bis_op))
    if total_valid == 0:
        return None

    r = {'caseid': caseid}
    r['NHD_seconds'] = int(np.nansum(dissoc))
    r['NHD_pct'] = float(r['NHD_seconds'] / total_valid * 100)
    r['HR_baseline'] = float(hr_bl)
    r['MAP_baseline'] = float(mbp_bl)

    return r


# ============================================================
# 主处理循环
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--end', type=int, default=None, help='End index (exclusive)')
    args = parser.parse_args()

    print("=" * 70)
    print("  OIH Study - Batch Track Data Download")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 加载eligible列表
    eligible = pd.read_csv(OIH_DIR / "oih_eligible_caseids.csv")['caseid'].tolist()
    df_clinical = pd.read_csv(OIH_DIR / "oih_eligible_clinical.csv")
    clinical_map = df_clinical.set_index('caseid')

    # 加载已有进度
    done_ids = load_progress()
    results = load_existing_results()

    # 切片
    work_list = eligible[args.start:args.end]
    remaining = [c for c in work_list if c not in done_ids]

    print(f"  Total eligible: {len(eligible)}")
    print(f"  Work range: [{args.start}:{args.end or len(eligible)}] = {len(work_list)}")
    print(f"  Already done: {len(done_ids)}")
    print(f"  Remaining: {len(remaining)}")

    if not remaining:
        print("  Nothing to do!")
        return

    # 定义所有轨迹（一次性加载减少API调用）
    all_tracks = [
        'Orchestra/RFTN20_CE',   # 0
        'Orchestra/RFTN20_RATE', # 1
        'Orchestra/RFTN20_VOL',  # 2
        'Orchestra/RFTN20_CT',   # 3
        'BIS/BIS',               # 4
        'BIS/SQI',               # 5
        'Solar8000/HR',          # 6
        'Solar8000/ART_MBP',     # 7
        'Solar8000/NIBP_MBP',    # 8
    ]

    times = []
    errors = []
    n_total = len(remaining)

    for i, caseid in enumerate(remaining):
        t0 = time.time()

        try:
            # 获取临床信息
            if caseid not in clinical_map.index:
                continue
            info = clinical_map.loc[caseid]
            weight = float(info['weight']) if pd.notna(info.get('weight')) and info['weight'] > 0 else 60.0
            opdur_min = float(info['opdur']) if pd.notna(info.get('opdur')) else 120.0
            opstart = float(info['opstart']) if pd.notna(info.get('opstart')) else 0
            opend = float(info['opend']) if pd.notna(info.get('opend')) else opstart + opdur_min * 60
            anestart = float(info['anestart']) if pd.notna(info.get('anestart')) else 0

            # opstart/opend是相对于casestart=0的秒数
            # vitaldb.load_case返回的数据也是从casestart=0开始
            opstart_sec = opstart
            opend_sec = opend

            # 下载数据
            data = vitaldb.load_case(caseid, all_tracks, interval=1)
            if data is None or data.shape[0] < 60:
                errors.append((caseid, 'no data'))
                continue

            # === RFTN暴露 ===
            # 检查是否有RFTN数据（Ce列不全为NaN）
            ce_col = data[:, 0]
            if np.sum(~np.isnan(ce_col)) > 60:
                rftn_data = data[:, :4]  # CE, RATE, VOL, CT
                rftn_r = process_rftn(
                    np.column_stack([rftn_data, data[:, 4:]]),  # 不需要这样
                    caseid, weight, opdur_min, conc=20
                )
                # 直接用前4列
                rftn_r2 = {'caseid': caseid, 'rftn_conc': 20}

                vol = data[:, 2]
                vol_v = vol[~np.isnan(vol)]
                total_vol = (np.nanmax(vol) - np.nanmin(vol_v[vol_v >= 0])) if len(vol_v) > 1 and np.any(vol_v >= 0) else 0
                rftn_r2['RFTN_total_mcg'] = float(total_vol * 20)
                rftn_r2['RFTN_total_mcg_kg'] = rftn_r2['RFTN_total_mcg'] / weight
                rftn_r2['RFTN_mcg_kg_hr'] = rftn_r2['RFTN_total_mcg_kg'] / (opdur_min / 60) if opdur_min > 0 else np.nan

                ce_c = np.where(np.isnan(ce_col), 0, ce_col)
                rftn_r2['RFTN_AUC_Ce'] = float(np.trapezoid(ce_c, dx=1/60))

                rate = data[:, 1]
                rate_conv = rate * 20 / 60 / weight
                rftn_r2['RFTN_rate_mean'] = float(np.nanmean(rate_conv))
                rftn_r2['RFTN_rate_peak'] = float(np.nanmax(rate_conv)) if np.any(~np.isnan(rate_conv)) else np.nan

                ce_v = ce_col[~np.isnan(ce_col)]
                rftn_r2['RFTN_Ce_mean'] = float(np.mean(ce_v))
                rftn_r2['RFTN_Ce_peak'] = float(np.max(ce_v))
                rftn_r2['RFTN_Ce_median'] = float(np.median(ce_v))

                tail = ce_col[-60:] if len(ce_col) >= 60 else ce_col
                tail_v = tail[~np.isnan(tail)]
                rftn_r2['RFTN_Ce_at_end'] = float(np.mean(tail_v)) if len(tail_v) > 0 else np.nan

                if len(ce_v) > 60:
                    rftn_r2['RFTN_Ce_SD'] = float(np.std(ce_v, ddof=1))
                    rftn_r2['RFTN_Ce_CV'] = rftn_r2['RFTN_Ce_SD'] / rftn_r2['RFTN_Ce_mean'] * 100 if rftn_r2['RFTN_Ce_mean'] > 0 else np.nan
                    rftn_r2['RFTN_Ce_ARV'] = float(np.mean(np.abs(np.diff(ce_v))))
                else:
                    rftn_r2['RFTN_Ce_SD'] = rftn_r2['RFTN_Ce_CV'] = rftn_r2['RFTN_Ce_ARV'] = np.nan

                ct = data[:, 3]
                ct_v = ct[~np.isnan(ct)]
                rftn_r2['RFTN_Ct_changes'] = int(np.sum(np.abs(np.diff(ct_v)) > 0.05)) if len(ct_v) > 1 else 0

                n_last = min(1800, len(ce_col))
                last_ce = ce_col[-n_last:]
                lv = last_ce[~np.isnan(last_ce)]
                rftn_r2['RFTN_taper_slope'] = float(np.polyfit(np.arange(len(lv)), lv, 1)[0] * 60) if len(lv) > 60 else np.nan

                rftn_r2['Time_Ce_above_4'] = float(np.sum(ce_col > 4) / 60)
                rftn_r2['Time_Ce_above_6'] = float(np.sum(ce_col > 6) / 60)
                rftn_r2['Time_Ce_above_8'] = float(np.sum(ce_col > 8) / 60)
                rftn_r2['RFTN_data_len'] = int(len(ce_col))
                rftn_r2['RFTN_nan_pct'] = float(np.sum(np.isnan(ce_col)) / len(ce_col) * 100)

                results['rftn'].append(rftn_r2)

            # === 血流动力学反弹 ===
            hr = data[:, 6]
            mbp_art = data[:, 7]
            mbp_nibp = data[:, 8]
            art_valid = np.sum(~np.isnan(mbp_art))
            nibp_valid = np.sum(~np.isnan(mbp_nibp))
            mbp = mbp_art if art_valid > nibp_valid else mbp_nibp
            mbp_src = 'ART' if art_valid > nibp_valid else 'NIBP'

            op_dur_sec = opend_sec - opstart_sec
            if op_dur_sec >= 90 * 60:
                total_len = len(hr)
                ss = max(0, min(int(opstart_sec + 30*60), total_len-1))
                se = max(0, min(int(opend_sec - 60*60), total_len-1))
                ls = max(0, min(int(opend_sec - 30*60), total_len-1))
                le = max(0, min(int(opend_sec), total_len-1))

                if ss < se and ls < le:
                    hr_sm = np.nanmean(hr[ss:se])
                    mbp_sm = np.nanmean(mbp[ss:se])
                    hr_lm = np.nanmean(hr[ls:le])
                    mbp_lm = np.nanmean(mbp[ls:le])

                    if not (np.isnan(hr_sm) or np.isnan(hr_lm)):
                        hemo_r = {
                            'caseid': caseid, 'mbp_source': mbp_src,
                            'HR_stable_mean': float(hr_sm), 'MAP_stable_mean': float(mbp_sm),
                            'HR_late_mean': float(hr_lm), 'MAP_late_mean': float(mbp_lm),
                            'HR_rebound': float(hr_lm - hr_sm),
                            'MAP_rebound': float(mbp_lm - mbp_sm),
                            'HR_rebound_pct': float((hr_lm - hr_sm) / hr_sm * 100) if hr_sm > 0 else np.nan,
                            'MAP_rebound_pct': float((mbp_lm - mbp_sm) / mbp_sm * 100) if mbp_sm > 0 else np.nan,
                        }

                        hr_late = hr[ls:le]
                        hlv = hr_late[~np.isnan(hr_late)]
                        if len(hlv) > 30:
                            hemo_r['HR_late_SD'] = float(np.std(hlv, ddof=1))
                            hemo_r['HR_late_CV'] = float(hemo_r['HR_late_SD'] / np.mean(hlv) * 100)
                        else:
                            hemo_r['HR_late_SD'] = hemo_r['HR_late_CV'] = np.nan

                        mbp_late = mbp[ls:le]
                        mlv = mbp_late[~np.isnan(mbp_late)]
                        if len(mlv) > 30:
                            hemo_r['MAP_late_SD'] = float(np.std(mlv, ddof=1))
                            hemo_r['MAP_late_CV'] = float(hemo_r['MAP_late_SD'] / np.mean(mlv) * 100)
                        else:
                            hemo_r['MAP_late_SD'] = hemo_r['MAP_late_CV'] = np.nan

                        # 术后30min
                        ps = int(opend_sec)
                        pe = int(opend_sec + 30*60)
                        if pe <= total_len:
                            hemo_r['HR_post30_mean'] = float(np.nanmean(hr[ps:pe]))
                            hemo_r['MAP_post30_mean'] = float(np.nanmean(mbp[ps:pe]))
                            hemo_r['HR_post_rebound'] = float(hemo_r['HR_post30_mean'] - hr_sm)
                            hemo_r['MAP_post_rebound'] = float(hemo_r['MAP_post30_mean'] - mbp_sm)
                        else:
                            hemo_r['HR_post30_mean'] = hemo_r['MAP_post30_mean'] = np.nan
                            hemo_r['HR_post_rebound'] = hemo_r['MAP_post_rebound'] = np.nan

                        results['hemo'].append(hemo_r)

            # === BIS指标 ===
            bis = data[:, 4]
            sqi = data[:, 5]
            s_idx = max(0, int(opstart_sec))
            e_idx = min(len(bis), int(opend_sec))
            bis_op = bis[s_idx:e_idx].copy()
            sqi_op = sqi[s_idx:e_idx]
            bis_op[sqi_op < 50] = np.nan
            bis_op[(bis_op < 0) | (bis_op > 100)] = np.nan

            bv = bis_op[~np.isnan(bis_op)]
            if len(bv) >= 300:
                bis_r = {'caseid': caseid}
                bis_r['TWA_BIS'] = float(np.mean(bv))
                bis_r['SD_BIS'] = float(np.std(bv, ddof=1))
                bis_r['CV_BIS'] = float(bis_r['SD_BIS'] / bis_r['TWA_BIS'] * 100)
                bis_r['ARV_BIS'] = float(np.mean(np.abs(np.diff(bv))))
                bis_r['BIS_pct_in_range'] = float(np.sum((bv >= 40) & (bv <= 60)) / len(bv) * 100)
                bis_r['AUT_below40'] = float(np.sum(np.maximum(0, 40 - bv)) / 60)
                bis_r['AUT_above60'] = float(np.sum(np.maximum(0, bv - 60)) / 60)
                bis_r['BIS_nan_pct'] = float(np.sum(np.isnan(bis_op)) / len(bis_op) * 100)
                results['bis'].append(bis_r)

            # === NHD指数 ===
            hr_op = hr[s_idx:e_idx]
            mbp_op = mbp[s_idx:e_idx]
            bl_end = min(30 * 60, len(hr_op))
            hr_bl = np.nanmean(hr_op[:bl_end])
            mbp_bl = np.nanmean(mbp_op[:bl_end])

            if not (np.isnan(hr_bl) or np.isnan(mbp_bl) or hr_bl == 0):
                bis_ok = (bis_op >= 40) & (bis_op <= 60)
                hr_high = hr_op > hr_bl * 1.2
                mbp_high = mbp_op > mbp_bl * 1.2
                dissoc = bis_ok & (hr_high | mbp_high)
                tv = np.sum(~np.isnan(bis_op))
                if tv > 0:
                    nhd_r = {
                        'caseid': caseid,
                        'NHD_seconds': int(np.nansum(dissoc)),
                        'NHD_pct': float(np.nansum(dissoc) / tv * 100),
                        'HR_baseline': float(hr_bl),
                        'MAP_baseline': float(mbp_bl),
                    }
                    results['nhd'].append(nhd_r)

        except Exception as e:
            errors.append((caseid, str(e)))

        elapsed = time.time() - t0
        times.append(elapsed)

        # 进度输出
        if (i + 1) % 10 == 0 or i == 0:
            avg_t = np.mean(times[-50:])
            eta = avg_t * (n_total - i - 1)
            eta_str = str(timedelta(seconds=int(eta)))
            n_rftn = len(results['rftn'])
            n_hemo = len(results['hemo'])
            print(f"  [{i+1:4d}/{n_total}] caseid={caseid:5d} | {elapsed:.1f}s | "
                  f"avg={avg_t:.1f}s | ETA={eta_str} | "
                  f"rftn={n_rftn} hemo={n_hemo} err={len(errors)}")

        # 断点保存
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            save_results(results)
            progress = {
                'last_index': args.start + i,
                'last_caseid': caseid,
                'timestamp': datetime.now().isoformat(),
                'total_rftn': len(results['rftn']),
                'total_hemo': len(results['hemo']),
                'total_bis': len(results['bis']),
                'total_nhd': len(results['nhd']),
                'errors': len(errors),
            }
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(progress, f, indent=2)
            print(f"  >>> Checkpoint saved ({len(results['rftn'])} rftn, {len(results['hemo'])} hemo)")

    # 最终保存
    save_results(results)
    print("\n" + "=" * 70)
    print("  BATCH DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"  RFTN exposure: {len(results['rftn'])} cases")
    print(f"  Hemodynamic rebound: {len(results['hemo'])} cases")
    print(f"  BIS metrics: {len(results['bis'])} cases")
    print(f"  NHD index: {len(results['nhd'])} cases")
    print(f"  Errors: {len(errors)}")
    if errors:
        print(f"  First 10 errors: {errors[:10]}")

    # 保存错误日志
    if errors:
        with open(OIH_DIR / "oih_download_errors.json", 'w') as f:
            json.dump(errors, f, indent=2)


if __name__ == '__main__':
    main()
