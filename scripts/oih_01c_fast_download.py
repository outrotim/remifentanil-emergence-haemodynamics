#!/usr/bin/env python3
"""
=================================================================
OIH Study - Phase 1c: 高速异步并行下载 + 指标计算
=================================================================
使用 asyncio + aiohttp 直接并行下载 VitalDB track CSV，
绕过 vitaldb.load_case() 的串行瓶颈。

性能对比:
  vitaldb.load_case (串行)  → ~2.5 cases/min → 4443例 ≈ 30小时
  asyncio + aiohttp (30并发) → ~150 cases/min → 4443例 ≈ 30分钟

特性：
- 30路并发HTTP下载
- 分批50例处理 + 自动保存
- 断点续传（跳过已完成case）
- 错误容忍
- 与原脚本完全一致的指标计算

用法：
  python oih_01c_fast_download.py
=================================================================
"""

import os
import sys
import io
import json
import time
import asyncio
import warnings
import argparse
import numpy as np
import pandas as pd
import aiohttp
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================
# 配置
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
SHARED_DIR = PROJECT_DIR.parent / "shared_data"   # 共享数据(track_list等)
OIH_DIR = PROJECT_DIR / "data"
LOGS_DIR = PROJECT_DIR / "logs"

RFTN_FILE = OIH_DIR / "oih_rftn_exposure.csv"
HEMO_FILE = OIH_DIR / "oih_hemodynamic_rebound.csv"
BIS_FILE  = OIH_DIR / "oih_bis_metrics.csv"
NHD_FILE  = OIH_DIR / "oih_nhd_index.csv"

API_URL = "https://api.vitaldb.net"
MAX_CONCURRENT = 30       # 并发HTTP请求上限
BATCH_SIZE = 50           # 每批处理case数
SAVE_INTERVAL = 50        # 每N例保存CSV
INTERVAL = 1              # 采样间隔(秒)

# OIH需要的9个track
TRACK_NAMES = [
    'Orchestra/RFTN20_CE',    # 0
    'Orchestra/RFTN20_RATE',  # 1
    'Orchestra/RFTN20_VOL',   # 2
    'Orchestra/RFTN20_CT',    # 3
    'BIS/BIS',                # 4
    'BIS/SQI',                # 5
    'Solar8000/HR',           # 6
    'Solar8000/ART_MBP',      # 7
    'Solar8000/NIBP_MBP',     # 8
]


# ============================================================
# 异步下载核心
# ============================================================
async def download_single_track(session, tid, sem):
    """下载单个track CSV，返回 1D numpy 数组"""
    if tid is None:
        return None

    async with sem:
        url = f"{API_URL}/{tid}"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=90)) as resp:
                if resp.status != 200:
                    return None
                text = await resp.text()
        except Exception:
            return None

    if not text.strip():
        return None

    try:
        dtvals = pd.read_csv(io.StringIO(text), na_values='-nan(ind)', dtype=np.float32).values
    except Exception:
        return None

    if len(dtvals) == 0:
        return np.empty(0)

    # 转换为密集数组（1秒间隔）
    dtvals[:, 0] /= INTERVAL
    nsamp = int(np.nanmax(dtvals[:, 0])) + 1
    ret = np.full(nsamp, np.nan, dtype=np.float32)

    if np.isnan(dtvals[:, 0]).any():  # 波形数据
        if nsamp != len(dtvals):
            ret = np.take(dtvals[:, 1], np.linspace(0, len(dtvals) - 1, nsamp).astype(np.int64))
        else:
            ret = dtvals[:, 1]
    else:  # 数值数据
        for idx, val in dtvals:
            if not np.isnan(idx):
                ret[int(idx)] = val

    return ret


async def download_case_tracks(session, tids, sem):
    """并行下载一个case的所有track，组装为2D numpy数组"""
    tasks = [download_single_track(session, tid, sem) for tid in tids]
    results = await asyncio.gather(*tasks)

    # 确定最长track长度
    maxlen = 0
    for trk in results:
        if trk is not None and len(trk) > maxlen:
            maxlen = len(trk)

    if maxlen == 0:
        return None

    # 组装 (n_timepoints, n_tracks)
    data = np.full((maxlen, len(tids)), np.nan, dtype=np.float32)
    for i, trk in enumerate(results):
        if trk is not None and len(trk) > 0:
            data[:len(trk), i] = trk

    return data


# ============================================================
# 指标计算函数（与原脚本完全一致）
# ============================================================
def compute_rftn(data, caseid, weight, opdur_min, conc=20):
    """计算瑞芬太尼暴露指标 (16维)"""
    ce = data[:, 0]
    rate = data[:, 1]
    vol = data[:, 2]
    ct = data[:, 3]

    ce_v = ce[~np.isnan(ce)]
    if len(ce_v) <= 60:
        return None

    r = {'caseid': caseid, 'rftn_conc': conc}

    # 总量
    vol_v = vol[~np.isnan(vol)]
    if len(vol_v) > 1 and np.any(vol_v >= 0):
        total_vol = np.nanmax(vol) - np.nanmin(vol_v[vol_v >= 0])
    else:
        total_vol = 0
    r['RFTN_total_mcg'] = float(total_vol * conc)
    r['RFTN_total_mcg_kg'] = r['RFTN_total_mcg'] / weight if weight > 0 else np.nan
    r['RFTN_mcg_kg_hr'] = r['RFTN_total_mcg_kg'] / (opdur_min / 60) if opdur_min > 0 else np.nan

    # Ce AUC
    ce_c = np.where(np.isnan(ce), 0, ce)
    r['RFTN_AUC_Ce'] = float(np.trapezoid(ce_c, dx=1/60))

    # 速率
    if weight > 0:
        rate_conv = rate * conc / 60 / weight
        r['RFTN_rate_mean'] = float(np.nanmean(rate_conv))
        r['RFTN_rate_peak'] = float(np.nanmax(rate_conv)) if np.any(~np.isnan(rate_conv)) else np.nan
    else:
        r['RFTN_rate_mean'] = r['RFTN_rate_peak'] = np.nan

    r['RFTN_Ce_mean'] = float(np.mean(ce_v))
    r['RFTN_Ce_peak'] = float(np.max(ce_v))
    r['RFTN_Ce_median'] = float(np.median(ce_v))

    # 术末Ce
    tail = ce[-60:] if len(ce) >= 60 else ce
    tail_v = tail[~np.isnan(tail)]
    r['RFTN_Ce_at_end'] = float(np.mean(tail_v)) if len(tail_v) > 0 else np.nan

    # 波动
    r['RFTN_Ce_SD'] = float(np.std(ce_v, ddof=1))
    r['RFTN_Ce_CV'] = r['RFTN_Ce_SD'] / r['RFTN_Ce_mean'] * 100 if r['RFTN_Ce_mean'] > 0 else np.nan
    r['RFTN_Ce_ARV'] = float(np.mean(np.abs(np.diff(ce_v))))

    # Ct调整次数
    ct_v = ct[~np.isnan(ct)]
    r['RFTN_Ct_changes'] = int(np.sum(np.abs(np.diff(ct_v)) > 0.05)) if len(ct_v) > 1 else 0

    # 术末30min斜率
    n_last = min(1800, len(ce))
    last_ce = ce[-n_last:]
    lv = last_ce[~np.isnan(last_ce)]
    r['RFTN_taper_slope'] = float(np.polyfit(np.arange(len(lv)), lv, 1)[0] * 60) if len(lv) > 60 else np.nan

    # 高浓度暴露时间(分钟)
    r['Time_Ce_above_4'] = float(np.sum(ce > 4) / 60)
    r['Time_Ce_above_6'] = float(np.sum(ce > 6) / 60)
    r['Time_Ce_above_8'] = float(np.sum(ce > 8) / 60)
    r['RFTN_data_len'] = int(len(ce))
    r['RFTN_nan_pct'] = float(np.sum(np.isnan(ce)) / len(ce) * 100)

    return r


def compute_hemo(data, caseid, opstart_sec, opend_sec):
    """计算血流动力学反弹指标"""
    hr = data[:, 6]
    mbp_art = data[:, 7]
    mbp_nibp = data[:, 8]

    art_valid = np.sum(~np.isnan(mbp_art))
    nibp_valid = np.sum(~np.isnan(mbp_nibp))
    mbp = mbp_art if art_valid > nibp_valid else mbp_nibp
    mbp_src = 'ART' if art_valid > nibp_valid else 'NIBP'

    op_dur_sec = opend_sec - opstart_sec
    if op_dur_sec < 90 * 60:
        return None

    total_len = len(hr)
    ss = max(0, min(int(opstart_sec + 30*60), total_len-1))
    se = max(0, min(int(opend_sec - 60*60), total_len-1))
    ls = max(0, min(int(opend_sec - 30*60), total_len-1))
    le = max(0, min(int(opend_sec), total_len-1))

    if ss >= se or ls >= le:
        return None

    hr_sm = np.nanmean(hr[ss:se])
    mbp_sm = np.nanmean(mbp[ss:se])
    hr_lm = np.nanmean(hr[ls:le])
    mbp_lm = np.nanmean(mbp[ls:le])

    if np.isnan(hr_sm) or np.isnan(hr_lm):
        return None

    r = {
        'caseid': caseid, 'mbp_source': mbp_src,
        'HR_stable_mean': float(hr_sm), 'MAP_stable_mean': float(mbp_sm),
        'HR_late_mean': float(hr_lm), 'MAP_late_mean': float(mbp_lm),
        'HR_rebound': float(hr_lm - hr_sm),
        'MAP_rebound': float(mbp_lm - mbp_sm),
        'HR_rebound_pct': float((hr_lm - hr_sm) / hr_sm * 100) if hr_sm > 0 else np.nan,
        'MAP_rebound_pct': float((mbp_lm - mbp_sm) / mbp_sm * 100) if mbp_sm > 0 else np.nan,
    }

    # 晚期变异度
    hr_late = hr[ls:le]
    hlv = hr_late[~np.isnan(hr_late)]
    if len(hlv) > 30:
        r['HR_late_SD'] = float(np.std(hlv, ddof=1))
        r['HR_late_CV'] = float(r['HR_late_SD'] / np.mean(hlv) * 100)
    else:
        r['HR_late_SD'] = r['HR_late_CV'] = np.nan

    mbp_late = mbp[ls:le]
    mlv = mbp_late[~np.isnan(mbp_late)]
    if len(mlv) > 30:
        r['MAP_late_SD'] = float(np.std(mlv, ddof=1))
        r['MAP_late_CV'] = float(r['MAP_late_SD'] / np.mean(mlv) * 100)
    else:
        r['MAP_late_SD'] = r['MAP_late_CV'] = np.nan

    # 术后30min
    ps = int(opend_sec)
    pe = int(opend_sec + 30*60)
    if pe <= total_len:
        r['HR_post30_mean'] = float(np.nanmean(hr[ps:pe]))
        r['MAP_post30_mean'] = float(np.nanmean(mbp[ps:pe]))
        r['HR_post_rebound'] = float(r['HR_post30_mean'] - hr_sm)
        r['MAP_post_rebound'] = float(r['MAP_post30_mean'] - mbp_sm)
    else:
        r['HR_post30_mean'] = r['MAP_post30_mean'] = np.nan
        r['HR_post_rebound'] = r['MAP_post_rebound'] = np.nan

    return r


def compute_bis(data, caseid, opstart_sec, opend_sec):
    """计算BIS指标"""
    bis = data[:, 4]
    sqi = data[:, 5]

    s_idx = max(0, int(opstart_sec))
    e_idx = min(len(bis), int(opend_sec))
    bis_op = bis[s_idx:e_idx].copy()
    sqi_op = sqi[s_idx:e_idx]

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
    r['AUT_below40'] = float(np.sum(np.maximum(0, 40 - bv)) / 60)
    r['AUT_above60'] = float(np.sum(np.maximum(0, bv - 60)) / 60)
    r['BIS_nan_pct'] = float(np.sum(np.isnan(bis_op)) / len(bis_op) * 100)

    return r


def compute_nhd(data, caseid, opstart_sec, opend_sec):
    """计算NHD (Nociception-Hypnosis Dissociation) 指数"""
    bis = data[:, 4]
    sqi = data[:, 5]
    hr = data[:, 6]
    # MBP: 优先 ART, 回退 NIBP
    mbp_art = data[:, 7]
    mbp_nibp = data[:, 8]
    mbp = mbp_art if np.sum(~np.isnan(mbp_art)) > np.sum(~np.isnan(mbp_nibp)) else mbp_nibp

    s_idx = max(0, int(opstart_sec))
    e_idx = min(len(bis), int(opend_sec))

    bis_op = bis[s_idx:e_idx].copy()
    sqi_op = sqi[s_idx:e_idx]
    hr_op = hr[s_idx:e_idx]
    mbp_op = mbp[s_idx:e_idx]

    bis_op[sqi_op < 50] = np.nan

    # 基线
    bl_end = min(30 * 60, len(hr_op))
    hr_bl = np.nanmean(hr_op[:bl_end])
    mbp_bl = np.nanmean(mbp_op[:bl_end])

    if np.isnan(hr_bl) or np.isnan(mbp_bl) or hr_bl == 0:
        return None

    bis_ok = (bis_op >= 40) & (bis_op <= 60)
    hr_high = hr_op > hr_bl * 1.2
    mbp_high = mbp_op > mbp_bl * 1.2
    dissoc = bis_ok & (hr_high | mbp_high)

    tv = np.sum(~np.isnan(bis_op))
    if tv == 0:
        return None

    return {
        'caseid': caseid,
        'NHD_seconds': int(np.nansum(dissoc)),
        'NHD_pct': float(np.nansum(dissoc) / tv * 100),
        'HR_baseline': float(hr_bl),
        'MAP_baseline': float(mbp_bl),
    }


# ============================================================
# 主异步流程
# ============================================================
async def process_batch(session, sem, batch_cases, tid_map, clinical_map):
    """处理一批case：并行下载 + 计算指标"""
    batch_results = {'rftn': [], 'hemo': [], 'bis': [], 'nhd': []}
    errors = []

    # 并行下载这一批所有case
    download_tasks = []
    valid_cases = []
    for caseid in batch_cases:
        tids = tid_map.get(caseid)
        if tids is None:
            errors.append((caseid, 'no TID mapping'))
            continue
        download_tasks.append(download_case_tracks(session, tids, sem))
        valid_cases.append(caseid)

    # 等待所有下载完成
    all_data = await asyncio.gather(*download_tasks, return_exceptions=True)

    # 逐例计算指标
    for caseid, data_or_err in zip(valid_cases, all_data):
        if isinstance(data_or_err, Exception):
            errors.append((caseid, str(data_or_err)))
            continue
        if data_or_err is None or data_or_err.shape[0] < 60:
            errors.append((caseid, 'no data or too short'))
            continue

        data = data_or_err

        try:
            info = clinical_map.loc[caseid]
            weight = float(info['weight']) if pd.notna(info.get('weight')) and info['weight'] > 0 else 60.0
            opdur_min = float(info['opdur']) if pd.notna(info.get('opdur')) else 120.0
            opstart_sec = float(info['opstart']) if pd.notna(info.get('opstart')) else 0
            opend_sec = float(info['opend']) if pd.notna(info.get('opend')) else opstart_sec + opdur_min * 60

            # RFTN
            rftn_r = compute_rftn(data, caseid, weight, opdur_min)
            if rftn_r:
                batch_results['rftn'].append(rftn_r)

            # 血流动力学反弹
            hemo_r = compute_hemo(data, caseid, opstart_sec, opend_sec)
            if hemo_r:
                batch_results['hemo'].append(hemo_r)

            # BIS
            bis_r = compute_bis(data, caseid, opstart_sec, opend_sec)
            if bis_r:
                batch_results['bis'].append(bis_r)

            # NHD
            nhd_r = compute_nhd(data, caseid, opstart_sec, opend_sec)
            if nhd_r:
                batch_results['nhd'].append(nhd_r)

        except Exception as e:
            errors.append((caseid, str(e)))

    return batch_results, errors


def save_results(all_results):
    """保存所有结果到CSV"""
    for key, filepath in [('rftn', RFTN_FILE), ('hemo', HEMO_FILE),
                          ('bis', BIS_FILE), ('nhd', NHD_FILE)]:
        if all_results[key]:
            df = pd.DataFrame(all_results[key])
            df.to_csv(filepath, index=False)


async def main():
    parser = argparse.ArgumentParser(description='OIH Study - 高速异步下载')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--concurrency', type=int, default=MAX_CONCURRENT)
    args = parser.parse_args()

    print("=" * 70)
    print("  OIH Study - 高速异步并行下载 (asyncio + aiohttp)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  并发数: {args.concurrency}")
    print("=" * 70)

    # 1. 加载队列
    eligible = pd.read_csv(OIH_DIR / "oih_eligible_caseids.csv")['caseid'].tolist()
    df_clinical = pd.read_csv(OIH_DIR / "oih_eligible_clinical.csv")
    clinical_map = df_clinical.set_index('caseid')

    work_list = eligible[args.start:args.end]
    print(f"  总队列: {len(eligible)} 例")
    print(f"  工作范围: [{args.start}:{args.end or len(eligible)}] = {len(work_list)} 例")

    # 2. 预计算 TID 映射
    print("  [Step 1] 预计算 TID 映射...")
    t0_map = time.time()
    df_trks = pd.read_csv(SHARED_DIR / "track_list.csv")

    tid_map = {}
    for caseid in work_list:
        case_trks = df_trks[df_trks['caseid'] == caseid]
        tids = []
        for tname in TRACK_NAMES:
            matches = case_trks.loc[case_trks['tname'] == tname, 'tid'].values
            tids.append(str(matches[0]) if len(matches) > 0 else None)
        tid_map[caseid] = tids

    print(f"           完成! {len(tid_map)} 例, 耗时 {time.time()-t0_map:.1f}s")

    # 3. 检查断点续传（读取已有结果）
    all_results = {'rftn': [], 'hemo': [], 'bis': [], 'nhd': []}
    done_ids = set()

    for key, filepath in [('rftn', RFTN_FILE), ('hemo', HEMO_FILE),
                          ('bis', BIS_FILE), ('nhd', NHD_FILE)]:
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                all_results[key] = df.to_dict('records')
                if key == 'rftn':  # 以RFTN为主判断已完成
                    done_ids = set(df['caseid'].tolist())
            except Exception:
                pass

    remaining = [c for c in work_list if c not in done_ids]
    print(f"  已完成: {len(done_ids)} 例")
    print(f"  待处理: {len(remaining)} 例")

    if not remaining:
        print("  没有需要处理的case!")
        return

    # 4. 开始异步下载
    print(f"\n  [Step 2] 开始异步并行下载...")
    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency, limit_per_host=args.concurrency)

    total_done = len(done_ids)
    total_target = len(work_list)
    total_errors = []
    start_time = time.time()

    async with aiohttp.ClientSession(connector=connector) as session:
        for batch_start in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[batch_start:batch_start + BATCH_SIZE]
            batch_t0 = time.time()

            batch_results, batch_errors = await process_batch(
                session, sem, batch, tid_map, clinical_map
            )

            # 合并结果
            for key in all_results:
                all_results[key].extend(batch_results[key])
            total_errors.extend(batch_errors)
            total_done += len(batch)

            # 保存
            if (batch_start + BATCH_SIZE) % SAVE_INTERVAL == 0 or batch_start + BATCH_SIZE >= len(remaining):
                save_results(all_results)

            # 进度
            elapsed = time.time() - start_time
            rate = (total_done - len(done_ids)) / elapsed if elapsed > 0 else 0
            remaining_n = total_target - total_done
            eta = remaining_n / rate if rate > 0 else 0
            eta_str = str(timedelta(seconds=int(eta)))

            batch_time = time.time() - batch_t0
            n_rftn = len(all_results['rftn'])
            n_hemo = len(all_results['hemo'])

            print(
                f"  [{total_done:5d}/{total_target}] "
                f"batch={len(batch)} in {batch_time:.1f}s | "
                f"rate={rate:.1f}/s ({rate*60:.0f}/min) | "
                f"ETA={eta_str} | "
                f"rftn={n_rftn} hemo={n_hemo} err={len(total_errors)}"
            )

    # 最终保存
    save_results(all_results)

    # 5. 完成报告
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  下载完成!")
    print("=" * 70)
    print(f"  总耗时: {elapsed/60:.1f} 分钟")
    print(f"  处理速率: {(total_done - len(done_ids.intersection(set(work_list))))/elapsed:.1f} cases/s "
          f"({(total_done - len(done_ids.intersection(set(work_list))))/elapsed*60:.0f} cases/min)")
    print(f"  RFTN暴露: {len(all_results['rftn'])} 例")
    print(f"  血流动力学反弹: {len(all_results['hemo'])} 例")
    print(f"  BIS指标: {len(all_results['bis'])} 例")
    print(f"  NHD指数: {len(all_results['nhd'])} 例")
    print(f"  错误: {len(total_errors)} 例")

    if total_errors:
        err_log = OIH_DIR / "oih_download_errors.json"
        with open(err_log, 'w') as f:
            json.dump([{'caseid': c, 'error': e} for c, e in total_errors], f, indent=2)
        print(f"  错误详情: {err_log}")

    # 保存下载日志
    log = {
        'timestamp': datetime.now().isoformat(),
        'total_cases': total_target,
        'processed': total_done,
        'rftn': len(all_results['rftn']),
        'hemo': len(all_results['hemo']),
        'bis': len(all_results['bis']),
        'nhd': len(all_results['nhd']),
        'errors': len(total_errors),
        'elapsed_min': round(elapsed / 60, 1),
        'rate_per_min': round((total_done - len(done_ids)) / elapsed * 60, 1) if elapsed > 0 else 0,
    }
    with open(OIH_DIR / "oih_download_log.json", 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\n  输出文件:")
    print(f"    {RFTN_FILE}")
    print(f"    {HEMO_FILE}")
    print(f"    {BIS_FILE}")
    print(f"    {NHD_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
