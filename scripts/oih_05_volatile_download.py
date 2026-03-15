#!/usr/bin/env python3
"""
=================================================================
OIH Study - Phase 5: 挥发性麻醉药数据下载 + TIVA/Balanced分类
=================================================================
下载 Primus/MAC, Primus/EXP_SEVO, Primus/EXP_DES 数据，
计算挥发性麻醉药暴露指标，分类 TIVA vs balanced anesthesia。

用法：
  python oih_05_volatile_download.py
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
SHARED_DIR = PROJECT_DIR.parent / "shared_data"
OIH_DIR = PROJECT_DIR / "data"

VOLATILE_FILE = OIH_DIR / "oih_volatile_data.csv"

API_URL = "https://api.vitaldb.net"
MAX_CONCURRENT = 30
BATCH_SIZE = 50
SAVE_INTERVAL = 50
INTERVAL = 1  # 采样间隔(秒)

# 挥发性麻醉药需要的3个track
VOLATILE_TRACKS = [
    'Primus/MAC',       # 0 - MAC值 (综合MAC)
    'Primus/EXP_SEVO',  # 1 - 呼出七氟烷浓度 (%)
    'Primus/EXP_DES',   # 2 - 呼出地氟烷浓度 (%)
]


# ============================================================
# 异步下载核心 (与 oih_01c_fast_download.py 一致)
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
    """并行下载一个case的所有track"""
    tasks = [download_single_track(session, tid, sem) for tid in tids]
    results = await asyncio.gather(*tasks)

    maxlen = 0
    for trk in results:
        if trk is not None and len(trk) > maxlen:
            maxlen = len(trk)

    if maxlen == 0:
        return None

    data = np.full((maxlen, len(tids)), np.nan, dtype=np.float32)
    for i, trk in enumerate(results):
        if trk is not None and len(trk) > 0:
            data[:len(trk), i] = trk

    return data


# ============================================================
# 挥发性麻醉药指标计算
# ============================================================
def compute_volatile(data, caseid, opstart_sec, opend_sec, opdur_min):
    """
    计算挥发性麻醉药暴露指标

    data columns:
      0 = Primus/MAC
      1 = Primus/EXP_SEVO (%)
      2 = Primus/EXP_DES (%)

    Returns dict with volatile metrics, or None if no valid data.
    """
    mac = data[:, 0]
    sevo = data[:, 1]
    des = data[:, 2]

    # 限制到术中时段
    s_idx = max(0, int(opstart_sec))
    e_idx = min(len(mac), int(opend_sec))

    if e_idx <= s_idx:
        return None

    mac_op = mac[s_idx:e_idx]
    sevo_op = sevo[s_idx:e_idx] if e_idx <= len(sevo) else np.full(e_idx - s_idx, np.nan)
    des_op = des[s_idx:e_idx] if e_idx <= len(des) else np.full(e_idx - s_idx, np.nan)

    # MAC有效数据
    mac_valid = mac_op[~np.isnan(mac_op)]
    sevo_valid = sevo_op[~np.isnan(sevo_op)]
    des_valid = des_op[~np.isnan(des_op)]

    r = {'caseid': caseid}

    # ------ MAC指标 ------
    if len(mac_valid) > 0:
        r['MAC_data_points'] = int(len(mac_valid))
        r['MAC_coverage_pct'] = float(len(mac_valid) / len(mac_op) * 100)
        r['MAC_mean'] = float(np.mean(mac_valid))
        r['MAC_median'] = float(np.median(mac_valid))
        r['MAC_peak'] = float(np.max(mac_valid))
        r['MAC_SD'] = float(np.std(mac_valid, ddof=1)) if len(mac_valid) > 1 else 0.0
        r['MAC_AUC'] = float(np.trapezoid(np.where(np.isnan(mac_op), 0, mac_op), dx=1/60))  # MAC·min

        # 有效挥发药暴露时间 (MAC > 0.3 的时间，分钟)
        r['time_MAC_above_0_3'] = float(np.sum(mac_valid > 0.3) / 60)
        r['time_MAC_above_0_5'] = float(np.sum(mac_valid > 0.5) / 60)
        r['time_MAC_above_0_8'] = float(np.sum(mac_valid > 0.8) / 60)

        # MAC > 0.3 持续时间占手术时间的比例
        if opdur_min > 0:
            r['MAC_exposure_fraction'] = float(r['time_MAC_above_0_3'] / opdur_min)
        else:
            r['MAC_exposure_fraction'] = np.nan
    else:
        r['MAC_data_points'] = 0
        r['MAC_coverage_pct'] = 0.0
        r['MAC_mean'] = np.nan
        r['MAC_median'] = np.nan
        r['MAC_peak'] = np.nan
        r['MAC_SD'] = np.nan
        r['MAC_AUC'] = np.nan
        r['time_MAC_above_0_3'] = 0.0
        r['time_MAC_above_0_5'] = 0.0
        r['time_MAC_above_0_8'] = 0.0
        r['MAC_exposure_fraction'] = 0.0

    # ------ 七氟烷指标 ------
    if len(sevo_valid) > 0:
        r['SEVO_mean'] = float(np.mean(sevo_valid))
        r['SEVO_peak'] = float(np.max(sevo_valid))
        r['SEVO_time_above_0_5'] = float(np.sum(sevo_valid > 0.5) / 60)  # 分钟
        r['has_sevo'] = True
    else:
        r['SEVO_mean'] = 0.0
        r['SEVO_peak'] = 0.0
        r['SEVO_time_above_0_5'] = 0.0
        r['has_sevo'] = False

    # ------ 地氟烷指标 ------
    if len(des_valid) > 0:
        r['DES_mean'] = float(np.mean(des_valid))
        r['DES_peak'] = float(np.max(des_valid))
        r['DES_time_above_1_0'] = float(np.sum(des_valid > 1.0) / 60)  # 分钟
        r['has_des'] = True
    else:
        r['DES_mean'] = 0.0
        r['DES_peak'] = 0.0
        r['DES_time_above_1_0'] = 0.0
        r['has_des'] = False

    # ------ TIVA vs Balanced 分类 ------
    # 标准：MAC > 0.3 持续 > 15分钟 → balanced anesthesia
    #        否则 → TIVA (total intravenous anesthesia)
    if r['time_MAC_above_0_3'] > 15:
        r['anes_type'] = 'balanced'
    else:
        r['anes_type'] = 'TIVA'

    # 更精细的分类
    if r['has_sevo'] and r['SEVO_time_above_0_5'] > 15:
        r['volatile_agent'] = 'sevoflurane'
    elif r['has_des'] and r['DES_time_above_1_0'] > 15:
        r['volatile_agent'] = 'desflurane'
    elif r['anes_type'] == 'balanced':
        r['volatile_agent'] = 'other/unknown'
    else:
        r['volatile_agent'] = 'none'

    return r


# ============================================================
# 主异步流程
# ============================================================
async def process_batch(session, sem, batch_cases, tid_map, clinical_map):
    """处理一批case"""
    batch_results = []
    errors = []

    download_tasks = []
    valid_cases = []
    for caseid in batch_cases:
        tids = tid_map.get(caseid)
        if tids is None:
            errors.append((caseid, 'no TID mapping'))
            continue
        download_tasks.append(download_case_tracks(session, tids, sem))
        valid_cases.append(caseid)

    all_data = await asyncio.gather(*download_tasks, return_exceptions=True)

    for caseid, data_or_err in zip(valid_cases, all_data):
        if isinstance(data_or_err, Exception):
            errors.append((caseid, str(data_or_err)))
            continue
        if data_or_err is None:
            # 没有MAC数据 → 标记为TIVA (没有挥发药)
            batch_results.append({
                'caseid': caseid,
                'MAC_data_points': 0, 'MAC_coverage_pct': 0.0,
                'MAC_mean': 0.0, 'MAC_median': 0.0, 'MAC_peak': 0.0,
                'MAC_SD': 0.0, 'MAC_AUC': 0.0,
                'time_MAC_above_0_3': 0.0, 'time_MAC_above_0_5': 0.0,
                'time_MAC_above_0_8': 0.0, 'MAC_exposure_fraction': 0.0,
                'SEVO_mean': 0.0, 'SEVO_peak': 0.0, 'SEVO_time_above_0_5': 0.0,
                'has_sevo': False,
                'DES_mean': 0.0, 'DES_peak': 0.0, 'DES_time_above_1_0': 0.0,
                'has_des': False,
                'anes_type': 'TIVA', 'volatile_agent': 'none',
            })
            continue

        try:
            info = clinical_map.loc[caseid]
            opdur_min = float(info['opdur']) if pd.notna(info.get('opdur')) else 120.0
            opstart_sec = float(info['opstart']) if pd.notna(info.get('opstart')) else 0
            opend_sec = float(info['opend']) if pd.notna(info.get('opend')) else opstart_sec + opdur_min * 60

            result = compute_volatile(data_or_err, caseid, opstart_sec, opend_sec, opdur_min)
            if result:
                batch_results.append(result)
            else:
                # 没有术中数据 → TIVA
                batch_results.append({
                    'caseid': caseid,
                    'MAC_data_points': 0, 'MAC_coverage_pct': 0.0,
                    'MAC_mean': 0.0, 'MAC_median': 0.0, 'MAC_peak': 0.0,
                    'MAC_SD': 0.0, 'MAC_AUC': 0.0,
                    'time_MAC_above_0_3': 0.0, 'time_MAC_above_0_5': 0.0,
                    'time_MAC_above_0_8': 0.0, 'MAC_exposure_fraction': 0.0,
                    'SEVO_mean': 0.0, 'SEVO_peak': 0.0, 'SEVO_time_above_0_5': 0.0,
                    'has_sevo': False,
                    'DES_mean': 0.0, 'DES_peak': 0.0, 'DES_time_above_1_0': 0.0,
                    'has_des': False,
                    'anes_type': 'TIVA', 'volatile_agent': 'none',
                })
        except Exception as e:
            errors.append((caseid, str(e)))

    return batch_results, errors


async def main():
    parser = argparse.ArgumentParser(description='OIH Study - 挥发性麻醉药数据下载')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--concurrency', type=int, default=MAX_CONCURRENT)
    args = parser.parse_args()

    print("=" * 70)
    print("  OIH Study - 挥发性麻醉药数据下载 (Primus/MAC + EXP_SEVO + EXP_DES)")
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("  并发数:", args.concurrency)
    print("=" * 70)

    # 1. 加载队列
    eligible = pd.read_csv(OIH_DIR / "oih_eligible_caseids.csv")['caseid'].tolist()
    df_clinical = pd.read_csv(OIH_DIR / "oih_eligible_clinical.csv")
    clinical_map = df_clinical.set_index('caseid')

    work_list = eligible[args.start:args.end]
    print(f"  总队列: {len(eligible)} 例")
    print(f"  工作范围: [{args.start}:{args.end or len(eligible)}] = {len(work_list)} 例")

    # 2. 预计算 TID 映射
    print("  [Step 1] 预计算 TID 映射 (MAC/SEVO/DES)...")
    t0_map = time.time()
    df_trks = pd.read_csv(SHARED_DIR / "track_list.csv")

    tid_map = {}
    track_stats = {t: 0 for t in VOLATILE_TRACKS}

    for caseid in work_list:
        case_trks = df_trks[df_trks['caseid'] == caseid]
        tids = []
        has_any = False
        for tname in VOLATILE_TRACKS:
            matches = case_trks.loc[case_trks['tname'] == tname, 'tid'].values
            if len(matches) > 0:
                tids.append(str(matches[0]))
                track_stats[tname] += 1
                has_any = True
            else:
                tids.append(None)
        tid_map[caseid] = tids

    print(f"           完成! {len(tid_map)} 例, 耗时 {time.time()-t0_map:.1f}s")
    for tname, cnt in track_stats.items():
        print(f"           {tname}: {cnt} 例有数据 ({cnt/len(work_list)*100:.1f}%)")

    # 3. 检查断点续传
    all_results = []
    done_ids = set()

    if VOLATILE_FILE.exists():
        try:
            df_done = pd.read_csv(VOLATILE_FILE)
            all_results = df_done.to_dict('records')
            done_ids = set(df_done['caseid'].tolist())
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

            all_results.extend(batch_results)
            total_errors.extend(batch_errors)
            total_done += len(batch)

            # 保存
            if (batch_start + BATCH_SIZE) % SAVE_INTERVAL == 0 or batch_start + BATCH_SIZE >= len(remaining):
                pd.DataFrame(all_results).to_csv(VOLATILE_FILE, index=False)

            # 进度
            elapsed = time.time() - start_time
            rate = (total_done - len(done_ids)) / elapsed if elapsed > 0 else 0
            remaining_n = total_target - total_done
            eta = remaining_n / rate if rate > 0 else 0
            eta_str = str(timedelta(seconds=int(eta)))

            batch_time = time.time() - batch_t0

            print(
                f"  [{total_done:5d}/{total_target}] "
                f"batch={len(batch)} in {batch_time:.1f}s | "
                f"rate={rate:.1f}/s ({rate*60:.0f}/min) | "
                f"ETA={eta_str} | "
                f"results={len(all_results)} err={len(total_errors)}"
            )

    # 最终保存
    df_final = pd.DataFrame(all_results)
    df_final.to_csv(VOLATILE_FILE, index=False)

    # 5. 完成报告
    elapsed = time.time() - start_time
    n_tiva = sum(1 for r in all_results if r.get('anes_type') == 'TIVA')
    n_balanced = sum(1 for r in all_results if r.get('anes_type') == 'balanced')
    n_sevo = sum(1 for r in all_results if r.get('volatile_agent') == 'sevoflurane')
    n_des = sum(1 for r in all_results if r.get('volatile_agent') == 'desflurane')

    print("\n" + "=" * 70)
    print("  下载完成!")
    print("=" * 70)
    print(f"  总耗时: {elapsed/60:.1f} 分钟")
    processed = total_done - len(done_ids.intersection(set(work_list)))
    if elapsed > 0 and processed > 0:
        print(f"  处理速率: {processed/elapsed:.1f} cases/s ({processed/elapsed*60:.0f} cases/min)")
    print(f"  总结果: {len(all_results)} 例")
    print(f"  错误: {len(total_errors)} 例")
    print()
    print(f"  === 麻醉类型分布 ===")
    print(f"  TIVA:      {n_tiva} ({n_tiva/len(all_results)*100:.1f}%)")
    print(f"  Balanced:  {n_balanced} ({n_balanced/len(all_results)*100:.1f}%)")
    print(f"    七氟烷:  {n_sevo}")
    print(f"    地氟烷:  {n_des}")
    print()

    # MAC统计
    mac_means = [r['MAC_mean'] for r in all_results if r.get('MAC_mean', 0) > 0]
    if mac_means:
        print(f"  === MAC统计 (n={len(mac_means)}) ===")
        print(f"  Mean MAC: {np.mean(mac_means):.3f} ± {np.std(mac_means):.3f}")
        print(f"  Median MAC: {np.median(mac_means):.3f}")
        print(f"  Peak MAC range: {np.min([r['MAC_peak'] for r in all_results if r.get('MAC_peak', 0) > 0]):.2f} - "
              f"{np.max([r['MAC_peak'] for r in all_results if r.get('MAC_peak', 0) > 0]):.2f}")

    if total_errors:
        err_log = OIH_DIR / "oih_volatile_errors.json"
        with open(err_log, 'w') as f:
            json.dump([{'caseid': c, 'error': e} for c, e in total_errors], f, indent=2)
        print(f"\n  错误详情: {err_log}")

    # 保存日志
    log = {
        'timestamp': datetime.now().isoformat(),
        'total_cases': total_target,
        'processed': len(all_results),
        'TIVA': n_tiva,
        'balanced': n_balanced,
        'sevoflurane': n_sevo,
        'desflurane': n_des,
        'errors': len(total_errors),
        'elapsed_min': round(elapsed / 60, 1),
    }
    with open(OIH_DIR / "oih_volatile_log.json", 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\n  输出文件: {VOLATILE_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
