#!/usr/bin/env python3
"""
download_5m_manager.py
自动抓取 BTC/USDT 5 分钟线数据
- 当前年份按月保存
- 早于当前年份按年保存
- 支持断点续传 (manifest.json)
- UTC 时间到分钟，不保存秒和毫秒
"""

import os
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import ccxt
import pandas as pd
from tqdm import tqdm
import sys  # 添加sys模块用于刷新输出

# ------------- CONFIG -------------
SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
START_YEAR = 2018
# 使用更稳健的方式获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "btc_data_5m")  # 项目根目录下的 data/btc_data_5m
LIMIT = 1000
RETRY_SLEEP = 5
MAX_RETRIES = 5
BACKFILL_DAYS = 3   # 回溯天数（0 表示不回溯）
MANIFEST_PATH = os.path.join(DATA_ROOT, "manifest.json")
ARCHIVE_MONTHLY = False  # True: 归档月文件; False: 删除月文件
# ----------------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def now_year():
    return datetime.now(timezone.utc).year

def month_start_end_ts_ms(year, month):
    start = datetime(year, month, 1, 0, 0, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, 0, 0, tzinfo=timezone.utc) - timedelta(milliseconds=1)
    else:
        end = datetime(year, month + 1, 1, 0, 0, tzinfo=timezone.utc) - timedelta(milliseconds=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)

def year_start_end_ts_ms(year):
    start = datetime(year, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(year + 1, 1, 1, 0, 0, tzinfo=timezone.utc) - timedelta(milliseconds=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)

def ms_to_iso_min(ms):
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M")

def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_manifest(man):
    tmp = MANIFEST_PATH + ".tmp"
    ensure_dir(os.path.dirname(MANIFEST_PATH))
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
    os.replace(tmp, MANIFEST_PATH)


def fetch_ohlcv_range(exchange, symbol, timeframe, since_ms, until_ms, limit=1000, desc=None):
    """分页拉取，返回 DataFrame indexed by UTC datetime"""
    all_rows = []
    fetch_since = since_ms
    bar_ms = 5 * 60 * 1000
    total_est = max(0, (until_ms - since_ms) // bar_ms)

    if desc is None:
        desc = f"{ms_to_iso_min(since_ms)}->{ms_to_iso_min(until_ms)}"
    else:
        msg = f"{desc} -> {ms_to_iso_min(since_ms)}..{ms_to_iso_min(until_ms)}"
        print(msg)
        sys.stdout.flush()
        time.sleep(0.05)

    with tqdm(total=total_est if total_est > 0 else None, desc=desc, unit="bars", leave=True, ncols=100) as pbar:
        while fetch_since <= until_ms:
            ok = False
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=limit)
                    ok = True
                    break
                except Exception as e:
                    warn_msg = f"[warn] fetch error: {e} (attempt {attempt}/{MAX_RETRIES}), sleep {RETRY_SLEEP}s"
                    print(warn_msg)
                    sys.stdout.flush()
                    time.sleep(RETRY_SLEEP)
            if not ok:
                raise RuntimeError("连续 fetch 失败")

            if not ohlcv:
                break

            added = 0
            for row in ohlcv:
                ts = int(row[0])
                if ts > until_ms:
                    break
                all_rows.append(row)
                added += 1

            pbar.update(added)
            if ohlcv:
                last_ts = int(ohlcv[-1][0])
                fetch_since = last_ts + 1
            else:
                fetch_since = last_ts + 1
            time.sleep(max(0.0, getattr(exchange, "rateLimit", 0) / 1000.0))
    time.sleep(0.2)
    sys.stdout.flush()

    if not all_rows:
        return pd.DataFrame(columns=['datetime', 'date_utc', 'datetime_cn', 'date_cn', 'open', 'high', 'low', 'close'])
    df = pd.DataFrame(all_rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime_obj'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['datetime_obj'] = df['datetime_obj'].dt.floor('min')

    df['date_utc'] = df['datetime_obj'].dt.strftime('%Y-%m-%d')
    df['datetime_cn_obj'] = df['datetime_obj'].dt.tz_convert('Asia/Shanghai')
    df['date_cn'] = df['datetime_cn_obj'].dt.strftime('%Y-%m-%d')

    df['datetime'] = df['datetime_obj'].dt.strftime('%Y-%m-%d %H:%M')
    df['datetime_cn'] = df['datetime_cn_obj'].dt.strftime('%Y-%m-%d %H:%M')

    # 不保留 volume 列
    df = df.drop(columns=['timestamp', 'volume', 'datetime_obj', 'datetime_cn_obj'])
    df = df[['datetime', 'date_utc', 'datetime_cn', 'date_cn', 'open', 'high', 'low', 'close']]
    df = df.drop_duplicates(subset=['datetime'], keep='first').sort_values(['datetime']).reset_index(drop=True)
    return df


def read_parquet_last_ts(fname):
    """尽量高效地读取 parquet 最后一行时间戳；回退到 pandas 全读"""
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(fname)
        rg = pf.num_row_groups - 1
        tbl = pf.read_row_group(rg, columns=['datetime'])
        df_rg = tbl.to_pandas()
        # 获取最后一行的日期和时间
        last_dt = pd.to_datetime(df_rg['datetime'].iloc[-1], format='%Y-%m-%d %H:%M', utc=True)
        return int(last_dt.timestamp() * 1000)
    except Exception:
        pass
    df = pd.read_parquet(fname)
    last_dt = pd.to_datetime(df['datetime'].iloc[-1], format='%Y-%m-%d %H:%M', utc=True)
    return int(last_dt.timestamp() * 1000)


def save_parquet_atomic(df, fname):
    tmp = fname + ".tmp"
    ensure_dir(os.path.dirname(fname) or ".")
    df.to_parquet(tmp, index=False)  # 不再将datetime作为索引保存
    os.replace(tmp, fname)


def merge_monthly_to_year(year):
    """合并 month files -> yearly parquet"""
    year_dir = os.path.join(DATA_ROOT, str(year))
    if not os.path.isdir(year_dir):
        print(f"[merge] {year} 无月目录, 跳过。")
        return
    files = sorted(Path(year_dir).glob("btc_usdt_5m_*.parquet"))
    if not files:
        print(f"[merge] {year} 目录无月文件, 跳过。")
        return
    print(f"[merge] 正在合并 {year} 的 {len(files)} 个月文件...")
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            cols = ['datetime', 'date_utc', 'datetime_cn', 'date_cn', 'open', 'high', 'low', 'close']
            existing_cols = [col for col in cols if col in df.columns]
            df = df[existing_cols]
            dfs.append(df)
        except Exception as e:
            print(f"[merge] 读取 {f} 失败: {e}")
    if not dfs:
        print(f"[merge] 没有可合并数据, 结束。")
        return
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values(['datetime']).drop_duplicates(subset=['datetime'], keep='first').reset_index(drop=True)
    out_fname = os.path.join(DATA_ROOT, f"btc_usdt_{TIMEFRAME}_{year}.parquet")
    save_parquet_atomic(df_all, out_fname)
    print(f"[merge] 已写入年度文件: {out_fname} ({len(df_all)} 行)")

    if ARCHIVE_MONTHLY:
        archive_dir = os.path.join(DATA_ROOT, f"{year}_monthly_archive")
        ensure_dir(archive_dir)
        for f in files:
            Path(f).rename(Path(archive_dir) / Path(f).name)
        print(f"[merge] 月文件已归档至 {archive_dir}")
    else:
        for f in files:
            try:
                os.remove(f)
            except Exception:
                pass
        print(f"[merge] 月文件已删除。")
    try:
        os.rmdir(year_dir)
    except Exception:
        pass

def process_month_file(exchange, year, month, manifest):
    """处理单个月份的抓取/续传逻辑"""
    current_year = datetime.now(timezone.utc).year
    current_month = datetime.now(timezone.utc).month
    if year == current_year and month == current_month:
        print(f"[skip] 跳过当前月份 {year}-{month:02d} 的下载")
        return

    year_dir = os.path.join(DATA_ROOT, str(year))
    ensure_dir(year_dir)
    fname = os.path.join(year_dir, f"btc_usdt_{TIMEFRAME}_{year}_{month:02d}.parquet")

    if os.path.exists(fname):
        print(f"{year} 年 {month:02d} 月文件已存在, 跳过。")
        return

    start_ms, end_ms = month_start_end_ts_ms(year, month)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if year == now_year():
        end_ms = min(end_ms, now_ms - 1000)

    backfill_ms = BACKFILL_DAYS * 24 * 3600 * 1000 if BACKFILL_DAYS > 0 else 0
    earliest_allowed_ms = year_start_end_ts_ms(START_YEAR)[0]
    fetch_since = max(earliest_allowed_ms, start_ms - backfill_ms)

    key = f"{year}-{month:02d}"
    if key in manifest:
        m_last = manifest[key].get("last_ts")
        if m_last:
            fetch_since = max(fetch_since, m_last + 1)

    if fetch_since > end_ms:
        print(f"[skip] {year}-{month:02d} 无需拉取 (fetch_since > end).")
        return

    df_new = fetch_ohlcv_range(exchange, SYMBOL, TIMEFRAME, fetch_since, end_ms, limit=LIMIT,
                               desc=f"{year}-{month:02d}")
    if df_new.empty:
        print(f"{year}-{month:02d} 无新数据。")
        return

    # 直接写入（如果存在旧文件则合并）
    if os.path.exists(fname):
        df_old = pd.read_parquet(fname)
        cols = ['datetime', 'date_utc', 'datetime_cn', 'date_cn', 'open', 'high', 'low', 'close']
        existing = [c for c in cols if c in df_old.columns]
        df_old = df_old[existing]
        df_combined = pd.concat([df_old, df_new], ignore_index=True).sort_values(['datetime'])
        df_combined = df_combined.drop_duplicates(subset=['datetime'], keep='first').reset_index(drop=True)
    else:
        df_combined = df_new

    save_parquet_atomic(df_combined, fname)

    last_dt = pd.to_datetime(df_combined['datetime'].iloc[-1], format='%Y-%m-%d %H:%M', utc=True)
    last_ts = int(last_dt.timestamp() * 1000)

    manifest[key] = {"filename": fname, "last_ts": last_ts, "rows": len(df_combined),
                     "updated_at": datetime.now(timezone.utc).isoformat()}
    save_manifest(manifest)
    ok_msg = f"[ok] 写入 {os.path.basename(fname)} ({len(df_combined)} 行)"
    print(ok_msg)
    sys.stdout.flush()
    time.sleep(0.2)

def main():
    ensure_dir(DATA_ROOT)
    manifest = load_manifest()
    exchange = ccxt.binance({'enableRateLimit': True})
    cy = now_year()
    for year in range(START_YEAR, cy + 1):
        if year < cy:
            yearly_file = os.path.join(DATA_ROOT, f"btc_usdt_{TIMEFRAME}_{year}.parquet")
            if os.path.exists(yearly_file):
                print(f"{year} 年度文件已存在，跳过。")
                continue
            month_dir = os.path.join(DATA_ROOT, str(year))
            if os.path.isdir(month_dir):
                merge_monthly_to_year(year)
                continue
            else:
                s_ms, e_ms = year_start_end_ts_ms(year)
                df = fetch_ohlcv_range(exchange, SYMBOL, TIMEFRAME, s_ms, e_ms, limit=LIMIT,
                                       desc=f"{year}")
                if df.empty:
                    print(f"[warn] 年 {year} 无数据, 跳过。")
                    continue
                out_fname = os.path.join(DATA_ROOT, f"btc_usdt_{TIMEFRAME}_{year}.parquet")
                save_parquet_atomic(df, out_fname)
                last_dt = pd.to_datetime(df['datetime'].iloc[-1], format='%Y-%m-%d %H:%M', utc=True)
                last_ts = int(last_dt.timestamp() * 1000)

                manifest[str(year)] = {"filename": out_fname, "last_ts": last_ts, "rows": len(df),
                                       "updated_at": datetime.now(timezone.utc).isoformat()}
                save_manifest(manifest)
                ok_msg = f"[ok] 写入年度文件 {os.path.basename(out_fname)} ({len(df)} 行)"
                print(ok_msg)
                sys.stdout.flush()
                time.sleep(0.2)
        else:
            for m in range(1, datetime.now(timezone.utc).month + 1):
                process_month_file(exchange, year, m, manifest)

    # rollover: 若上年度的月文件还没合并，合并它
    last_year = cy - 1
    prev_month_dir = os.path.join(DATA_ROOT, str(last_year))
    yearly_file = os.path.join(DATA_ROOT, f"btc_usdt_{TIMEFRAME}_{last_year}.parquet")
    if os.path.isdir(prev_month_dir) and not os.path.exists(yearly_file):
        print(f"[rollover] 检测到 {last_year} 的月文件且未合并成年文件, 开始合并...")
        merge_monthly_to_year(last_year)
        if os.path.exists(yearly_file):
            df = pd.read_parquet(yearly_file)
            # 获取最后一行的时间戳用于manifest
            last_dt = pd.to_datetime(df['datetime'].iloc[-1], format='%Y-%m-%d %H:%M', utc=True)
            last_ts = int(last_dt.timestamp() * 1000)

            manifest[str(last_year)] = {"filename": yearly_file, "last_ts": last_ts, "rows": len(df),
                                        "updated_at": datetime.now(timezone.utc).isoformat()}
            save_manifest(manifest)
            print(f"[rollover] 合并并更新 manifest 完成: {os.path.basename(yearly_file)}")

    print("[done] 本次运行结束。")

if __name__ == "__main__":
    main()
