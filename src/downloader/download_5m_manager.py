#!/usr/bin/env python3
"""
download_5m_manager.py
自动抓取 BTC/USDT 5 分钟线数据
- 当前年份按月保存
- 早于当前年份按年保存
- 支持断点续传 (manifest.json)
- UTC 时间到分钟，不保存秒和毫秒
- 简化版，只保留 datetime + OHLC
"""

import os
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import ccxt
import pandas as pd

# ------------- CONFIG -------------
SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
START_YEAR = 2018
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "btc_data_5m")
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

def fetch_ohlcv_range(exchange, symbol, timeframe, since_ms, until_ms, limit=1000):
    """分页拉取，返回 DataFrame indexed by UTC datetime"""
    all_rows = []
    fetch_since = since_ms
    bar_ms = 5 * 60 * 1000

    while fetch_since <= until_ms:
        ok = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=limit)
                ok = True
                break
            except Exception as e:
                print(f"[warn] fetch error: {e} (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(RETRY_SLEEP)
        if not ok:
            raise RuntimeError("连续 fetch 失败")

        if not ohlcv:
            break

        for row in ohlcv:
            ts = int(row[0])
            if ts > until_ms:
                break
            all_rows.append(row)

        if ohlcv:
            last_ts = int(ohlcv[-1][0])
            fetch_since = last_ts + 1
        else:
            fetch_since += bar_ms
        time.sleep(max(0.0, getattr(exchange, "rateLimit", 0) / 1000.0))

    if not all_rows:
        return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close'])

    df = pd.DataFrame(all_rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime_obj'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['datetime'] = df['datetime_obj'].dt.strftime('%Y-%m-%d %H:%M')
    df = df[['datetime', 'open', 'high', 'low', 'close']]
    df = df.drop_duplicates(subset=['datetime'], keep='first').sort_values(['datetime']).reset_index(drop=True)
    return df

def save_parquet_atomic(df, fname):
    tmp = fname + ".tmp"
    ensure_dir(os.path.dirname(fname) or ".")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, fname)

def merge_monthly_to_year(year):
    year_dir = os.path.join(DATA_ROOT, str(year))
    if not os.path.isdir(year_dir):
        print(f"[merge] {year} 无月目录, 跳过。")
        return
    files = sorted(Path(year_dir).glob("btc_usdt_5m_*.parquet"))
    if not files:
        print(f"[merge] {year} 目录无月文件, 跳过。")
        return
    print(f"[merge] 正在合并 {year} 年的 {len(files)} 个月文件...")
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"[merge] 读取 {f} 失败: {e}")
    if not dfs:
        return
    df_all = pd.concat(dfs, ignore_index=True).sort_values(['datetime']).drop_duplicates(subset=['datetime']).reset_index(drop=True)
    out_fname = os.path.join(DATA_ROOT, f"btc_usdt_{TIMEFRAME}_{year}.parquet")
    save_parquet_atomic(df_all, out_fname)
    print(f"[ok] 写入 {year} 年度文件: {out_fname} ({len(df_all)} 行)")
    if ARCHIVE_MONTHLY:
        archive_dir = os.path.join(DATA_ROOT, f"{year}_monthly_archive")
        ensure_dir(archive_dir)
        for f in files:
            Path(f).rename(Path(archive_dir) / Path(f).name)
    else:
        for f in files:
            try: os.remove(f)
            except: pass
    try: os.rmdir(year_dir)
    except: pass

def process_month_file(exchange, year, month, manifest):
    current_year = datetime.now(timezone.utc).year
    current_month = datetime.now(timezone.utc).month
    if year == current_year and month == current_month:
        print(f"跳过当前月份 {year}-{month:02d} 的下载")
        return

    year_dir = os.path.join(DATA_ROOT, str(year))
    ensure_dir(year_dir)
    fname = os.path.join(year_dir, f"btc_usdt_{TIMEFRAME}_{year}_{month:02d}.parquet")

    start_ms, end_ms = month_start_end_ts_ms(year, month)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if year == current_year:
        end_ms = min(end_ms, now_ms - 1000)

    backfill_ms = BACKFILL_DAYS * 24 * 3600 * 1000 if BACKFILL_DAYS > 0 else 0
    fetch_since = start_ms
    key = f"{year}-{month:02d}"

    if key in manifest and manifest[key].get("last_ts"):
        fetch_since = max(fetch_since, manifest[key]["last_ts"] + 1)
    else:
        fetch_since = max(start_ms, start_ms - backfill_ms)

    if os.path.exists(fname):
        df_old = pd.read_parquet(fname)
        if not df_old.empty:
            last_dt = pd.to_datetime(df_old['datetime'].iloc[-1], format='%Y-%m-%d %H:%M', utc=True)
            fetch_since = max(fetch_since, int(last_dt.timestamp() * 1000) + 1)

    if fetch_since > end_ms:
        print(f"[skip] {year}-{month:02d} 无需拉取 (fetch_since > end).")
        return

    df_new = fetch_ohlcv_range(exchange, SYMBOL, TIMEFRAME, fetch_since, end_ms, limit=LIMIT)
    if df_new.empty:
        print(f"{year}-{month:02d} 月度文件已存在，跳过。")
        return

    if os.path.exists(fname):
        df_old = pd.read_parquet(fname)
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
    print(f"[ok] 写入 {year} 年 {month:02d} 月文件: {os.path.basename(fname)} ({len(df_combined)} 行)")

def main():
    ensure_dir(DATA_ROOT)
    manifest = load_manifest()
    exchange = ccxt.binance({'enableRateLimit': True})
    cy = now_year()

    # --- 年度抓取 ---
    years_to_fetch = list(range(START_YEAR, cy))
    for year in years_to_fetch:
        yearly_file = os.path.join(DATA_ROOT, f"btc_usdt_{TIMEFRAME}_{year}.parquet")
        if os.path.exists(yearly_file):
            print(f"[skip] {year} 年度文件已存在")
            continue
        month_dir = os.path.join(DATA_ROOT, str(year))
        if os.path.isdir(month_dir):
            merge_monthly_to_year(year)
            continue
        s_ms, e_ms = year_start_end_ts_ms(year)
        df = fetch_ohlcv_range(exchange, SYMBOL, TIMEFRAME, s_ms, e_ms, limit=LIMIT)
        if df.empty:
            print(f"[warn] 年 {year} 无数据, 跳过。")
            continue
        save_parquet_atomic(df, yearly_file)
        last_dt = pd.to_datetime(df['datetime'].iloc[-1], format='%Y-%m-%d %H:%M', utc=True)
        last_ts = int(last_dt.timestamp() * 1000)
        manifest[str(year)] = {"filename": yearly_file, "last_ts": last_ts, "rows": len(df),
                               "updated_at": datetime.now(timezone.utc).isoformat()}
        save_manifest(manifest)
        print(f"[ok] 写入 {year} 年度文件: {os.path.basename(yearly_file)} ({len(df)} 行)")

    # --- 当前年份月度抓取 ---
    current_months = [m for m in range(1, datetime.now(timezone.utc).month + 1)]
    for m in current_months:
        process_month_file(exchange, cy, m, manifest)

    # --- 回滚上一年度未合并月文件 ---
    last_year = cy - 1
    prev_month_dir = os.path.join(DATA_ROOT, str(last_year))
    yearly_file = os.path.join(DATA_ROOT, f"btc_usdt_{TIMEFRAME}_{last_year}.parquet")
    if os.path.isdir(prev_month_dir) and not os.path.exists(yearly_file):
        print(f"[rollover] 检测到 {last_year} 的月文件且未合并成年文件, 开始合并...")
        merge_monthly_to_year(last_year)
        if os.path.exists(yearly_file):
            df = pd.read_parquet(yearly_file)
            last_dt = pd.to_datetime(df['datetime'].iloc[-1], format='%Y-%m-%d %H:%M', utc=True)
            last_ts = int(last_dt.timestamp() * 1000)
            manifest[str(last_year)] = {"filename": yearly_file, "last_ts": last_ts, "rows": len(df),
                                        "updated_at": datetime.now(timezone.utc).isoformat()}
            save_manifest(manifest)
            print(f"[rollover] 合并并更新 manifest 完成: {last_year} 年度文件 {os.path.basename(yearly_file)}")

    print("本次运行结束。")

if __name__ == "__main__":
    main()
    # TODO: 进度条未处理