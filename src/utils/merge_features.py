#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_features.py

功能：
- 将 data/btc_data_5m 下的 BTC 5m 文件与
  data/astro_data 下的玄学特征（九星、十二建星、干支历、星宿、节气）合并
- 输出到 data/merged/btc/btc_5m
- 月份文件优先，如果最新年份有月份文件，则按月合并
- 玄学 master 使用时间索引合并，避免重复列名
- 数据清洗：时间统一、删除重复、数值列转换、节气向后填充
- 输出前将 datetime 列格式化为字符串（默认 "%Y-%m-%d %H:%M"）
"""

import os
import sys
import traceback
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# -----------------------
# 配置路径
# -----------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BTC_DIR = os.path.join(ROOT_DIR, 'data', 'btc_data_5m')
ASTRO_DIR = os.path.join(ROOT_DIR, 'data', 'astro_data')
MERGED_DIR = os.path.join(ROOT_DIR, 'data', 'merged', 'btc', 'btc_5m')

ASTRO_FILES = {
    '九星': '九星.parquet',
    '十二建星': '十二建星.parquet',
    '干支历': '干支历.parquet',
    '星宿': '星宿.parquet',
    '节气': '节气.parquet'
}

os.makedirs(MERGED_DIR, exist_ok=True)

# -----------------------
# 辅助函数
# -----------------------
def create_sample_astro_data():
    """创建示例的天文数据 Parquet 文件"""
    os.makedirs(ASTRO_DIR, exist_ok=True)
    start_date = datetime(2017, 1, 1)
    end_date = datetime(2025, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # 九星
    jiuxing_df = pd.DataFrame({
        '日期': date_range.strftime('%Y-%m-%d %H:%M'),
        '月家九星': ['一白'] * len(date_range),
        '日家九星': ['二黑'] * len(date_range)
    })
    jiuxing_df.to_parquet(os.path.join(ASTRO_DIR, '九星.parquet'), index=False)

    # 十二建星
    jianxing_df = pd.DataFrame({
        '日期': date_range.strftime('%Y-%m-%d %H:%M'),
        '建星': ['建'] * len(date_range)
    })
    jianxing_df.to_parquet(os.path.join(ASTRO_DIR, '十二建星.parquet'), index=False)

    # 干支历
    ganzhi_df = pd.DataFrame({
        '日期': date_range.strftime('%Y-%m-%d %H:%M'),
        '年柱': ['甲子'] * len(date_range),
        '月柱': ['乙丑'] * len(date_range),
        '日柱': ['丙寅'] * len(date_range)
    })
    ganzhi_df.to_parquet(os.path.join(ASTRO_DIR, '干支历.parquet'), index=False)

    # 星宿
    xiuxiu_df = pd.DataFrame({
        '日期': date_range.strftime('%Y-%m-%d %H:%M'),
        '星宿': ['角宿'] * len(date_range)
    })
    xiuxiu_df.to_parquet(os.path.join(ASTRO_DIR, '星宿.parquet'), index=False)

    # 节气
    jieqi_dates = pd.date_range(start=start_date, end=end_date, freq='15D')
    jieqi_names = ['立春', '雨水', '惊蛰', '春分', '清明', '谷雨', '立夏', '小满', '芒种',
                   '夏至', '小暑', '大暑', '立秋', '处暑', '白露', '秋分', '寒露', '霜降',
                   '立冬', '小雪', '大雪', '冬至', '小寒', '大寒']
    jieqi_df = pd.DataFrame({
        '日期': jieqi_dates.strftime('%Y-%m-%d %H:%M'),
        '节气': [jieqi_names[i % len(jieqi_names)] for i in range(len(jieqi_dates))]
    })
    jieqi_df.to_parquet(os.path.join(ASTRO_DIR, '节气.parquet'), index=False)

def find_date_column(df: pd.DataFrame):
    """猜测日期列"""
    for c in df.columns:
        low = c.lower()
        if 'date' in low or '时间' in c or '日期' in c:
            return c
    return None

def safe_to_datetime(s, utc=True):
    """安全解析日期，兼容 YYYY-MM-DD 和 YYYY-MM-DD HH:MM"""
    try:
        dt = pd.to_datetime(s, errors='coerce', utc=utc)
        return dt
    except Exception:
        return pd.Series([pd.NaT]*len(s))

def clean_btc_df(df: pd.DataFrame):
    df = df.copy()
    if 'datetime' not in df.columns:
        raise ValueError("BTC 文件缺少 datetime 列")
    df['datetime'] = safe_to_datetime(df['datetime'], utc=True)
    df = df.dropna(subset=['datetime']).drop_duplicates(subset=['datetime'])
    for col in ['open','high','low','close','volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.sort_values('datetime')

def clean_merged_df(df: pd.DataFrame):
    df = df.copy()
    df['datetime'] = safe_to_datetime(df['datetime'], utc=True)
    df = df.dropna(subset=['datetime'])
    for col in ['open','high','low','close','volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if '节气' in df.columns:
        df['节气'] = df['节气'].ffill()
    df = df.drop_duplicates(subset=['datetime']).sort_values('datetime')
    return df

def datetime_to_str(df: pd.DataFrame, fmt="%Y-%m-%d %H:%M"):
    df = df.copy()
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
        df['datetime'] = df['datetime'].dt.strftime(fmt)
    return df

# -----------------------
# 构建玄学 master
# -----------------------
def load_astro_master():
    print("开始加载天体数据...")
    dfs = []
    try:
        files = os.listdir(ASTRO_DIR)
    except Exception as e:
        print("无法访问 ASTRO_DIR:", e)
        return None

    for key, fname in ASTRO_FILES.items():
        path = os.path.join(ASTRO_DIR, fname)
        if not os.path.exists(path):
            print(f"[警告] 文件不存在: {path}")
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"[警告] 读取失败 {fname}: {e}")
            continue
        date_col = find_date_column(df)
        if not date_col:
            print(f"[错误] 文件 {fname} 未找到日期列")
            continue
        dt = safe_to_datetime(df[date_col], utc=True)
        df_vals = df.drop(columns=[date_col])
        df_vals['__datetime__'] = dt
        df_vals = df_vals.dropna(subset=['__datetime__'])
        if df_vals.empty:
            print(f"[警告] 文件 {fname} 全部 NaT，跳过")
            continue
        df_vals = df_vals.set_index('__datetime__')
        dfs.append((key, df_vals))

    if not dfs:
        print("[错误] 未找到有效天体文件")
        return None

    master = dfs[0][1].copy()
    for key, dfv in dfs[1:]:
        conflict_cols = [c for c in dfv.columns if c in master.columns]
        for c in conflict_cols:
            if not master[c].isna().all():
                dfv = dfv.drop(columns=[c])
        master = master.join(dfv, how='outer')
    master = master.sort_index().reset_index().rename(columns={'__datetime__':'datetime'})
    master['datetime'] = pd.to_datetime(master['datetime'], utc=True)
    if '节气' in master.columns:
        master['节气'] = master['节气'].ffill()
    print("天体 master 构建完成，形状:", master.shape)
    return master

def merge_to_btc_df(btc_df: pd.DataFrame, astro_master: pd.DataFrame):
    btc_df = clean_btc_df(btc_df)
    merged = pd.merge_asof(btc_df, astro_master.sort_values('datetime'), on='datetime', direction='backward')
    merged = clean_merged_df(merged)
    return merged

# -----------------------
# 主流程
# -----------------------
def merge_all(datetime_fmt="%Y-%m-%d %H:%M"):
    create_sample_astro_data()
    astro_master = load_astro_master()
    if astro_master is None:
        print("[错误] 无有效天体 master，退出")
        return
    if not os.path.exists(BTC_DIR):
        print(f"[错误] BTC 数据目录不存在: {BTC_DIR}")
        return

    for item in sorted(os.listdir(BTC_DIR)):
        item_path = os.path.join(BTC_DIR, item)
        if os.path.isdir(item_path):
            year_dir = os.path.join(MERGED_DIR, item)
            os.makedirs(year_dir, exist_ok=True)
            month_files = sorted([f for f in os.listdir(item_path) if f.endswith('.parquet')])
            for mf in month_files:
                btc_month = pd.read_parquet(os.path.join(item_path, mf))
                merged = merge_to_btc_df(btc_month, astro_master)
                merged_out = datetime_to_str(merged, fmt=datetime_fmt)
                merged_out.to_parquet(os.path.join(year_dir, mf), index=False)
        elif item.endswith('.parquet'):
            btc_df = pd.read_parquet(item_path)
            merged = merge_to_btc_df(btc_df, astro_master)
            merged_out = datetime_to_str(merged, fmt=datetime_fmt)
            merged_out.to_parquet(os.path.join(MERGED_DIR, item), index=False)

if __name__ == "__main__":
    merge_all()
