#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多时区回测系统
信号日开盘买入，当日收盘或触发止盈止损时平仓
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
from colorama import init, Fore, Style

init(autoreset=True)
warnings.filterwarnings("ignore")

# ==================== 时区映射 ====================
TIMEZONE_MAP = {
    "UTC0": "UTC",
    "UTC-1": "Etc/GMT+1", "UTC-2": "Etc/GMT+2", "UTC-3": "Etc/GMT+3",
    "UTC-4": "Etc/GMT+4", "UTC-5": "Etc/GMT+5", "UTC-6": "Etc/GMT+6",
    "UTC-7": "Etc/GMT+7", "UTC-8": "Etc/GMT+8", "UTC-9": "Etc/GMT+9",
    "UTC-10": "Etc/GMT+10", "UTC-11": "Etc/GMT+11", "UTC-12": "Etc/GMT+12",
    "UTC1": "Etc/GMT-1", "UTC2": "Etc/GMT-2", "UTC3": "Etc/GMT-3",
    "UTC4": "Etc/GMT-4", "UTC5": "Etc/GMT-5", "UTC6": "Etc/GMT-6",
    "UTC7": "Etc/GMT-7", "UTC8": "Etc/GMT-8", "UTC9": "Etc/GMT-9",
    "UTC10": "Etc/GMT-10", "UTC11": "Etc/GMT-11", "UTC12": "Etc/GMT-12",
}

# ==================== 路径配置 ====================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "merged" / "btc" / "btc_5m"
OUT_DIR = ROOT_DIR / "results" / "backtest_多时区"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 回测参数 ====================
TIMEZONE = None                     # None=全时区测试，"UTC7"=单时区
TRADE_DIRECTION = "多"              # "多"或"空"
STAR_COL = "星宿"                   # 信号列名
TARGET_STAR = "角宿"                # 目标信号

INITIAL_CAPITAL = 1000.0
PEAK_PERCENT = 1                    # 仓位比例（支持杠杆>1）
TAKE_PROFIT_PERCENT = 1             # 止盈比例
STOP_LOSS_PERCENT = 1               # 止损比例

TAKER_FEE = 0.0005                  # 交易费率
MAKER_FEE = 0.0003                  # 手续费率
FUNDING_RATE = 0.0003               # 资金费率
SLIPPAGE = 0.0005                   # 滑点

# ==================== 绘图配置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def run_backtest(df_raw, timezone_name, timezone_str):
    """单时区回测：时区转换 → 信号识别 → 交易模拟 → 指标计算"""
    df = df_raw.copy()

    # 时区转换
    df["datetime_tz"] = df["datetime"].dt.tz_convert(timezone_str)
    df = df.sort_values("datetime_tz").reset_index(drop=True)
    df["date_tz"] = df["datetime_tz"].dt.floor("D")

    # 识别信号起点
    star_mask = (df[STAR_COL] == TARGET_STAR)
    segment_starts = df.index[star_mask & (~star_mask.shift(1, fill_value=False))].tolist()
    print(Fore.CYAN + f"  [{timezone_name}] 找到 {len(segment_starts)} 个信号" + Style.RESET_ALL)

    # 初始化
    processed_dates = set()
    capital = float(INITIAL_CAPITAL)
    daily_closes = []
    trades = []
    capital_curve = []
    liquidated = False

    # 遍历信号
    for start_idx in segment_starts:
        if liquidated or start_idx not in df.index:
            break

        start_row = df.loc[start_idx]
        start_date = start_row['date_tz']

        if start_date in processed_dates:
            continue

        day_rows = df[df['date_tz'] == start_date]
        if day_rows.empty:
            continue

        # 开仓
        open_row = day_rows.iloc[0]
        open_price = float(open_row['open'])
        open_time = open_row['datetime_tz']

        if TRADE_DIRECTION == "多":
            stop_loss_price = open_price * (1 - STOP_LOSS_PERCENT)
            take_profit_price = open_price * (1 + TAKE_PROFIT_PERCENT)
        else:
            stop_loss_price = open_price * (1 + STOP_LOSS_PERCENT)
            take_profit_price = open_price * (1 - TAKE_PROFIT_PERCENT)

        # 扫描止盈止损
        exit_type = "正常平仓"
        exit_row = day_rows.iloc[-1]

        for idx, row in day_rows.iterrows():
            if row["datetime"] <= open_time:
                continue

            high_price = float(row["high"])
            low_price = float(row["low"])

            if TRADE_DIRECTION == "多":
                if high_price >= take_profit_price:
                    exit_type = "止盈平仓"
                    exit_row = row
                    break
                if low_price <= stop_loss_price:
                    exit_type = "止损平仓"
                    exit_row = row
                    break
            else:
                if low_price <= take_profit_price:
                    exit_type = "止盈平仓"
                    exit_row = row
                    break
                if high_price >= stop_loss_price:
                    exit_type = "止损平仓"
                    exit_row = row
                    break

        # 平仓（含滑点）
        close_price = float(exit_row["close"])
        close_time = exit_row["datetime_tz"]

        if TRADE_DIRECTION == "多":
            open_price_slippage = open_price * (1 + SLIPPAGE)
            close_price_slippage = close_price * (1 - SLIPPAGE)
        else:
            open_price_slippage = open_price * (1 - SLIPPAGE)
            close_price_slippage = close_price * (1 + SLIPPAGE)

        # 计算仓位（基于峰值）
        peak_close = max(daily_closes) if daily_closes else INITIAL_CAPITAL
        trade_size = peak_close * PEAK_PERCENT
        units = trade_size / open_price_slippage

        # 计算成本
        open_fee = trade_size * TAKER_FEE
        funding_fee = trade_size * FUNDING_RATE
        close_value = units * close_price_slippage
        close_fee = close_value * TAKER_FEE
        total_fees = open_fee + funding_fee + close_fee

        # 计算盈亏
        if TRADE_DIRECTION == "多":
            pnl = units * (close_price_slippage - open_price_slippage) - total_fees
        else:
            pnl = units * (open_price_slippage - close_price_slippage) - total_fees

        prev_capital = capital
        capital += pnl

        if capital <= 0:
            capital = 0.0
            liquidated = True

        daily_closes.append(capital)
        processed_dates.add(start_date)

        # 记录交易
        price_change = (close_price - open_price) / open_price if TRADE_DIRECTION == "多" \
                       else (open_price - close_price) / open_price

        trades.append({
            "开仓时间": open_time, "平仓时间": close_time,
            "开仓价": open_price, "平仓价": close_price, "方向": TRADE_DIRECTION,
            "开仓前资金_USD": round(prev_capital, 8), "投入(允许杠杆)": round(trade_size, 8),
            "实际涨跌幅": f"{price_change*100:.2f}%", "收益_USD": round(pnl, 8),
            "平仓后资金_USD": round(capital, 8), "平仓类型": exit_type
        })
        capital_curve.append(capital)

    # 补齐完整日期
    all_dates = sorted(df["date_tz"].unique())
    full_capital_curve = []
    last_capital = INITIAL_CAPITAL
    j = 0
    processed_dates_list = sorted(list(processed_dates))

    for date in all_dates:
        while j < len(processed_dates_list) and processed_dates_list[j] < date:
            j += 1

        if j < len(processed_dates_list) and processed_dates_list[j] == date:
            full_capital_curve.append(capital_curve[j])
            last_capital = capital_curve[j]
            j += 1
        else:
            full_capital_curve.append(last_capital)

    if liquidated and trades:
        liquidated_date = trades[-1]["开仓时间"]
        for i, date in enumerate(all_dates):
            if date > liquidated_date:
                full_capital_curve[i] = 0.0

    # 计算指标
    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        final_capital = full_capital_curve[-1]
        total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL

        days = (all_dates[-1] - all_dates[0]).days + 1
        years = days / 365.0
        annualized_return = -1.0 if final_capital <= 0 else \
                           (final_capital / INITIAL_CAPITAL) ** (1.0 / years) - 1.0

        returns = [t["收益_USD"] / INITIAL_CAPITAL for t in trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) \
                      if len(returns) > 1 and np.std(returns) != 0 else 0

        pnl_arr = [t["收益_USD"] for t in trades]
        wins = sum(1 for pnl in pnl_arr if pnl > 0)
        win_rate = wins / len(pnl_arr)

        winning_trades = [pnl for pnl in pnl_arr if pnl > 0]
        losing_trades = [pnl for pnl in pnl_arr if pnl < 0]
        profit_loss_ratio = np.mean(winning_trades) / abs(np.mean(losing_trades)) \
                           if winning_trades and losing_trades else 0

        equity = np.array(full_capital_curve, dtype=float)
        cummax = np.maximum.accumulate(equity)
        drawdown = (cummax - equity) / np.where(cummax == 0, 1, cummax)
        max_dd = float(np.nanmax(drawdown))

        max_drawdown_duration = 0
        i = len(equity) - 1
        while i >= 0:
            if equity[i] == cummax[i]:
                peak_date = all_dates[i]
                recovery_date = next((all_dates[j] for j in range(i + 1, len(equity))
                                     if equity[j] >= equity[i]), None)
                duration = (recovery_date - peak_date).days if recovery_date else \
                          (all_dates[-1] - peak_date).days
                max_drawdown_duration = max(max_drawdown_duration, duration)
            i -= 1

        tz_number = int(timezone_name.replace("UTC", ""))
        summary = {
            "时区": tz_number, "初始资金": INITIAL_CAPITAL, "结束资金": round(final_capital, 2),
            "累计收益率": round(total_return * 100, 2), "年化收益率": round(annualized_return * 100, 2),
            "夏普比率": round(sharpe_ratio, 2), "覆盖天数": int(days), "覆盖年数": round(years, 2),
            "交易次数": len(trades), "盈利次数": int(wins), "亏损次数": int(len(pnl_arr) - wins),
            "胜率": round(win_rate * 100, 2), "盈亏比": round(profit_loss_ratio, 2),
            "最大回撤": round(max_dd * 100, 2), "最大回撤时长": max_drawdown_duration,
            "是否爆仓": bool(liquidated)
        }
    else:
        tz_number = int(timezone_name.replace("UTC", ""))
        summary = {
            "时区": tz_number, "初始资金": INITIAL_CAPITAL, "结束资金": INITIAL_CAPITAL,
            "累计收益率": 0, "年化收益率": 0, "夏普比率": 0, "覆盖天数": 0, "覆盖年数": 0,
            "交易次数": 0, "盈利次数": 0, "亏损次数": 0, "胜率": 0, "盈亏比": 0,
            "最大回撤": 0, "最大回撤时长": 0, "是否爆仓": False
        }

    return summary, trades_df, all_dates, full_capital_curve


# ==================== 主程序 ====================
if __name__ == "__main__":
    print(Fore.YELLOW + "="*50 + "\n         多时区回测系统启动\n" + "="*50 + Style.RESET_ALL)

    # 加载数据
    parquet_files = sorted(DATA_DIR.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"在 {DATA_DIR} 未找到数据文件")

    dfs = [pd.read_parquet(p) for p in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # 确定时区
    timezones_to_test = list(TIMEZONE_MAP.keys()) if TIMEZONE is None else [TIMEZONE]
    print(Fore.GREEN + f"将测试 {len(timezones_to_test)} 个时区" + Style.RESET_ALL)

    all_summaries = []

    # 执行回测
    for tz_name in timezones_to_test:
        tz_str = TIMEZONE_MAP[tz_name]
        print(f"\n{Fore.BLUE}{'='*50}\n正在测试: {tz_name}\n{'='*50}{Style.RESET_ALL}")

        try:
            summary, trades_df, all_dates, full_capital_curve = run_backtest(df, tz_name, tz_str)
            all_summaries.append(summary)

            if not trades_df.empty:
                # 保存CSV
                csv_path = OUT_DIR / f"回测_交易记录_{tz_name}_{TRADE_DIRECTION}.csv"
                trades_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                print(Fore.GREEN + f"  已保存: {csv_path.name}" + Style.RESET_ALL)

                # 绘图
                plt.figure(figsize=(15, 8))
                plt.plot(all_dates, full_capital_curve, linewidth=2, color='#1f77b4')
                plt.title(f'资金曲线 - {tz_name} - {TRADE_DIRECTION}单', fontsize=16)
                plt.xlabel('日期', fontsize=12)
                plt.ylabel('资金 (USDT)', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
                plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
                plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', alpha=0.7,
                           label=f'初始: {INITIAL_CAPITAL}')
                plt.annotate(f'最终: {full_capital_curve[-1]:.2f}',
                            xy=(all_dates[-1], full_capital_curve[-1]),
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
                plt.legend()
                plt.tight_layout()

                img_path = OUT_DIR / f"回测资金曲线_{tz_name}_{TRADE_DIRECTION}.png"
                plt.savefig(img_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(Fore.GREEN + f"  已保存: {img_path.name}" + Style.RESET_ALL)

            print(f"  收益: {summary['累计收益率']:.2f}% | 年化: {summary['年化收益率']:.2f}% | "
                  f"交易: {summary['交易次数']} | 胜率: {summary['胜率']:.2f}%")

        except Exception as e:
            print(Fore.RED + f"  {tz_name} 回测失败: {e}" + Style.RESET_ALL)

    # 生成汇总
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries).sort_values('年化收益率', ascending=False)
        summary_path = OUT_DIR / f"回测_汇总表_{TRADE_DIRECTION}.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

        print(f"\n{Fore.YELLOW}{'='*50}\n          回测完成\n{'='*50}{Style.RESET_ALL}")
        print(Fore.GREEN + f"汇总表: {summary_path}" + Style.RESET_ALL)
        print(f"\n{Fore.CYAN}TOP 5:{Style.RESET_ALL}")
        print(summary_df[['时区', '年化收益率', '累计收益率', '交易次数', '胜率']].head().to_string(index=False))
    else:
        print(Fore.RED + "无成功回测" + Style.RESET_ALL)