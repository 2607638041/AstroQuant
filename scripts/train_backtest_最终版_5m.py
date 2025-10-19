#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测脚本（优化版）：
- 使用向量化操作提高性能
- 改进时区处理和数据结构
- 完整的资金曲线和统计指标计算
- 非持仓状态最高资金作为杠杆基准
- 配置区预分类信号列，后续代码自动使用
"""
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import init, Fore, Style
import time
import matplotlib
matplotlib.use('Agg')

init(autoreset=True)
warnings.filterwarnings("ignore")

# ==================== 用户配置区（仅修改此部分）====================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "merged" / "btc" / "btc_5m"             # 数据文件路径
OUT_DIR = ROOT_DIR / "results" / "backtest_最终版"                      # 结果输出路径

INITIAL_CAPITAL = 1000.0               # 初始资金（USDT）
START_DATE = "2018-04-01"              # 回测开始日期

# 信号分类配置（在此添加新列和信号）
SIGNAL_CLASSIFICATION = {
    "星宿": {
        "轸宿": {
            "PEAK_PERCENT": 1.5,
            "TAKE_PROFIT_PERCENT": 0.107,
            "STOP_LOSS_PERCENT": 0.053,
            "TIMEZONE": "UTC0"
        },                                  #完成
        "柳宿": {
            "PEAK_PERCENT": 1.35,
            "TAKE_PROFIT_PERCENT": 0.092,
            "STOP_LOSS_PERCENT": 0.04,
            "TIMEZONE": "UTC-8"
        },                                  #完成
        "毕宿": {
            "PEAK_PERCENT": 0.85,
            "TAKE_PROFIT_PERCENT": 0.156,
            "STOP_LOSS_PERCENT": 0.069,
            "TIMEZONE": "UTC+8"
        },                                  #完成
        "角宿": {
            "PEAK_PERCENT": 1.25,
            "TAKE_PROFIT_PERCENT": 0.1,
            "STOP_LOSS_PERCENT": 0.062,
            "TIMEZONE": "UTC-1"
        },                                  #待完成
        "氐宿": {
            "PEAK_PERCENT": 1.8,
            "TAKE_PROFIT_PERCENT": 0.1,
            "STOP_LOSS_PERCENT": 0.1,
            "TIMEZONE": "UTC+11"
        },
        "参宿": {
            "PEAK_PERCENT": 1,
            "TAKE_PROFIT_PERCENT": 0.1,
            "STOP_LOSS_PERCENT": 0.1,
            "TIMEZONE": "UTC-5"
        },
        "觜宿": {
            "PEAK_PERCENT": 0.95,
            "TAKE_PROFIT_PERCENT": 0.1,
            "STOP_LOSS_PERCENT": 0.1,
            "TIMEZONE": "UTC-10"
        },

        "箕宿": {
            "PEAK_PERCENT": 0.9,
            "TAKE_PROFIT_PERCENT": 0.1,
            "STOP_LOSS_PERCENT": 0.1,
            "TIMEZONE": "UTC-1"
        },
        "尾宿": {
            "PEAK_PERCENT": 0.75,
            "TAKE_PROFIT_PERCENT": 0.1,
            "STOP_LOSS_PERCENT": 0.1,
            "TIMEZONE": "UTC0"
        },
    },
    "建星": {
        "危": {
            "PEAK_PERCENT": 1.2,
            "TAKE_PROFIT_PERCENT": 0.1,
            "STOP_LOSS_PERCENT": 0.1,
            "TIMEZONE": "UTC-12"
        },
        "除": {
            "PEAK_PERCENT": 1.05,
            "TAKE_PROFIT_PERCENT": 0.1,
            "STOP_LOSS_PERCENT": 0.1,
            "TIMEZONE": "UTC+8"
        },
    }
}

# ==================== 内部常量（勿修改）====================

OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEZONE_MAP = {
    "UTC+8": "Etc/GMT-8",
    "UTC0": "UTC",
    **{f"UTC-{i}": f"Etc/GMT+{i}" for i in range(1, 13)},
    **{f"UTC+{i}": f"Etc/GMT-{i}" for i in range(11, 12)}
}

# 交易费用
TAKER_FEE = 0.0005
MAKER_FEE = 0.0003
FUNDING_RATE = 0.0002
SLIPPAGE = 0.0005
ENABLE_CHARTS = True                 # 是否绘制资金曲线图

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 从分类配置构建目标信号字典
def build_target_signals(signal_classification):
    """自动扁平化配置中的所有信号"""
    result = {}
    for col_name, signals_dict in signal_classification.items():
        for signal_name, params in signals_dict.items():
            result[signal_name] = {
                **params,
                "_column": col_name
            }
    return result

TARGET_SIGNALS = build_target_signals(SIGNAL_CLASSIFICATION)


# ==================== 工具函数 ====================

# 格式化时间显示
def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{int(seconds)}秒"
    elif seconds < 3600:
        return f"{int(seconds // 60)}分{int(seconds % 60)}秒"
    return f"{int(seconds // 3600)}小时{int((seconds % 3600) // 60)}分"

# 向量化检查止盈止损触发
def check_exit_trigger_vectorized(high, low, tp_price, sl_price):
    """向量化检查止盈止损触发 (1=止盈, 2=止损, 0=未触发)"""
    return np.where(high >= tp_price, 1, np.where(low <= sl_price, 2, 0))

# 预处理单个信号的时区数据
def preprocess_signal_data(df_raw, signal_name, signal_params):
    """预处理单个信号的时区数据"""
    df = df_raw.copy()
    tz_str = TIMEZONE_MAP[signal_params["TIMEZONE"]]

    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize('UTC')

    df["datetime_tz"] = df["datetime"].dt.tz_convert(tz_str)
    df = df.sort_values("datetime_tz").reset_index(drop=True)
    df["date_tz"] = df["datetime_tz"].dt.floor("D")

    return df

# 构建完整的资金曲线
def build_full_capital_curve(all_dates, trade_dict, initial_capital, liquidated, liquidation_date):
    """构建完整的资金曲线"""
    full_curve = [initial_capital]
    last_capital = initial_capital

    for i in range(1, len(all_dates)):
        if all_dates[i] in trade_dict:
            last_capital = trade_dict[all_dates[i]]

        if liquidated and all_dates[i] > liquidation_date:
            full_curve.append(0.0)
        else:
            full_curve.append(last_capital)

    return full_curve

# 计算统计指标
def calculate_summary_stats(trades, full_curve, all_dates, initial_capital):
    """计算统计指标"""
    if not trades:
        return {
            "初始资金": initial_capital,
            "最终资金": initial_capital,
            "累计收益率": "0.00%",
            "年化收益率": "0.00%",
            "夏普比率": 0,
            "胜率": "0.00%",
            "盈亏比": 0,
            "最大回撤": "0.00%",
            "最大回撤时长": 0,
            "总交易次数": 0
        }

    final_capital = full_curve[-1]
    total_return = (final_capital - initial_capital) / initial_capital

    date_diff = all_dates[-1] - all_dates[0]
    if isinstance(date_diff, np.timedelta64):
        days = int(date_diff / np.timedelta64(1, 'D')) + 1
    else:
        days = date_diff.days + 1

    years = days / 365.0
    annual_return = -1.0 if final_capital <= 0 else (final_capital / initial_capital) ** (1/years) - 1.0

    pnl_arr = np.array([t["收益（U）"] for t in trades])
    returns = pnl_arr / initial_capital
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if len(returns) > 1 and np.std(returns) > 0 else 0

    wins = np.sum(pnl_arr > 0)
    win_rate = wins / len(pnl_arr)

    win_trades = pnl_arr[pnl_arr > 0]
    loss_trades = pnl_arr[pnl_arr < 0]
    pl_ratio = np.mean(win_trades) / abs(np.mean(loss_trades)) if len(win_trades) > 0 and len(loss_trades) > 0 else 0

    equity = np.array(full_curve)
    cummax = np.maximum.accumulate(equity)
    drawdown = (cummax - equity) / np.where(cummax == 0, 1, cummax)
    max_dd = float(np.max(drawdown))

    peak_indices = np.where(equity == cummax)[0]
    max_dd_days = 0
    for i in peak_indices:
        if i < len(equity) - 1:
            recovery_idx = np.where(equity[i+1:] >= equity[i])[0]
            if len(recovery_idx) > 0:
                recovery_idx = recovery_idx[0] + i + 1
                dd_duration_delta = all_dates[recovery_idx] - all_dates[i]
                if isinstance(dd_duration_delta, np.timedelta64):
                    dd_duration = int(dd_duration_delta / np.timedelta64(1, 'D'))
                else:
                    dd_duration = dd_duration_delta.days
                max_dd_days = max(max_dd_days, dd_duration)

    return {
        "初始资金": initial_capital,
        "最终资金": round(final_capital, 2),
        "累计收益率": f"{total_return * 100:.2f}%",
        "年化收益率": f"{annual_return * 100:.2f}%",
        "夏普比率": round(sharpe, 2),
        "覆盖年数": round(years, 2),
        "胜率": f"{win_rate * 100:.2f}%",
        "盈亏比": round(pl_ratio, 2),
        "最大回撤": f"{max_dd * 100:.2f}%",
        "最大回撤时长": max_dd_days,
        "总交易次数": len(trades)
    }

# 执行完整回测逻辑
def run_backtest(df_raw):
    """执行完整回测逻辑"""
    all_trades = []
    max_non_holding_capital = INITIAL_CAPITAL
    current_capital = INITIAL_CAPITAL

    all_signals_list = []

    for signal_name, signal_params in TARGET_SIGNALS.items():
        target_col = signal_params["_column"]

        df_signal = preprocess_signal_data(df_raw, signal_name, signal_params)

        # 找信号起点（向量化）
        signal_mask = (df_signal[target_col] == signal_name).values
        segment_starts = np.where(signal_mask & ~np.roll(signal_mask, 1, axis=0))[0]

        for start_idx in segment_starts:
            start_date = df_signal.iloc[start_idx]["date_tz"]
            all_signals_list.append({
                "signal_name": signal_name,
                "start_idx": start_idx,
                "start_date": start_date,
                "params": signal_params,
                "df": df_signal,
                "target_col": target_col
            })

    # 按日期排序
    all_signals_list.sort(key=lambda x: x["start_date"])

    liquidated = False
    liquidation_date = None
    processed_dates_by_signal = {sig: set() for sig in TARGET_SIGNALS}

    for signal in all_signals_list:
        if liquidated:
            break

        signal_name = signal["signal_name"]
        start_idx = signal["start_idx"]
        start_date = signal["start_date"]
        params = signal["params"]
        df_signal = signal["df"]
        target_col = signal["target_col"]

        if start_date in processed_dates_by_signal[signal_name]:
            continue

        day_data = df_signal[df_signal["date_tz"] == start_date]
        if day_data.empty:
            continue

        dates = day_data["date_tz"].values
        datetimes = day_data["datetime_tz"].values
        opens = day_data["open"].values.astype(float)
        closes = day_data["close"].values.astype(float)
        highs = day_data["high"].values.astype(float)
        lows = day_data["low"].values.astype(float)

        open_price = opens[0]
        open_time = datetimes[0]

        tp_price = open_price * (1 + params["TAKE_PROFIT_PERCENT"])
        sl_price = open_price * (1 - params["STOP_LOSS_PERCENT"])

        exit_type = "正常平仓"
        exit_idx = len(highs) - 1

        if len(highs) > 1:
            triggers = check_exit_trigger_vectorized(highs[1:], lows[1:], tp_price, sl_price)
            first_trigger = np.argmax(triggers > 0)
            if triggers[first_trigger] > 0:
                exit_idx = first_trigger + 1
                exit_type = "止盈平仓" if triggers[first_trigger] == 1 else "止损平仓"

        close_price = closes[exit_idx]
        close_time = datetimes[exit_idx]

        open_price_slip = open_price * (1 + SLIPPAGE)
        close_price_slip = close_price * (1 - SLIPPAGE)

        trade_size = max_non_holding_capital * params["PEAK_PERCENT"]
        units = trade_size / open_price_slip

        open_fee = trade_size * TAKER_FEE
        funding_fee = trade_size * FUNDING_RATE
        close_value = units * close_price_slip
        close_fee = close_value * TAKER_FEE
        total_fees = open_fee + funding_fee + close_fee

        pnl = units * (close_price_slip - open_price_slip) - total_fees
        new_capital = current_capital + pnl

        if new_capital <= 0:
            new_capital = 0.0
            liquidated = True
            liquidation_date = start_date

        price_change = (close_price - open_price) / open_price

        all_trades.append({
            "时区": params["TIMEZONE"],
            "信号": signal_name,
            "信号分类": target_col,
            "开仓时间": pd.Timestamp(open_time).strftime('%Y-%m-%d %H:%M:%S'),
            "平仓时间": pd.Timestamp(close_time).strftime('%Y-%m-%d %H:%M:%S'),
            "开仓价": round(open_price, 4),
            "平仓价": round(close_price, 4),
            "开仓前资金（U）": round(current_capital, 2),
            "杠杆基准（U）": round(max_non_holding_capital, 2),
            "杠杆倍数": round(trade_size / current_capital if current_capital > 0 else 0, 2),
            "投入(允许杠杆)": round(trade_size, 2),
            "实际涨跌幅": f"{price_change * 100:.2f}%",
            "收益（U）": round(pnl, 2),
            "平仓后资金（U）": round(new_capital, 2),
            "平仓类型": exit_type
        })

        max_non_holding_capital = max(max_non_holding_capital, new_capital)
        current_capital = new_capital
        processed_dates_by_signal[signal_name].add(start_date)

    return all_trades, liquidated, liquidation_date

# 结果保存
def save_results_csv(trades_df):
    """保存CSV结果"""
    csv_path = OUT_DIR / "回测_信号交易结果.csv"
    trades_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(Fore.GREEN + f"已保存交易明细: {csv_path.name}" + Style.RESET_ALL)

# 资金曲线图保存
def save_charts(all_dates, full_curve):
    """保存图表"""
    if not ENABLE_CHARTS:
        return

    plt.figure(figsize=(15, 8))
    plt.plot(all_dates, full_curve, linewidth=2, color='#1f77b4')
    plt.title('回测资金曲线', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('账户资金 (U)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())

    plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', alpha=0.7, label=f'初始资金: {INITIAL_CAPITAL} U')

    final_capital = full_curve[-1]
    plt.annotate(f'最终资金: {final_capital:,.2f} U',
                xy=(all_dates[-1], final_capital),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

    plt.legend()
    plt.tight_layout()
    curve_path = OUT_DIR / "回测资金曲线.png"
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(Fore.GREEN + f"已保存资金曲线图: {curve_path.name}" + Style.RESET_ALL)

# 结果打印
def print_results_summary(stats):
    """打印结果汇总"""
    print("\n" + "="*50)
    print(Fore.YELLOW + "           回测结果统计" + Style.RESET_ALL)
    print("="*50)
    print(Fore.WHITE + f"开始时间:     {stats.get('开始时间', 'N/A')}" + Style.RESET_ALL)
    print(Fore.WHITE + f"结束时间:     {stats.get('结束时间', 'N/A')}" + Style.RESET_ALL)
    print(Fore.GREEN + f"初始资金:     {stats['初始资金']:,.2f} U" + Style.RESET_ALL)
    print(Fore.GREEN + f"最终资金:     {stats['最终资金']:,.2f} U" + Style.RESET_ALL)
    print(Fore.GREEN + f"累计收益率:   {stats['累计收益率']}" + Style.RESET_ALL)
    print("\n" + Fore.GREEN + f"年化收益率:   {stats['年化收益率']}" + Style.RESET_ALL)
    print(Fore.GREEN + f"夏普比率:     {stats['夏普比率']}" + Style.RESET_ALL)
    print(Fore.GREEN + f"覆盖年数:     {stats['覆盖年数']}" + Style.RESET_ALL)
    print(Fore.GREEN + f"最大回撤:     {stats['最大回撤']}" + Style.RESET_ALL)
    print(Fore.GREEN + f"最大回撤时长: {stats['最大回撤时长']} 天" + Style.RESET_ALL)
    print("\n" + Fore.MAGENTA + f"总交易次数:   {stats['总交易次数']}" + Style.RESET_ALL)
    print(Fore.MAGENTA + f"胜率:         {stats['胜率']}" + Style.RESET_ALL)
    print(Fore.MAGENTA + f"盈亏比:       {stats['盈亏比']}" + Style.RESET_ALL)
    print("="*50)


# ==================== 主程序 ====================
if __name__ == '__main__':
    try:
        start_time = time.time()
        print(Fore.YELLOW + "="*50 + "\n    信号交易回测系统\n" + "="*50 + Style.RESET_ALL)
        print(f"{Fore.GREEN}图表生成: {'开启' if ENABLE_CHARTS else '关闭'}{Style.RESET_ALL}\n")

        # 加载数据
        parquet_files = sorted(DATA_DIR.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"在 {DATA_DIR} 未找到任何 parquet 文件")

        print(f"{Fore.CYAN}加载数据...{Style.RESET_ALL}")
        dfs = [pd.read_parquet(p) for p in parquet_files]
        df = pd.concat(dfs, ignore_index=True)

        if df.empty:
            raise RuntimeError("合并后的 DataFrame 为空")

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        print(f"{Fore.GREEN}原始数据时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}{Style.RESET_ALL}")

        # 应用 START_DATE 过滤
        start_date_ts = pd.to_datetime(START_DATE, utc=True)
        df = df[df["datetime"] >= start_date_ts].reset_index(drop=True)

        if df.empty:
            raise RuntimeError(f"起始时间设置为 {START_DATE}，过滤后无数据。"
                             f"请检查数据是否包含该日期之后的记录。")

        print(f"{Fore.GREEN}已应用起始时间过滤 ({START_DATE}){Style.RESET_ALL}")
        print(f"{Fore.GREEN}过滤后数据时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}{Style.RESET_ALL}\n")

        # 执行回测
        print(f"{Fore.CYAN}执行回测...{Style.RESET_ALL}")
        all_trades, liquidated, liquidation_date = run_backtest(df)

        if not all_trades:
            print(Fore.RED + "无有效交易记录" + Style.RESET_ALL)
        else:
            trades_df = pd.DataFrame(all_trades)
            save_results_csv(trades_df)

            all_dates = np.array(sorted(df["datetime"].dt.date.unique()))
            trade_dict = {
                pd.to_datetime(t["开仓时间"]).date(): t["平仓后资金（U）"]
                for t in all_trades
            }
            full_curve = build_full_capital_curve(all_dates, trade_dict, INITIAL_CAPITAL, liquidated, liquidation_date)

            stats = calculate_summary_stats(all_trades, full_curve, all_dates, INITIAL_CAPITAL)
            if all_trades:
                start_time_val = all_trades[0]["开仓时间"]
                end_time_val = all_trades[-1]["开仓时间"]
                stats['开始时间'] = start_time_val[:10]
                stats['结束时间'] = end_time_val[:10]

            print_results_summary(stats)
            save_charts(all_dates, full_curve)

            elapsed = time.time() - start_time
            print(Fore.GREEN + f"\n总耗时: {format_time(elapsed)}" + Style.RESET_ALL)

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}程序被用户中断{Style.RESET_ALL}")
    except Exception as e:
        print(Fore.RED + f"\n程序执行错误: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()