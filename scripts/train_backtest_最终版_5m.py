#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测脚本（优化版）：
- 使用向量化操作提高性能
- 改进时区处理和数据结构
- 完整的资金曲线和统计指标计算
- 非持仓状态最高资金作为杠杆基准
"""

import warnings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import init, Fore, Style
import time

init(autoreset=True)
warnings.filterwarnings("ignore")

# ==================== 配置参数 ====================
TIMEZONE_MAP = {
    "UTC+8": "Etc/GMT-8",
    "UTC0": "UTC",
    **{f"UTC-{i}": f"Etc/GMT+{i}" for i in range(1, 13)},
    **{f"UTC {i}": f"Etc/GMT-{i}" for i in range(11, 12)}
}

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "merged" / "btc" / "btc_5m"
OUT_DIR = ROOT_DIR / "results" / "backtest_最终版"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 信号列名
STAR_COL = "星宿"                       # 星宿信号列
JIAN_XING_COL = "建星"                  # 建星信号列

# 定义目标星宿及其参数
TARGET_STARS = {
    "毕宿": {
        "PEAK_PERCENT": 1,
        "TAKE_PROFIT_PERCENT": 0.156,
        "STOP_LOSS_PERCENT": 0.07,
        "TIMEZONE": "UTC+8"
    },
    "氐宿": {
        "PEAK_PERCENT": 1,
        "TAKE_PROFIT_PERCENT": 0.187,
        "STOP_LOSS_PERCENT": 0.077,
        "TIMEZONE": "UTC+8"
    },
    "参宿": {
        "PEAK_PERCENT": 1,
        "TAKE_PROFIT_PERCENT": 0.151,
        "STOP_LOSS_PERCENT": 0.104,
        "TIMEZONE": "UTC+8"
    },
    "尾宿": {
        "PEAK_PERCENT": 1,
        "TAKE_PROFIT_PERCENT": 0.056,
        "STOP_LOSS_PERCENT": 0.10,
        "TIMEZONE": "UTC0"
    },
    "轸宿": {
        "PEAK_PERCENT": 1,
        "TAKE_PROFIT_PERCENT": 0.107,
        "STOP_LOSS_PERCENT": 0.171,
        "TIMEZONE": "UTC0"
    },
    # 添加建星相关配置
    "除": {
        "PEAK_PERCENT": 1,
        "TAKE_PROFIT_PERCENT": 0.121,
        "STOP_LOSS_PERCENT": 0.094,
        "TIMEZONE": "UTC0"
    }
}

INITIAL_CAPITAL = 1000.0               # 初始资金（USDT）
START_DATE = None                      # 开始日期，格式为 "2020-01-01"，设为 None 表示不限制开始时间

TAKER_FEE, MAKER_FEE, FUNDING_RATE, SLIPPAGE = 0.0005, 0.0003, 0.0002, 0.0005
ENABLE_CHARTS = True           # 是否生成图表

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 工具函数 ====================
def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{int(seconds)}秒"
    elif seconds < 3600:
        return f"{int(seconds // 60)}分{int(seconds % 60)}秒"
    return f"{int(seconds // 3600)}小时{int((seconds % 3600) // 60)}分"


def check_exit_trigger_vectorized(high, low, tp_price, sl_price):
    """向量化检查止盈止损触发 (1=止盈, 2=止损, 0=未触发)"""
    return np.where(high >= tp_price, 1, np.where(low <= sl_price, 2, 0))


def preprocess_star_data(df_raw, star_name, star_params):
    """预处理单个星宿的时区数据"""
    df = df_raw.copy()
    tz_str = TIMEZONE_MAP[star_params["TIMEZONE"]]

    # 确保 UTC 时区
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize('UTC')

    # 转换到目标时区
    df["datetime_tz"] = df["datetime"].dt.tz_convert(tz_str)
    df = df.sort_values("datetime_tz").reset_index(drop=True)
    df["date_tz"] = df["datetime_tz"].dt.floor("D")

    return df


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


def calculate_summary_stats(trades, full_curve, all_dates, initial_capital):
    """计算统计指标"""
    if not trades:
        return {
            "初始资金": initial_capital, "最终资金": initial_capital,
            "累计收益率": "0.00%", "年化收益率": "0.00%",
            "夏普比率": 0, "胜率": "0.00%", "盈亏比": 0,
            "最大回撤": "0.00%", "最大回撤时长": 0, "总交易次数": 0
        }

    # 基础收益指标
    final_capital = full_curve[-1]
    total_return = (final_capital - initial_capital) / initial_capital

    # 修复：正确计算日期差
    date_diff = all_dates[-1] - all_dates[0]
    if isinstance(date_diff, np.timedelta64):
        days = int(date_diff / np.timedelta64(1, 'D')) + 1
    else:
        # 如果是 Python 的 timedelta 对象
        days = date_diff.days + 1

    years = days / 365.0
    annual_return = -1.0 if final_capital <= 0 else (final_capital / initial_capital) ** (1/years) - 1.0

    # 夏普比率
    pnl_arr = np.array([t["收益（U）"] for t in trades])
    returns = pnl_arr / initial_capital
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if len(returns) > 1 and np.std(returns) > 0 else 0

    # 胜率和盈亏比
    wins = np.sum(pnl_arr > 0)
    win_rate = wins / len(pnl_arr)

    win_trades = pnl_arr[pnl_arr > 0]
    loss_trades = pnl_arr[pnl_arr < 0]
    pl_ratio = np.mean(win_trades) / abs(np.mean(loss_trades)) if len(win_trades) > 0 and len(loss_trades) > 0 else 0

    # 最大回撤
    equity = np.array(full_curve)
    cummax = np.maximum.accumulate(equity)
    drawdown = (cummax - equity) / np.where(cummax == 0, 1, cummax)
    max_dd = float(np.max(drawdown))

    # 最大回撤时长
    peak_indices = np.where(equity == cummax)[0]
    max_dd_days = 0
    for i in peak_indices:
        if i < len(equity) - 1:
            recovery_idx = np.where(equity[i+1:] >= equity[i])[0]
            if len(recovery_idx) > 0:
                recovery_idx = recovery_idx[0] + i + 1
                # 修复：正确处理时间差
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


def run_backtest(df_raw):
    """执行完整回测逻辑"""
    all_trades = []
    all_dates_set = set()
    processed_dates_by_star = {star: set() for star in TARGET_STARS}

    # 非持仓最高资金追踪
    max_non_holding_capital = INITIAL_CAPITAL
    current_capital = INITIAL_CAPITAL

    # 收集所有信号，按时间排序
    all_signals = []

    for star_name, star_params in TARGET_STARS.items():
        df_star = preprocess_star_data(df_raw, star_name, star_params)
        tz_str = TIMEZONE_MAP[star_params["TIMEZONE"]]

        # 判断目标列
        target_col = STAR_COL if star_name in ["毕宿", "氐宿", "参宿", "尾宿", "轸宿"] else JIAN_XING_COL

        # 找信号起点（向量化）
        star_mask = (df_star[target_col] == star_name).values
        segment_starts = np.where(star_mask & ~np.roll(star_mask, 1, axis=0))[0]

        for start_idx in segment_starts:
            start_date = df_star.iloc[start_idx]["date_tz"]
            all_signals.append({
                "star_name": star_name,
                "start_idx": start_idx,
                "start_date": start_date,
                "params": star_params,
                "df": df_star,
                "target_col": target_col
            })

    # 按日期排序
    all_signals.sort(key=lambda x: x["start_date"])

    # 按时间顺序执行交易
    liquidated = False
    liquidation_date = None

    for signal in all_signals:
        if liquidated:
            break

        star_name = signal["star_name"]
        start_idx = signal["start_idx"]
        start_date = signal["start_date"]
        params = signal["params"]
        df_star = signal["df"]

        # 检查该星宿该日期是否已交易
        if start_date in processed_dates_by_star[star_name]:
            continue

        # 获取当日所有K线
        day_data = df_star[df_star["date_tz"] == start_date]
        if day_data.empty:
            continue

        # NumPy 数组化加速访问
        dates = day_data["date_tz"].values
        datetimes = day_data["datetime_tz"].values
        opens = day_data["open"].values.astype(float)
        closes = day_data["close"].values.astype(float)
        highs = day_data["high"].values.astype(float)
        lows = day_data["low"].values.astype(float)

        # 开仓
        open_price = opens[0]
        open_time = datetimes[0]

        # 计算止盈止损
        tp_price = open_price * (1 + params["TAKE_PROFIT_PERCENT"])
        sl_price = open_price * (1 - params["STOP_LOSS_PERCENT"])

        # 检查触发
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

        # 滑点处理
        open_price_slip = open_price * (1 + SLIPPAGE)
        close_price_slip = close_price * (1 - SLIPPAGE)

        # 使用非持仓状态最高资金作为杠杆基准
        trade_size = max_non_holding_capital * params["PEAK_PERCENT"]
        units = trade_size / open_price_slip

        # 费用计算
        open_fee = trade_size * TAKER_FEE
        funding_fee = trade_size * FUNDING_RATE
        close_value = units * close_price_slip
        close_fee = close_value * TAKER_FEE
        total_fees = open_fee + funding_fee + close_fee

        # 收益计算（多头）
        pnl = units * (close_price_slip - open_price_slip) - total_fees
        new_capital = current_capital + pnl

        # 检查爆仓
        if new_capital <= 0:
            new_capital = 0.0
            liquidated = True
            liquidation_date = start_date

        price_change = (close_price - open_price) / open_price

        all_trades.append({
            "信号": star_name,
            "信号类型": "建星" if signal["target_col"] == JIAN_XING_COL else "星宿",
            "开仓时间": open_time,
            "平仓时间": close_time,
            "开仓价": round(open_price, 4),
            "平仓价": round(close_price, 4),
            "开仓前资金（U）": round(current_capital, 2),
            "投入(允许杠杆)": round(trade_size, 2),
            "实际涨跌幅": f"{price_change * 100:.2f}%",
            "收益（U）": round(pnl, 2),
            "平仓后资金（U）": round(new_capital, 2),
            "平仓类型": exit_type
        })

        # 更新非持仓最高资金和当前资金
        max_non_holding_capital = max(max_non_holding_capital, new_capital)
        current_capital = new_capital

    return all_trades, all_dates_set, liquidated, liquidation_date


def save_results_csv(trades_df):
    """保存CSV结果"""
    csv_path = OUT_DIR / "回测_信号交易结果.csv"
    trades_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(Fore.GREEN + f"已保存交易明细: {csv_path.name}" + Style.RESET_ALL)


def save_charts(all_dates, full_curve):
    """保存图表"""
    if not ENABLE_CHARTS:
        return

    # 资金曲线图
    plt.figure(figsize=(15, 8))
    plt.plot(all_dates, full_curve, linewidth=2, color='#1f77b4')
    plt.title('回测资金曲线 - 信号交易策略', fontsize=16)
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
    curve_path = OUT_DIR / "回测资金曲线_信号交易策略.png"
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(Fore.GREEN + f"已保存资金曲线图: {curve_path.name}" + Style.RESET_ALL)


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
        print(f"{Fore.GREEN}数据时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}{Style.RESET_ALL}\n")

        # 执行回测
        print(f"{Fore.CYAN}执行回测...{Style.RESET_ALL}")
        all_trades, all_dates_set, liquidated, liquidation_date = run_backtest(df)

        if not all_trades:
            print(Fore.RED + "无有效交易记录" + Style.RESET_ALL)
        else:
            # 保存交易明细
            trades_df = pd.DataFrame(all_trades)
            save_results_csv(trades_df)

            # 构建完整资金曲线
            all_dates = np.array(sorted(df["datetime"].dt.date.unique()))
            trade_dict = {
                pd.to_datetime(t["开仓时间"]).date(): t["平仓后资金（U）"]
                for t in all_trades
            }
            full_curve = build_full_capital_curve(all_dates, trade_dict, INITIAL_CAPITAL, liquidated, liquidation_date)

            # 计算统计指标
            stats = calculate_summary_stats(all_trades, full_curve, all_dates, INITIAL_CAPITAL)
            if all_trades:
                start_time_val = all_trades[0]["开仓时间"]
                end_time_val = all_trades[-1]["开仓时间"]
                # 处理 numpy.datetime64 对象
                stats['开始时间'] = pd.Timestamp(start_time_val).date()
                stats['结束时间'] = pd.Timestamp(end_time_val).date()
            else:
                stats['开始时间'] = None
                stats['结束时间'] = None

            # 显示结果
            print_results_summary(stats)

            # 保存图表
            save_charts(all_dates, full_curve)

            elapsed = time.time() - start_time
            print(Fore.GREEN + f"\n总耗时: {format_time(elapsed)}" + Style.RESET_ALL)

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}程序被用户中断{Style.RESET_ALL}")
    except Exception as e:
        print(Fore.RED + f"\n程序执行错误: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()