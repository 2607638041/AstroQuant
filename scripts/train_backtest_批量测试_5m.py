#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import init, Fore, Style
from multiprocessing import Pool, cpu_count
import time

init(autoreset=True)
warnings.filterwarnings("ignore")

# ==================== 配置参数 ====================
TIMEZONE_MAP = {
    "UTC+8": "Etc/GMT-8",
    "UTC0": "UTC",
    **{f"UTC-{i}": f"Etc/GMT+{i}" for i in range(1, 13)},   # UTC 1~12
    **{f"UTC {i}": f"Etc/GMT-{i}" for i in range(11, 12)}   # UTC 11~12
}

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "merged" / "btc" / "btc_5m"
OUT_DIR = ROOT_DIR / "results" / "backtest_批量测试"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# 策略参数
TIMEZONE = None                          # None=全时区，"UTC8"=单时区
TRADE_DIRECTION = "空"                    # "多"/"空"
STAR_COL = "建星"                      # 信号列名

# 资金管理
INITIAL_CAPITAL = 1000.0        # 初始资金
PEAK_PERCENT = 1                # 仓位比例（>1为杠杆）
TAKE_PROFIT_PERCENT = 1         # 止盈百分比（1 表示 100%）
STOP_LOSS_PERCENT = 1           # 止损百分比（1 表示 100%）
START_DATE = None               # 开始日期，格式为 "2020-01-01"，设为 None 表示不限制开始时间

# 交易成本
TAKER_FEE, MAKER_FEE, FUNDING_RATE, SLIPPAGE = 0.0005, 0.0003, 0.0002, 0.0005

NUM_PROCESSES = max(1, cpu_count() - 3)     #用CPU核心并行总数数减1,（避免占用全部 CPU 资源，保留系统运行空间）
ENABLE_CHARTS = True                        #是否开启 “图表功能”，

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 工具函数 ====================
def format_time(seconds):
    # 格式化时间显示
    if seconds < 60:
        return f"{int(seconds)}秒"
    elif seconds < 3600:
        return f"{int(seconds // 60)}分{int(seconds % 60)}秒"
    return f"{int(seconds // 3600)}小时{int((seconds % 3600) // 60)}分"


def calculate_slippage_price(price, is_long, is_open):
    # 计算含滑点的成交价
    factor = 1 + SLIPPAGE if is_open else 1 - SLIPPAGE
    return price * factor if is_long else price / factor


def check_exit_trigger_vectorized(high, low, tp_price, sl_price, is_long):
    # 向量化检查止盈止损触发 (1=止盈, 2=止损, 0=未触发)
    if is_long:
        return np.where(high >= tp_price, 1, np.where(low <= sl_price, 2, 0))
    return np.where(low <= tp_price, 1, np.where(high >= sl_price, 2, 0))


def preprocess_timezone_data(df_raw, tz_str):
    # 预处理时区数据
    df = df_raw.copy()
    df["datetime_tz"] = df["datetime"].dt.tz_convert(tz_str)
    df = df.sort_values("datetime_tz").reset_index(drop=True)
    df["date_tz"] = df["datetime_tz"].dt.floor("D")
    return df


# ==================== 回测核心逻辑 ====================
def build_full_curve(all_dates, processed_dates, capital_curve, initial_capital, liquidated, liq_date):
    # 构建完整资金曲线
    full_curve = []
    last_capital = initial_capital
    j = 0
    sorted_dates = sorted(processed_dates)

    for date in all_dates:
        if j < len(sorted_dates) and sorted_dates[j] == date:
            full_curve.append(capital_curve[j])
            last_capital = capital_curve[j]
            j += 1
        else:
            full_curve.append(last_capital)

    if liquidated and liq_date:
        full_curve = [0.0 if date > liq_date else val for date, val in zip(all_dates, full_curve)]

    return full_curve


def calculate_summary_stats(trades, full_curve, all_dates, target_star, tz_name, liquidated, initial_capital):
    # 计算统计指标
    # 基础收益率
    final_capital = full_curve[-1]
    days = (all_dates[-1] - all_dates[0]).astype('timedelta64[D]').astype(int) + 1
    years = days / 365.0
    total_return = (final_capital - initial_capital) / initial_capital
    annual_return = -1.0 if final_capital <= 0 else (final_capital / initial_capital) ** (1/years) - 1.0

    # 夏普比率
    pnl_arr = np.array([t["收益_USD"] for t in trades])
    returns = pnl_arr / initial_capital
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if len(returns) > 1 and np.std(returns) > 0 else 0

    # 胜率
    wins = np.sum(pnl_arr > 0)
    win_rate = wins / len(pnl_arr)

    # 盈亏比
    win_trades = pnl_arr[pnl_arr > 0]
    loss_trades = pnl_arr[pnl_arr < 0]
    pl_ratio = np.mean(win_trades) / abs(np.mean(loss_trades)) if len(win_trades) > 0 and len(loss_trades) > 0 else 0

    # 最大回撤与最大回撤时长
    equity = np.array(full_curve)
    cummax = np.maximum.accumulate(equity)
    drawdown = (cummax - equity) / np.where(cummax == 0, 1, cummax)
    max_dd = float(np.max(drawdown))

    peak_indices = np.where(equity == cummax)[0]
    max_dd_days = 0
    for i in peak_indices:
        if i < len(equity) - 1:
            recovery_idx = np.where(equity[i+1:] >= equity[i])[0]
            recovery_idx = recovery_idx[0] + i + 1 if len(recovery_idx) > 0 else len(equity) - 1
            dd_duration = (all_dates[recovery_idx] - all_dates[i]).astype('timedelta64[D]').astype(int)
            max_dd_days = max(max_dd_days, dd_duration)

    # 时区和杠杆计算
    tz_num = int(tz_name.replace("UTC", "")) if tz_name != "UTC0" else 0
    optimal_leverage = 0.2 / max_dd if max_dd > 0 else 0
    adjusted_annual_return = annual_return * optimal_leverage

    return {
        "最佳杠杆": round(optimal_leverage, 2),
        "触发信号": target_star,
        "时区": tz_num,
        "中国时间": (8 - tz_num) % 24,
        "累计收益率": f"{total_return * 100:.2f}%",
        "年化收益率": f"{adjusted_annual_return * 100:.2f}%",
        "夏普比率": round(sharpe, 2),
        "覆盖年数": round(years, 2),
        "胜率": f"{win_rate * 100:.2f}%",
        "盈亏比": round(pl_ratio, 2),
        "最大回撤": f"{max_dd * 100:.2f}%",
        "最大回撤时长": max_dd_days
    }


def create_empty_summary(target_star, tz_name):
    # 创建空结果摘要
    tz_num = int(tz_name.replace("UTC", "")) if tz_name != "UTC0" else 0
    return {
        "最佳杠杆": 0,
        "触发信号": target_star,
        "时区": tz_num,
        "中国时间": (8 - tz_num) % 24,
        "累计收益率": "0.00%",
        "年化收益率": "0.00%",
        "夏普比率": 0,
        "覆盖年数": 0,
        "胜率": "0.00%",
        "盈亏比": 0,
        "最大回撤": "0.00%",
        "最大回撤时长": 0
    }


def run_single_star_backtest(args):
    # 单个信号单个时区的回测（支持多进程）
    df, target_star, tz_name, tz_str = args

    try:
        # 识别信号段
        star_mask = (df[STAR_COL] == target_star).values
        segment_starts = np.where(star_mask & ~np.roll(star_mask, 1, axis=0))[0]

        if len(segment_starts) == 0:
            return create_empty_summary(target_star, tz_name), None, None, None, None

        # 初始化交易参数
        processed_dates = set()
        capital = float(INITIAL_CAPITAL)
        daily_closes = []
        trades = []
        capital_curve = []
        liquidated = False
        is_long = (TRADE_DIRECTION == "多")

        # 转换为NumPy数组加速访问
        dates = df["date_tz"].values
        datetimes = df["datetime_tz"].values
        opens = df["open"].values.astype(float)
        closes = df["close"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)

        # 遍历每个信号触发点
        for start_idx in segment_starts:
            if liquidated:
                break

            start_date = dates[start_idx]
            if start_date in processed_dates:
                continue

            # 获取当天所有K线
            day_mask = dates == start_date
            day_indices = np.where(day_mask)[0]

            if len(day_indices) == 0:
                continue

            # 开仓
            first_idx = day_indices[0]
            open_price = opens[first_idx]
            open_time = datetimes[first_idx]

            # 计算止盈止损价格
            tp_price = open_price * (1 + TAKE_PROFIT_PERCENT) if is_long else open_price * (1 - TAKE_PROFIT_PERCENT)
            sl_price = open_price * (1 - STOP_LOSS_PERCENT) if is_long else open_price * (1 + STOP_LOSS_PERCENT)

            # 检查是否触发止盈止损
            exit_type = "正常平仓"
            exit_idx = day_indices[-1]

            check_indices = day_indices[1:]
            if len(check_indices) > 0:
                triggers = check_exit_trigger_vectorized(highs[check_indices], lows[check_indices], tp_price, sl_price, is_long)
                first_trigger = np.argmax(triggers > 0)
                if triggers[first_trigger] > 0:
                    exit_idx = check_indices[first_trigger]
                    exit_type = "止盈平仓" if triggers[first_trigger] == 1 else "止损平仓"

            # 平仓
            close_price = closes[exit_idx]
            close_time = datetimes[exit_idx]

            # 计算滑点后价格
            open_price_slip = calculate_slippage_price(open_price, is_long, True)
            close_price_slip = calculate_slippage_price(close_price, is_long, False)

            # 计算仓位
            peak = max(daily_closes) if daily_closes else INITIAL_CAPITAL
            trade_size = peak * PEAK_PERCENT
            units = trade_size / open_price_slip

            # 计算费用和收益
            open_fee = trade_size * TAKER_FEE
            funding_fee = trade_size * FUNDING_RATE
            close_value = units * close_price_slip
            close_fee = close_value * TAKER_FEE
            total_fees = open_fee + funding_fee + close_fee

            pnl = units * (close_price_slip - open_price_slip) - total_fees if is_long else units * (open_price_slip - close_price_slip) - total_fees
            prev_capital = capital
            capital += pnl

            # 检查爆仓
            if capital <= 0:
                capital = 0.0
                liquidated = True

            daily_closes.append(capital)
            processed_dates.add(start_date)

            price_change = (close_price - open_price) / open_price * (1 if is_long else -1)

            trades.append({
                "开仓时间": open_time,
                "平仓时间": close_time,
                "开仓价": open_price,
                "平仓价": close_price,
                "方向": TRADE_DIRECTION,
                "开仓前资金_USD": round(prev_capital, 8),
                "投入(允许杠杆)": round(trade_size, 8),
                "实际涨跌幅": f"{price_change * 100:.2f}%",
                "收益_USD": round(pnl, 8),
                "平仓后资金_USD": round(capital, 8),
                "平仓类型": exit_type
            })
            capital_curve.append(capital)

        if not trades:
            return create_empty_summary(target_star, tz_name), None, None, None, None

        # 构建完整资金曲线并计算统计
        all_dates = np.unique(dates)
        full_curve = build_full_curve(all_dates, processed_dates, capital_curve, INITIAL_CAPITAL, liquidated, trades[-1]["开仓时间"] if trades else None)
        summary = calculate_summary_stats(trades, full_curve, all_dates, target_star, tz_name, liquidated, INITIAL_CAPITAL)

        return summary, pd.DataFrame(trades), all_dates, full_curve, (target_star, tz_name)

    except Exception as e:
        print(f"\n{Fore.RED}错误 [{target_star}-{tz_name}]: {e}{Style.RESET_ALL}")
        return create_empty_summary(target_star, tz_name), None, None, None, None


# ==================== 结果保存 ====================
def save_results(target_star, tz_name, trades_df, all_dates, full_curve):
    # 保存CSV和图表（每个信号独立文件夹）
    if trades_df is None or trades_df.empty:
        return

    star_dir = OUT_DIR / str(target_star)
    star_dir.mkdir(parents=True, exist_ok=True)

    csv_path = star_dir / f"回测_{target_star}_{tz_name}_{TRADE_DIRECTION}.csv"
    trades_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    if ENABLE_CHARTS:
        plt.figure(figsize=(15, 8))
        plt.plot(all_dates, full_curve, linewidth=2, color='#1f77b4')
        plt.title(f'资金曲线 - {target_star} - {tz_name} - {TRADE_DIRECTION}单', fontsize=16)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('资金 (USDT)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
        plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', alpha=0.7, label=f'初始: {INITIAL_CAPITAL}')

        final_capital = full_curve[-1] if full_curve else INITIAL_CAPITAL
        plt.annotate(f'最终: {final_capital:.2f}', xy=(all_dates[-1], final_capital),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
        plt.legend()
        plt.tight_layout()

        img_path = star_dir / f"回测资金曲线_{target_star}_{tz_name}_{TRADE_DIRECTION}.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()


# ==================== 主程序 ====================
if __name__ == '__main__':
    try:
        start_time = time.time()
        print(Fore.YELLOW + "="*50 + "\n    批量回测系统\n" + "="*50 + Style.RESET_ALL)
        print(f"{Fore.GREEN}并行进程数: {NUM_PROCESSES} | 图表生成: {'开启' if ENABLE_CHARTS else '关闭'}{Style.RESET_ALL}\n")

        # 第一步：读取并合并数据
        parquet_files = sorted(DATA_DIR.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"在 {DATA_DIR} 未找到任何 parquet 文件")

        dfs = [pd.read_parquet(p) for p in parquet_files]
        df = pd.concat(dfs, ignore_index=True)
        if df.empty:
            raise RuntimeError("合并后的 DataFrame 为空")

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

        # 第二步：获取信号列表并排序
        unique_stars = df[STAR_COL].dropna().unique()
        star_first_occurrence = {star: df[df[STAR_COL] == star].index[0] for star in unique_stars}
        all_xiuxiu = sorted(unique_stars, key=lambda x: star_first_occurrence[x])
        print(f"{Fore.GREEN}信号数量: {len(all_xiuxiu)}{Style.RESET_ALL}")

        # 第三步：确定时区列表
        timezones = [TIMEZONE] if TIMEZONE else list(TIMEZONE_MAP.keys())
        print(f"{Fore.GREEN}测试时区: {TIMEZONE if TIMEZONE else f'全部 ({len(timezones)}个)'}{Style.RESET_ALL}\n")

        # 第四步：预处理时区数据
        print(f"{Fore.CYAN}预处理时区数据...{Style.RESET_ALL}")
        tz_data = {tz_name: preprocess_timezone_data(df, TIMEZONE_MAP[tz_name]) for tz_name in timezones}

        # 第五步：准备并执行回测任务
        tasks = [(tz_data[tz_name], target_star, tz_name, TIMEZONE_MAP[tz_name])
                 for target_star in all_xiuxiu for tz_name in timezones]
        total_tasks = len(tasks)
        print(f"{Fore.CYAN}开始回测 ({total_tasks} 个任务)...{Style.RESET_ALL}\n")

        all_summaries = []
        completed = 0
        task_start_time = time.time()

        with Pool(processes=NUM_PROCESSES) as pool:
            for result in pool.imap_unordered(run_single_star_backtest, tasks):
                summary, trades_df, all_dates, full_curve, info = result
                all_summaries.append(summary)

                if info is not None:
                    target_star, tz_name = info
                    save_results(target_star, tz_name, trades_df, all_dates, full_curve)

                completed += 1
                progress = completed / total_tasks
                bar_length = 40
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)

                elapsed = time.time() - task_start_time
                avg_time = elapsed / completed if completed > 0 else 0
                remaining_time = avg_time * (total_tasks - completed)

                print(f"\r{Fore.CYAN}[{bar}] {progress*100:.1f}% ({completed}/{total_tasks}) | "
                      f"已用:{format_time(elapsed)} | 剩余:{format_time(remaining_time)}{Style.RESET_ALL}",
                      end='', flush=True)

        print()

        # 第六步：保存汇总结果并显示
        if all_summaries:
            summary_df = pd.DataFrame(all_summaries)

            summary_df['年化收益率_数值'] = summary_df['年化收益率'].str.rstrip('%').astype(float)
            summary_df = summary_df.sort_values('年化收益率_数值', ascending=False)
            summary_df = summary_df.drop(columns=['年化收益率_数值'])

            summary_csv_path = OUT_DIR / f"回测_汇总表_所有信号所有时区_{TRADE_DIRECTION}.csv"
            summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

            elapsed_time = time.time() - start_time
            print(f"\n{Fore.YELLOW}{'='*50}")
            print("           回测完成")
            print(f"{'='*50}{Style.RESET_ALL}")
            print(Fore.GREEN + f"汇总文件: {summary_csv_path.name}" + Style.RESET_ALL)
            print(Fore.GREEN + f"结果目录: {OUT_DIR}" + Style.RESET_ALL)
            print(Fore.GREEN + f"总耗时: {elapsed_time:.2f} 秒" + Style.RESET_ALL)

            # 显示TOP 10结果
            top_columns = ['最佳杠杆', '触发信号', '时区', '中国时间', '累计收益率', '年化收益率', '夏普比率', '覆盖年数', '胜率', '盈亏比', '最大回撤', '最大回撤时长']
            top_df = summary_df[top_columns].head(10)

            print(f"\n{Fore.CYAN}TOP 10 年化收益率:{Style.RESET_ALL}")
            print(f"{'序号':>4} {'最佳杠杆':>8} {'信号':>6} {'时区':>4} {'中国':>4} {'累计':>10} {'年化':>10} {'夏普':>6} {'年数':>6} {'胜率':>8} {'盈亏比':>6} {'回撤':>8} {'回撤天':>6}")
            print("-" * 120)
            for i, (_, row) in enumerate(top_df.iterrows(), 1):
                print(f"{i:>4} {float(row['最佳杠杆']):>8.2f} {row['触发信号']:>6} {row['时区']:>4} {row['中国时间']:>4} "
                      f"{row['累计收益率']:>10} {row['年化收益率']:>10} {float(row['夏普比率']):>6.2f} {float(row['覆盖年数']):>6.2f} "
                      f"{row['胜率']:>8} {float(row['盈亏比']):>6.2f} {row['最大回撤']:>8} {row['最大回撤时长']:>6}")
        else:
            print(Fore.RED + "\n无成功回测结果" + Style.RESET_ALL)

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}程序被用户中断{Style.RESET_ALL}")
    except Exception as e:
        print(Fore.RED + f"\n程序执行错误: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()