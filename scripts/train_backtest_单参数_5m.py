#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多时区回测系统：信号日开盘买入，当日收盘或触发止盈止损时平仓"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
from colorama import init, Fore, Style

init(autoreset=True)
warnings.filterwarnings("ignore")

# 时区映射
TIMEZONE_MAP = {
    "UTC+8": "Etc/GMT-8",
    "UTC0": "UTC",
    **{f"UTC-{i}": f"Etc/GMT+{i}" for i in range(1, 13)},   # UTC 1~12
    **{f"UTC {i}": f"Etc/GMT-{i}" for i in range(11, 12)}   # UTC 11~12
}

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "merged" / "btc" / "btc_5m"
OUT_DIR = ROOT_DIR / "results" / "backtest_多时区"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 策略参数
TIMEZONE = "UTC0"                           # None=全时区，"UTC+8"=单时区
TRADE_DIRECTION = "多"                    # "多"/"空"
STAR_COL, TARGET_STAR = "星宿", "轸宿"     # 信号列名，信号参数

# 资金管理
INITIAL_CAPITAL = 1000.0        # 初始资金
PEAK_PERCENT = 1.5                # 仓位比例（>1为杠杆）
TAKE_PROFIT_PERCENT = 1         # 止盈百分比（1 表示 100%）
STOP_LOSS_PERCENT = 1           # 止损百分比（1 表示 100%）
START_DATE = None               # 开始日期，格式为 "2020-01-01"，设为 None 表示不限制开始时间

# 交易成本
TAKER_FEE, MAKER_FEE, FUNDING_RATE, SLIPPAGE = 0.0005, 0.0003, 0.0002, 0.0005

# 绘图配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def print_single_timezone_result(summary, trades_df, all_dates, full_curve):
    """单时区详细统计模式 - 表格形式"""
    if trades_df.empty:
        print(Fore.YELLOW + "未生成交易" + Style.RESET_ALL)
        return

    final_capital = full_curve[-1]
    start_date = all_dates[0].strftime("%Y-%m-%d")
    end_date = all_dates[-1].strftime("%Y-%m-%d")

    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100 if INITIAL_CAPITAL != 0 else 0

    # 构建详细统计表
    print(Fore.CYAN + "=" * 50)
    print(f"          {TARGET_STAR}回测结果统计")
    print("=" * 50 + Style.RESET_ALL)

    print(Fore.CYAN + f"开始时间:     {start_date}" + Style.RESET_ALL)
    print(Fore.CYAN + f"结束时间:     {end_date}" + Style.RESET_ALL)
    print(Fore.YELLOW + f"初始资金:     {INITIAL_CAPITAL:>15,.2f} U" + Style.RESET_ALL)
    print(Fore.YELLOW + f"最终资金:     {final_capital:>15,.2f} U" + Style.RESET_ALL)
    print(Fore.GREEN + f"累计收益率:   {total_return:>15.2f}%" + Style.RESET_ALL)
    print()
    print(Fore.GREEN + f"年化收益率:   {summary['年化收益率']:>15.2f}%" + Style.RESET_ALL)
    print(Fore.GREEN + f"夏普比率:     {summary['夏普比率']:>15.2f}" + Style.RESET_ALL)
    print(Fore.CYAN + f"覆盖年数:     {summary['覆盖年数']:>15.2f}" + Style.RESET_ALL)
    print(Fore.RED + f"最大回撤:     {summary['最大回撤']:>15.2f}%" + Style.RESET_ALL)
    print(Fore.RED + f"最大回撤时长: {summary['最大回撤时长']:>15} 天" + Style.RESET_ALL)
    print()
    print(Fore.CYAN + f"总交易次数:   {summary['交易次数']:>15}" + Style.RESET_ALL)
    print(Fore.GREEN + f"盈利次数:     {summary['盈利次数']:>15}" + Style.RESET_ALL)
    print(Fore.RED + f"亏损次数:     {summary['亏损次数']:>15}" + Style.RESET_ALL)
    print(Fore.GREEN + f"胜率:         {summary['胜率']:>15.2f}%" + Style.RESET_ALL)
    print(Fore.GREEN + f"盈亏比:       {summary['盈亏比']:>15.2f}" + Style.RESET_ALL)

    print(Fore.CYAN + "=" * 50 + Style.RESET_ALL)


def print_multi_timezone_progress(current, total):
    """多时区进度条模式 - 动态更新"""
    percent = current / total
    bar_length = 40
    filled = int(bar_length * percent)
    bar = '█' * filled + '░' * (bar_length - filled)

    # 使用 \r 实现动态覆盖
    print(f"\r处理进度: [{bar}] {current}/{total}", end='', flush=True)


def calculate_slippage_price(price, is_long, is_open):
    """计算含滑点的成交价"""
    factor = SLIPPAGE if (is_long == is_open) else -SLIPPAGE
    return price * (1 + factor)


def check_exit_trigger(row, open_time, tp_price, sl_price, is_long):
    """检查是否触发止盈止损"""
    if row["datetime"] <= open_time:
        return None

    high, low = float(row["high"]), float(row["low"])

    if is_long:
        if high >= tp_price:
            return "止盈平仓"
        if low <= sl_price:
            return "止损平仓"
    else:
        if low <= tp_price:
            return "止盈平仓"
        if high >= sl_price:
            return "止损平仓"
    return None


def run_backtest(df_raw, tz_name, tz_str):
    """单时区回测主函数"""
    df = df_raw.copy()
    df["datetime_tz"] = df["datetime"].dt.tz_convert(tz_str)
    df = df.sort_values("datetime_tz").reset_index(drop=True)
    df["date_tz"] = df["datetime_tz"].dt.floor("D")

    # 识别信号
    star_mask = (df[STAR_COL] == TARGET_STAR)
    segment_starts = df.index[star_mask & (~star_mask.shift(1, fill_value=False))].tolist()

    # 初始化
    processed_dates = set()
    capital, daily_closes, trades, capital_curve = float(INITIAL_CAPITAL), [], [], []
    liquidated = False
    is_long = (TRADE_DIRECTION == "多")

    # 遍历信号
    for start_idx in segment_starts:
        if liquidated or start_idx not in df.index:
            break

        start_date = df.loc[start_idx, 'date_tz']    # 信号日期
        if start_date in processed_dates:
            continue

        day_rows = df[df['date_tz'] == start_date]
        if day_rows.empty:
            continue

        # 开仓
        open_row = day_rows.iloc[0]
        open_price = float(open_row['open'])
        open_time = open_row['datetime_tz']

        tp_price = open_price * (1 + TAKE_PROFIT_PERCENT if is_long else -TAKE_PROFIT_PERCENT)
        sl_price = open_price * (1 - STOP_LOSS_PERCENT if is_long else STOP_LOSS_PERCENT)

        # 检测止盈止损
        exit_type = "正常平仓"
        exit_row = day_rows.iloc[-1]

        for _, row in day_rows.iterrows():
            trigger = check_exit_trigger(row, open_time, tp_price, sl_price, is_long)
            if trigger:
                exit_type = trigger
                exit_row = row
                break

        # 平仓计算
        close_price = float(exit_row["close"])
        open_price_slip = calculate_slippage_price(open_price, is_long, True)
        close_price_slip = calculate_slippage_price(close_price, is_long, False)

        peak = max(daily_closes) if daily_closes else INITIAL_CAPITAL
        trade_size = peak * PEAK_PERCENT
        # 防止除以零
        if open_price_slip == 0:
            units = 0
        else:
            units = trade_size / open_price_slip

        fees = trade_size * (TAKER_FEE + FUNDING_RATE) + units * close_price_slip * TAKER_FEE
        pnl = units * (close_price_slip - open_price_slip) * (1 if is_long else -1) - fees

        prev_capital = capital
        capital += pnl

        if capital <= 0:
            capital = 0.0
            liquidated = True

        daily_closes.append(capital)
        processed_dates.add(start_date)

        # 价格变动按方向
        price_change = (close_price - open_price) / open_price * (1 if is_long else -1) if open_price != 0 else 0
        trades.append({
            "开仓时间": open_time, "平仓时间": exit_row['datetime_tz'],
            "开仓价": open_price, "平仓价": close_price, "方向": TRADE_DIRECTION,
            "开仓前资金_USD": round(prev_capital, 8), "投入(允许杠杆)": round(trade_size, 8),
            "实际涨跌幅": f"{price_change*100:.2f}%", "收益_USD": round(pnl, 8),
            "平仓后资金_USD": round(capital, 8), "平仓类型": exit_type
        })
        capital_curve.append(capital)

    # 补齐完整日期
    all_dates = sorted(df["date_tz"].unique())
    full_curve = []
    last_capital = INITIAL_CAPITAL
    j = 0
    sorted_dates = sorted(processed_dates)

    for date in all_dates:
        if j < len(sorted_dates) and sorted_dates[j] == date:
            full_curve.append(capital_curve[j])
            last_capital = capital_curve[j]
            j += 1
        else:
            full_curve.append(last_capital)

    if liquidated and trades:
        liq_date = trades[-1]["开仓时间"]
        full_curve = [0.0 if date > liq_date else val for date, val in zip(all_dates, full_curve)]

    # 计算指标
    if not trades:
        return create_empty_summary(tz_name), pd.DataFrame(), all_dates, full_curve

    final_capital = full_curve[-1]
    days = (all_dates[-1] - all_dates[0]).days + 1
    years = days / 365.0 if days > 0 else 1.0

    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL if INITIAL_CAPITAL != 0 else 0
    annual_return = -1.0 if final_capital <= 0 else (final_capital / INITIAL_CAPITAL) ** (1/years) - 1.0

    pnl_arr = [t["收益_USD"] for t in trades]
    returns = [pnl / INITIAL_CAPITAL for pnl in pnl_arr] if INITIAL_CAPITAL != 0 else [0 for _ in pnl_arr]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if len(returns) > 1 and np.std(returns) > 0 else 0

    wins = sum(1 for pnl in pnl_arr if pnl > 0)
    win_rate = wins / len(pnl_arr) if pnl_arr else 0

    win_trades = [pnl for pnl in pnl_arr if pnl > 0]
    loss_trades = [pnl for pnl in pnl_arr if pnl < 0]
    pl_ratio = np.mean(win_trades) / abs(np.mean(loss_trades)) if win_trades and loss_trades else 0

    equity = np.array(full_curve)
    cummax = np.maximum.accumulate(equity)
    drawdown = (cummax - equity) / np.where(cummax == 0, 1, cummax)
    max_dd = float(np.nanmax(drawdown)) if drawdown.size > 0 else 0

    max_dd_days = max((
        (next((all_dates[j] for j in range(i+1, len(equity)) if equity[j] >= equity[i]), all_dates[-1]) - all_dates[i]).days
        for i in range(len(equity)) if equity[i] == cummax[i]
    ), default=0)

    # 安全解析时区数字（处理 UTC0 / UTC-1 / UTC1 等）
    tz_num_str = tz_name.replace("UTC", "")
    try:
        tz_num = int(tz_num_str)
    except Exception:
        # 尝试处理形如 'UTC-1'
        try:
            tz_num = int(tz_num_str.replace('-', ''))
        except Exception:
            tz_num = 0

    summary = {
        "时区": tz_num, "初始资金": INITIAL_CAPITAL, "结束资金": round(final_capital, 2),
        "累计收益率": round(total_return * 100, 2), "年化收益率": round(annual_return * 100, 2),
        "夏普比率": round(sharpe, 2), "覆盖天数": days, "覆盖年数": round(years, 2),
        "交易次数": len(trades), "盈利次数": wins, "亏损次数": len(pnl_arr) - wins,
        "胜率": round(win_rate * 100, 2), "盈亏比": round(pl_ratio, 2),
        "最大回撤": round(max_dd * 100, 2), "最大回撤时长": max_dd_days, "是否爆仓": liquidated,
        "中国时间": (8 - tz_num) % 24
    }

    return summary, pd.DataFrame(trades), all_dates, full_curve


def create_empty_summary(tz_name):
    """创建空结果摘要"""
    tz_num_str = tz_name.replace("UTC", "")
    try:
        tz_num = int(tz_num_str)
    except Exception:
        try:
            tz_num = int(tz_num_str.replace('-', ''))
        except Exception:
            tz_num = 0
    return {
        "时区": tz_num, "初始资金": INITIAL_CAPITAL, "结束资金": INITIAL_CAPITAL,
        "累计收益率": 0, "年化收益率": 0, "夏普比率": 0, "覆盖天数": 0, "覆盖年数": 0,
        "交易次数": 0, "盈利次数": 0, "亏损次数": 0, "胜率": 0, "盈亏比": 0,
        "最大回撤": 0, "最大回撤时长": 0, "是否爆仓": False,
        "中国时间": (8 - tz_num) % 24
    }


def save_results(tz_name, trades_df, all_dates, full_curve):
    """保存CSV和图表"""
    if trades_df.empty:
        return

    # 保存CSV
    csv_path = OUT_DIR / f"回测_交易记录_{tz_name}_{TRADE_DIRECTION}.csv"
    trades_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(Fore.GREEN + f"  CSV: {csv_path.name}" + Style.RESET_ALL)

    # 绘图
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(all_dates, full_curve, linewidth=2)
    ax.set_title(f'资金曲线 - {tz_name} - {TRADE_DIRECTION}单', fontsize=16)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('资金 (USDT)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    ax.axhline(y=INITIAL_CAPITAL, linestyle='--', alpha=0.7, label=f'初始: {INITIAL_CAPITAL}')
    if full_curve:
        ax.annotate(f'最终: {full_curve[-1]:.2f}', xy=(all_dates[-1], full_curve[-1]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
    ax.legend()
    plt.tight_layout()

    img_path = OUT_DIR / f"回测资金曲线_{tz_name}_{TRADE_DIRECTION}.png"
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(Fore.GREEN + f"  图表: {img_path.name}" + Style.RESET_ALL)


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 加载数据
    parquet_files = sorted(DATA_DIR.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"未找到数据: {DATA_DIR}")

    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # 确定时区
    timezones = list(TIMEZONE_MAP.keys()) if TIMEZONE is None else [TIMEZONE]
    is_multi_tz = (TIMEZONE is None)

    if is_multi_tz:
        print(Fore.GREEN + f"测试时区数: {len(timezones)}\n" + Style.RESET_ALL)

    all_summaries = []
    total_tz = len(timezones)

    # 执行回测
    for idx, tz_name in enumerate(timezones, 1):
        try:
            summary, trades_df, all_dates, full_curve = run_backtest(df, tz_name, TIMEZONE_MAP[tz_name])
            all_summaries.append(summary)

            if is_multi_tz:
                # 多时区：进度条模式
                print_multi_timezone_progress(idx, total_tz)
            else:
                # 单时区：详细统计模式
                print_single_timezone_result(summary, trades_df, all_dates, full_curve)
                save_results(tz_name, trades_df, all_dates, full_curve)

        except Exception as e:
            if is_multi_tz:
                print(Fore.RED + f" 错误" + Style.RESET_ALL)
            else:
                print(Fore.RED + f"  失败: {e}\n" + Style.RESET_ALL)

    # 进度条完成换行
    if is_multi_tz and all_summaries:
        print()  # 换行

    # 生成汇总（仅多时区）
    if all_summaries and is_multi_tz:
        summary_df = pd.DataFrame(all_summaries).sort_values('年化收益率', ascending=False)

        # 添加最佳杠杆列
        # 最佳杠杆 = 0.2 / 最大回撤率
        summary_df['最佳杠杆'] = (0.2 / (summary_df['最大回撤'] / 100)).round(2)
        # 调整年化收益率 = 原年化收益率 * 最佳杠杆
        summary_df['年化收益率'] = (summary_df['年化收益率'] * summary_df['最佳杠杆'] / 100).round(2)

        # 删除指定列
        columns_to_remove = ['初始资金', '结束资金', '覆盖天数', '交易次数', '盈利次数', '亏损次数', '是否爆仓']
        summary_df = summary_df.drop(columns=columns_to_remove)

        # 生成一个专用于保存的字符串格式化 DataFrame（百分号形式）
        formatted_df = summary_df.copy()
        percent_cols = ['累计收益率', '年化收益率', '胜率', '最大回撤']
        for col in percent_cols:
            if col in formatted_df.columns:
                # 如果是 NaN 或非数值，保护处理
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{float(x):.2f}%" if pd.notnull(x) else "")

        summary_path = OUT_DIR / f"回测_汇总表_{TRADE_DIRECTION}.csv"
        formatted_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

        print(f"{Fore.YELLOW}{'='*50}\n          完成\n{'='*50}{Style.RESET_ALL}")
        print(Fore.GREEN + f"汇总: {summary_path.name}\n" + Style.RESET_ALL)
        print(f"{Fore.CYAN}TOP 5:{Style.RESET_ALL}")
        top_columns = ['时区', '中国时间', '最佳杠杆', '累计收益率', '年化收益率', '夏普比率', '胜率', '盈亏比', '最大回撤', '最大回撤时长']
        headers = ['  时区', '中国时间', '  最佳杠杆', '累计收益率', '年化收益率', '夏普比率', '    胜率', '  盈亏比', '  最大回撤', '最大回撤时长']
        header_str = ''.join([f'{h:>10}' for h in headers])
        print(header_str)

        # 为终端打印保留数值格式（便于对齐），从 summary_df 里取前 5 行并按格式化打印
        print(summary_df[top_columns].head(5).to_string(
            index=False,
            header=False,
            formatters={
                '时区': '{:>10}'.format,
                '中国时间': '{:>10}'.format,
                '最佳杠杆': '{:>10.2f}'.format,
                '累计收益率': lambda x: f"{float(x):>12.2f}%",
                '年化收益率': lambda x: f"{float(x):>10.2f}%",
                '夏普比率': '{:>12.2f}'.format,
                '胜率': lambda x: f"{float(x):>12.2f}%",
                '盈亏比': '{:>9.2f}'.format,
                '最大回撤': lambda x: f"{float(x):>10.2f}%",
                '最大回撤时长': '{:>11}'.format
            }
        ))
    elif not all_summaries:
        print(Fore.RED + "无成功回测" + Style.RESET_ALL)