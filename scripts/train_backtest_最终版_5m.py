#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测脚本（更新版，含年化收益率）：
- 毕宿/柳宿开盘买入，当日收盘平仓
- 每次名义仓位 = 历史日结后的最高总资金 * 参数配置（PEAK_PERCENT）
- 允许杠杆（不限制 trade_size 相对于当前资金）
- 若平仓后资金 <= 0，视为爆仓（资金归零），记录并终止回测
- 输入：项目根目录 data/merged/btc/btc_5m 下的 parquet 文件
- 输出：results/backtest_5m 下的 CSV/PNG（包含年化收益率）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
from colorama import init, Fore, Style

init(autoreset=True)  # 初始化colorama并启用自动重置
warnings.filterwarnings("ignore")
TIMEZONE_MAP = {"UTC0": "UTC", "UTC8": "Asia/Shanghai"}

# ------------------ 配置 ------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "merged" / "btc" / "btc_5m"
OUT_DIR = ROOT_DIR / "results" / "backtest_5m"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAR_COL = "星宿"
JIAN_XING_COL = "十二建星"  # 添加建星列支持
# 定义目标星宿及其参数
TARGET_STARS = {
    "毕宿": {
        "PEAK_PERCENT": 1.3,
        "TAKE_PROFIT_PERCENT": 0.156,
        "STOP_LOSS_PERCENT": 0.07,
        "TIMEZONE": "UTC8"
    },
    "氐宿": {
        "PEAK_PERCENT": 0.8,
        "TAKE_PROFIT_PERCENT": 0.187,
        "STOP_LOSS_PERCENT": 0.077,
        "TIMEZONE": "UTC8"
    },
    "参宿": {
        "PEAK_PERCENT": 0.8,
        "TAKE_PROFIT_PERCENT": 0.151,
        "STOP_LOSS_PERCENT": 0.104,
        "TIMEZONE": "UTC8"
    },
    "尾宿": {
        "PEAK_PERCENT": 0.8,
        "TAKE_PROFIT_PERCENT": 0.056,
        "STOP_LOSS_PERCENT": 0.10,
        "TIMEZONE": "UTC0"
    },
    "轸宿": {
        "PEAK_PERCENT": 1.5,
        "TAKE_PROFIT_PERCENT": 0.107,
        "STOP_LOSS_PERCENT": 0.171,
        "TIMEZONE": "UTC0"
    },
    # 添加建星相关配置
    "闭": {
        "PEAK_PERCENT": 1.0,
        "TAKE_PROFIT_PERCENT": 0.082,
        "STOP_LOSS_PERCENT": 0.116,
        "TIMEZONE": "UTC8"
    }
}

INITIAL_CAPITAL = 1000.0  # 初始资金（USDT）

# 回测时间范围配置
START_DATE = None     # 开始日期，格式为 "2020-01-01"，设为 None 表示不限制开始时间
END_DATE = None       # 结束日期，格式为 "2023-12-31"，设为 None 表示不限制结束时间

# 手续费、资金费率和滑点配置
TAKER_FEE = 0.0005  # 永续合约吃单手续费（0.05%）
MAKER_FEE = 0.0002  # 永续合约挂单手续费（0.02%）
FUNDING_RATE = 0.0001  # 日资金费率（0.03%）
SLIPPAGE = 0.0005  # 滑点（0.05%）

# 解决中文乱码显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ------------------ 数据加载与预处理 ------------------
def load_and_prepare_data(data_dir: Path) -> pd.DataFrame:
    """
    加载并准备数据，确保时间列正确解析为datetime类型
    """
    # 查找所有Parquet文件
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"在 {data_dir} 未找到任何 parquet 文件，请检查路径或先运行合并脚本。")
    
    # 读取所有Parquet文件
    dfs = []
    for p in parquet_files:
        try:
            df_tmp = pd.read_parquet(p)
            dfs.append(df_tmp)
        except Exception as e:
            print(f"[警告] 读取文件失败: {p} -> {e}")
    
    # 合并所有数据
    df = pd.concat(dfs, ignore_index=True)
    if df.empty:
        raise RuntimeError("合并后的 DataFrame 为空，无法回测。")

    # 确保 datetime 列被正确解析为 datetime 类型
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s')  # 假设是Unix时间戳
    else:
        raise ValueError("数据中缺少 'datetime' 列")

    # 创建 UTC 和 CN 时间列
    df['date_utc'] = df['datetime'].dt.date
    df['time_utc'] = df['datetime'].dt.time

    # 设置时区转换
    timezone = TIMEZONE_MAP.get("UTC8", "Asia/Shanghai")
    df['datetime_cn'] = df['datetime'].dt.tz_convert(timezone)
    df['date_cn'] = df['datetime_cn'].dt.date
    df['time_cn'] = df['datetime_cn'].dt.time

    # 根据配置的时间范围过滤数据
    if START_DATE is not None:
        start_date = pd.to_datetime(START_DATE).tz_localize('UTC')
        df = df[df['datetime'] >= start_date]
    if END_DATE is not None:
        end_date = pd.to_datetime(END_DATE).tz_localize('UTC')
        df = df[df['datetime'] <= end_date]
    
    # 再次验证过滤后的时间范围
    print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")

    return df

# 加载数据
df = load_and_prepare_data(DATA_DIR)

# ------------------ 回测初始化 ------------------
# 全局资金历史记录，用于实现仓位只增不减原则
global_history = [INITIAL_CAPITAL]
all_trades = []
processed_dates_by_star = {star_name: set() for star_name in TARGET_STARS}

# 收集所有信号并按时间排序
all_signals = []
for star_name, star_params in TARGET_STARS.items():
    # 获取星宿特定的时区设置，如果没有设置应该报错而不是使用默认值
    if "TIMEZONE" not in star_params:
        raise ValueError(f"星宿 {star_name} 缺少 TIMEZONE 配置")
    star_timezone = star_params["TIMEZONE"]
    if star_timezone not in TIMEZONE_MAP:
        raise ValueError(f"星宿 {star_name} 的时区设置 '{star_timezone}' 无效，支持的时区: {list(TIMEZONE_MAP.keys())}")
    star_timezone = TIMEZONE_MAP[star_timezone]

    # 为当前星宿创建带有时区信息的数据副本
    df_star = df.copy()
    df_star["datetime_cn"] = df_star["datetime"].dt.tz_convert(star_timezone)
    df_star = df_star.sort_values("datetime_cn").reset_index(drop=True)
    df_star["date_cn"] = df_star["datetime_cn"].dt.floor("D")

    # 判断是星宿还是建星
    target_col = STAR_COL if star_name in ["毕宿", "氐宿", "参宿", "尾宿", "轸宿"] else JIAN_XING_COL
    
    star_mask = (df_star[target_col] == star_name)
    segment_starts = df_star.index[star_mask & (~star_mask.shift(1, fill_value=False))].tolist()

    for start_idx in segment_starts:
        if start_idx not in df_star.index:
            continue
        start_row = df_star.loc[start_idx]
        start_date = start_row['date_cn']
        all_signals.append({
            'star_name': star_name,
            'start_idx': start_idx,
            'start_date': start_date,
            'star_params': star_params,
            'df': df_star,  # 传递带有时区信息的数据帧
            'timezone': star_timezone,
            'target_col': target_col  # 添加目标列信息
        })

# 按日期排序所有信号
all_signals.sort(key=lambda x: x['start_date'])
print(Fore.BLUE + f"总共找到 {len(all_signals)} 个信号" + Style.RESET_ALL)

# 按时间顺序执行交易
for signal in all_signals:
    star_name = signal['star_name']
    start_idx = signal['start_idx']
    start_date = signal['start_date']
    star_params = signal['star_params']
    df_with_tz = signal['df']  # 使用信号中传递的带有时区信息的数据帧
    target_col = signal['target_col']  # 获取目标列

    # 检查该星宿在该日期是否已交易
    if start_date in processed_dates_by_star[star_name]:
        continue

    # 提取当日所有记录
    day_rows = df_with_tz[df_with_tz['date_cn'] == start_date]
    if day_rows.empty:
        continue

    # 开仓信息 - 使用当日第一条记录
    open_row = day_rows.iloc[0]
    open_price = float(open_row['open'])
    open_time = open_row['datetime_cn']

    # 计算止盈止损价格
    stop_loss_price = open_price * (1 - star_params["STOP_LOSS_PERCENT"])
    take_profit_price = open_price * (1 + star_params["TAKE_PROFIT_PERCENT"])

    # 查找是否触发止盈止损
    exit_type = "正常平仓"
    exit_row = day_rows.iloc[-1]

    # 遍历当天数据，检查是否触发止盈止损
    for idx, row in day_rows.iterrows():
        if row["datetime_cn"] <= open_time:
            continue

        high_price = float(row["high"])
        low_price = float(row["low"])

        # 检查是否触发止盈
        if high_price >= take_profit_price:
            exit_type = "止盈平仓"
            exit_row = row
            break

        # 检查是否触发止损
        if low_price <= stop_loss_price:
            exit_type = "止损平仓"
            exit_row = row
            break

    # 平仓信息
    close_price = float(exit_row["close"])
    close_time = exit_row["datetime_cn"]

    # 计算考虑滑点后的实际成交价
    open_price_slippage = open_price * (1 + SLIPPAGE)
    close_price_slippage = close_price * (1 - SLIPPAGE)

    # 名义仓位 = 整体账户历史日结后的最高资金 * PEAK_PERCENT
    peak_close = max(global_history)
    nominal_size = peak_close * star_params["PEAK_PERCENT"]
    trade_size = float(nominal_size)

    # 计算单位与 pnl（多头）
    units = trade_size / open_price_slippage

    # 计算手续费和资金费
    open_fee = trade_size * TAKER_FEE
    funding_fee = trade_size * FUNDING_RATE
    close_value = units * close_price_slippage
    close_fee = close_value * TAKER_FEE

    # 总费用
    total_fees = open_fee + funding_fee + close_fee

    # 计算净收益
    pnl = units * (close_price_slippage - open_price_slippage) - total_fees

    # 更新全局资金历史（用于后续仓位计算）
    new_global_capital = peak_close + pnl  # 简化的全局资金更新
    global_history.append(new_global_capital)

    # 记录交易
    processed_dates_by_star[star_name].add(start_date)

    # 计算实际涨跌幅
    price_change = (close_price - open_price) / open_price

    all_trades.append({
        "信号": star_name,
        "信号类型": "建星" if target_col == JIAN_XING_COL else "星宿",
        "开仓时间": open_time,
        "平仓时间": close_time,
        "开仓价": open_price,
        "平仓价": close_price,
        "方向": "多",
        "名义资金（U）": round(peak_close, 2),
        "投入(允许杠杆)": round(trade_size, 2),
        "实际涨跌幅": f"{price_change * 100:.2f}%",
        "收益（U）": round(pnl, 2),
        "平仓后资金（U）": round(new_global_capital, 2),
        "平仓类型": exit_type
    })

# ------------------ 输出 ------------------
trades_df = pd.DataFrame(all_trades)

# ------------------ 统计指标 ------------------
if not trades_df.empty:
    # 计算关键指标
    final_capital = all_trades[-1]["平仓后资金（U）"]
    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # 计算年化收益率
    first_trade_date = min(trade["开仓时间"] for trade in all_trades)
    last_trade_date = max(trade["开仓时间"] for trade in all_trades)
    days = (last_trade_date - first_trade_date).days
    years = days / 365.0 if days > 0 else 1.0
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # 计算夏普比率（假设无风险收益率为0）
    returns = [t["收益（U）"] / INITIAL_CAPITAL for t in all_trades]
    if len(returns) > 1 and np.std(returns) != 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(len(returns))
    else:
        sharpe_ratio = 0

    # 计算胜率和盈亏比
    pnl_arr = [t["收益（U）"] for t in all_trades]
    wins = sum(1 for pnl in pnl_arr if pnl > 0)
    win_rate = wins / len(pnl_arr) if pnl_arr else 0

    # 计算盈亏比（平均盈利 / 平均亏损）
    winning_trades = [pnl for pnl in pnl_arr if pnl > 0]
    losing_trades = [pnl for pnl in pnl_arr if pnl < 0]

    if winning_trades and losing_trades:
        avg_win = np.mean(winning_trades)
        avg_loss = abs(np.mean(losing_trades))  # 取绝对值
        profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
    else:
        profit_loss_ratio = 0

    # 计算最大回撤和最大回撤时长
    capital_curve = [INITIAL_CAPITAL]
    trade_times = [pd.to_datetime(all_trades[0]["开仓时间"]).date()]  # 初始化第一个交易日

    for trade in all_trades:
        capital_curve.append(trade["平仓后资金（U）"])
        trade_times.append(pd.to_datetime(trade["平仓时间"]).date())

    peak = INITIAL_CAPITAL
    peak_index = 0
    max_drawdown = 0
    max_drawdown_duration = 0

    for i, capital in enumerate(capital_curve):
        if capital > peak:
            peak = capital
            peak_index = i

        drawdown = (peak - capital) / peak if peak > 0 else 0
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        # 计算回撤持续时间
        if i > peak_index and trade_times:
            # 确保索引在范围内
            if i < len(trade_times) and peak_index < len(trade_times):
                drawdown_duration = (trade_times[i] - trade_times[peak_index]).days
                if drawdown_duration > max_drawdown_duration:
                    max_drawdown_duration = drawdown_duration

    # 美化输出
    print("\n" + "=" * 50)
    print(Fore.YELLOW + "           回测结果统计" + Style.RESET_ALL)
    print("=" * 50)
    print(Fore.WHITE + f"开始时间:     {first_trade_date.date()}" + Style.RESET_ALL)
    print(Fore.WHITE + f"结束时间:     {last_trade_date.date()}" + Style.RESET_ALL)
    print(Fore.GREEN + f"初始资金:     {INITIAL_CAPITAL:,.2f} U" + Style.RESET_ALL)
    print(Fore.GREEN + f"结束资金:     {final_capital:,.2f} U" + Style.RESET_ALL)
    print(Fore.GREEN + f"累计收益率:   {total_return * 100:,.2f}%" + Style.RESET_ALL)
    print("\n" + Fore.GREEN + f"年化收益率:   {annualized_return * 100:.2f}%" + Style.RESET_ALL)
    print(Fore.GREEN + f"夏普比率:     {sharpe_ratio:.2f}" + Style.RESET_ALL)
    print(Fore.GREEN + f"最大回撤:     {max_drawdown * 100:.2f}%" + Style.RESET_ALL)
    print(Fore.GREEN + f"最大回撤时长:  {max_drawdown_duration} 天" + Style.RESET_ALL)
    print("\n" + Fore.MAGENTA + f"总交易次数:    {len(all_trades)}" + Style.RESET_ALL)
    print(Fore.MAGENTA + f"盈利/亏损:    {wins}/{len(pnl_arr) - wins}" + Style.RESET_ALL)
    print(Fore.MAGENTA + f"胜率:         {win_rate * 100:.2f}%" + Style.RESET_ALL)
    print(Fore.MAGENTA + f"盈亏比:       {profit_loss_ratio:.2f}" + Style.RESET_ALL)
    print("=" * 50)
else:
    print("\n" + "=" * 50)
    print(Fore.YELLOW + "           回测结果统计" + Style.RESET_ALL)
    print("=" * 50)
    print(Fore.RED + "无有效交易记录" + Style.RESET_ALL)
    print("=" * 50)

# 保存交易明细
csv_trades = OUT_DIR / "回测_组合星宿开盘平仓.csv"
trades_df.to_csv(csv_trades, index=False, encoding="utf-8-sig")
print(Fore.GREEN + f"\n已保存逐笔交易明细: {csv_trades}" + Style.RESET_ALL)

# ------------------ 生成资金曲线图 ------------------
# 构建完整的资金曲线，包含所有日期
all_dates = sorted(df["date_cn"].unique())
full_capital_curve = [INITIAL_CAPITAL] * len(all_dates)
last_capital = INITIAL_CAPITAL

# 按日期更新资金曲线
trade_idx = 0
for i, date in enumerate(all_dates):
    # 确保date是datetime对象并获取日期部分
    if hasattr(date, 'date'):
        date_only = date.date()
    else:
        date_only = pd.to_datetime(date).date()

    # 查找在该日期是否有交易
    while trade_idx < len(all_trades):
        trade_date = all_trades[trade_idx]["开仓时间"]
        # 确保trade_date是datetime对象
        if isinstance(trade_date, str):
            trade_date = pd.to_datetime(trade_date)
        trade_date_only = trade_date.date()

        if trade_date_only == date_only:
            full_capital_curve[i] = all_trades[trade_idx]["平仓后资金（U）"]
            last_capital = full_capital_curve[i]
            trade_idx += 1
        elif trade_date_only < date_only:
            trade_idx += 1
        else:
            break
    if i > 0 and full_capital_curve[i] == INITIAL_CAPITAL:
        full_capital_curve[i] = last_capital

# 检查是否发生爆仓，如果发生则后续资金为0
liquidated = False
for i, trade in enumerate(all_trades):
    if trade["平仓后资金（U）"] <= 0:
        liquidated = True
        liquidation_time = trade["开仓时间"]
        # 确保liquidation_time是datetime对象
        if isinstance(liquidation_time, str):
            liquidation_time = pd.to_datetime(liquidation_time)
        liquidation_date = liquidation_time.date()
        # 从爆仓日期之后，资金都为0
        for j, date in enumerate(all_dates):
            # 确保date是datetime对象并获取日期部分
            if hasattr(date, 'date'):
                date_only = date.date()
            else:
                date_only = pd.to_datetime(date).date()

            if date_only > liquidation_date:
                full_capital_curve[j] = 0.0
        break

# 绘制资金曲线图
plt.figure(figsize=(15, 8))
plt.plot(all_dates, full_capital_curve, linewidth=2, color='#1f77b4')
plt.title('回测资金曲线 - 星宿策略组合', fontsize=16)
plt.xlabel('日期', fontsize=12)
plt.ylabel('账户资金 (U)', fontsize=12)
plt.grid(True, alpha=0.3)

# 设置x轴日期格式，使用年度刻度避免标签过于密集
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
plt.xticks(rotation=0)

# 根据START_DATE和END_DATE设置x轴显示范围
if START_DATE is not None:
    plt.xlim(left=pd.to_datetime(START_DATE))
if END_DATE is not None:
    plt.xlim(right=pd.to_datetime(END_DATE))

# 设置y轴范围和刻度，提高可读性
y_min, y_max = min(full_capital_curve), max(full_capital_curve)
y_range = y_max - y_min
# 添加10%的边距
y_margin = y_range * 0.1 if y_range > 0 else 100  # 如果范围为0，使用默认边距
plt.ylim(y_min - y_margin, y_max + y_margin)

# 添加初始资金参考线
plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', alpha=0.7, label=f'初始资金: {INITIAL_CAPITAL} U')

# 标注最终资金
final_capital = full_capital_curve[-1] if full_capital_curve else INITIAL_CAPITAL
plt.annotate(f'最终资金: {final_capital:,.2f} U',
             xy=(all_dates[-1] if all_dates else 0, final_capital),
             xytext=(10, 10), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

plt.legend()
plt.tight_layout()
capital_curve_path = OUT_DIR / "回测资金曲线_星宿策略组合.png"
plt.savefig(capital_curve_path, dpi=300, bbox_inches='tight')
plt.close()
print(Fore.GREEN + f"已保存资金曲线图: {capital_curve_path}" + Style.RESET_ALL)

# ------------------ 计算并绘制滚动年化收益 ------------------
if len(all_trades) > 0:
    # 创建一个包含日期和资金的DataFrame
    capital_df = pd.DataFrame({
        'date': all_dates,
        'capital': full_capital_curve
    })

    # 计算每日收益率
    capital_df['returns'] = capital_df['capital'].pct_change()

    # 计算滚动年化收益（每1年计算CAGR）
    rolling_cagr = []
    rolling_dates = []

    # 确保有足够数据进行至少一年的计算
    # 注意：只有当回测时间跨度超过365天时，才会开始计算滚动年化收益
    # 第一个滚动CAGR值出现在第365天之后，确保每一期都是完整滚动一年的年化收益
    if len(capital_df) >= 365:
        for i in range(365, len(capital_df)):
            # 取过去365天作为滚动窗口（完整一年）
            start_capital = capital_df.iloc[i - 365]['capital']
            end_capital = capital_df.iloc[i]['capital']

            # 计算CAGR (复合年增长率)，仅当起始资金大于0时有效
            if start_capital > 0:
                cagr = (end_capital / start_capital) ** (1.0 / 1) - 1  # 1年期年化收益率
                rolling_cagr.append(cagr)
                rolling_dates.append(capital_df.iloc[i]['date'])  # 记录该CAGR对应的结束日期

    # 绘制滚动年化收益曲线
    if rolling_cagr:
        plt.figure(figsize=(15, 8))
        plt.plot(rolling_dates, rolling_cagr, linewidth=1.5, color='#2ca02c', antialiased=True)
        plt.title('滚动年化收益曲线 (1年期CAGR)', fontsize=18, fontweight='bold')
        plt.xlabel('日期', fontsize=14, fontweight='bold')
        plt.ylabel('年化收益率', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 设置x轴日期格式，使用年度刻度避免标签过于密集
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
        plt.xticks(rotation=0, fontsize=12)
        plt.yticks(fontsize=12)
        
        # 添加0%参考线
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)

        # 获取Y轴数据范围
        y_min, y_max = min(rolling_cagr), max(rolling_cagr)
        margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.5
        plt.ylim(y_min - margin, y_max + margin)

        # 设置Y轴主刻度：每0.5（即50%）一个刻度
        ax = plt.gca()
        ax.yaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(0.5))
        # 格式化为百分比，保留两位小数
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
        
        # 添加图例
        plt.legend(['1年期CAGR'], loc='upper left', fontsize=12)
        
        # 根据START_DATE和END_DATE设置x轴显示范围
        if START_DATE is not None:
            plt.xlim(left=pd.to_datetime(START_DATE))
        if END_DATE is not None:
            plt.xlim(right=pd.to_datetime(END_DATE))

        plt.tight_layout()
        rolling_cagr_path = OUT_DIR / "滚动年化收益曲线.png"
        plt.savefig(rolling_cagr_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(Fore.GREEN + f"已保存滚动年化收益曲线: {rolling_cagr_path}" + Style.RESET_ALL)

print(Fore.GREEN + "回测完成" + Style.RESET_ALL)