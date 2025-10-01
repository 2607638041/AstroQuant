#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测脚本（更新版，含年化收益率）：
- 所有开仓信号开盘买入，当日收盘平仓（每个信号单独测试）
- 每次名义仓位 = 历史日结后的最高总资金 * 20%（PEAK_PERCENT=0.20）
- 允许杠杆（不限制 trade_size 相对于当前资金）
- 若平仓后资金 <= 0，视为爆仓（资金归零），记录并终止回测
- 输入：项目根目录 data/merged/btc/btc_5m 下的 parquet 文件
- 输出：results/backtest_5m 下的 CSV/PNG（包含年化收益率，每个信号单独输出）
"""

import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from colorama import init, Fore, Style

init(autoreset=True)  # 初始化colorama并启用自动重置
warnings.filterwarnings("ignore")  # 忽略所有警告信息
TIMEZONE_MAP = {"UTC0": "UTC", "UTC8": "Asia/Shanghai"}

# ------------------ 配置 ------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "merged" / "btc" / "btc_5m"
OUT_DIR = ROOT_DIR / "results" / "backtest_5m"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEZONE = "UTC8"  # 时区选择
if TIMEZONE in TIMEZONE_MAP:  # 如果TIMEZONE是简写形式，则转换为标准形式
    TIMEZONE = TIMEZONE_MAP[TIMEZONE]
TRADE_DIRECTION = "多"  # 交易方向
STAR_COL = "十二建星"    # 信号类型

INITIAL_CAPITAL = 1000.0 # 初始资金（USDT）
PEAK_PERCENT = 1         # 杠杆比
TAKE_PROFIT_PERCENT = 1  # 止盈百分比
STOP_LOSS_PERCENT = 1    # 止损百分比

# 回测时间范围配置
START_DATE = None  # 开始日期，格式为 "2018-01-01"，设为 None 表示不限制开始时间

# 手续费、资金费率和滑点配置
TAKER_FEE = 0.0005  # 永续合约吃单手续费（0.05%）
MAKER_FEE = 0.0002  # 永续合约挂单手续费（0.02%）
FUNDING_RATE = 0.0003  # 日资金费率（0.03%）
SLIPPAGE = 0.0005  # 滑点（0.05%）

# 解决中文乱码显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

try:
    # ------------------ 读取数据 ------------------
    parquet_files = sorted(DATA_DIR.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"在 {DATA_DIR} 未找到任何 parquet 文件，请检查路径或先运行合并脚本。")

    dfs = []
    for p in parquet_files:
        try:
            df_tmp = pd.read_parquet(p)
            dfs.append(df_tmp)
        except Exception as e:
            print(f"[警告] 读取文件失败: {p} -> {e}")

    df = pd.concat(dfs, ignore_index=True)
    if df.empty:
        raise RuntimeError("合并后的 DataFrame 为空，无法回测。")

    # ------------------ 时间处理 ------------------
    if "datetime" not in df.columns:
        raise RuntimeError("数据缺少 'datetime' 列")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["datetime_cn"] = df["datetime"].dt.tz_convert(TIMEZONE)
    df = df.sort_values("datetime_cn").reset_index(drop=True)
    df["date_cn"] = df["datetime_cn"].dt.floor("D")

    # 从合并后的数据中获取实际使用的信号列表
    all_xiuxiu = sorted(df[STAR_COL].dropna().unique().tolist())
    print(f"从合并数据中获取到信号列表: {all_xiuxiu}")

    # 存储所有的回测结果
    all_summaries = []

    # 为每个信号单独进行回测
    for index, TARGET_STAR in enumerate(all_xiuxiu, 1):
        print(f"\n{Fore.CYAN + f'开始测试第 {index}/{len(all_xiuxiu)} 个: {TARGET_STAR}'}{Style.RESET_ALL}")

        # ------------------ 识别信号区间 ------------------
        # 查找所有当前信号
        star_mask = (df[STAR_COL] == TARGET_STAR)
        # 识别信号区间的开始点（当前为True，前一个为False或不存在）
        segment_starts = df.index[star_mask & (~star_mask.shift(1, fill_value=False))].tolist()

        print(Fore.BLUE + f"找到 {len(segment_starts)}个 {TARGET_STAR} 信号" + Style.RESET_ALL)

        # 记录已处理的日期，避免同一天多次开仓
        processed_dates = set()

        # ------------------ 回测初始化 ------------------
        capital = float(INITIAL_CAPITAL)
        daily_closes = []  # 记录每个交易日平仓后的资金（仅日结后值用于peak计算）
        trades = []
        capital_curve = []
        liquidated = False  # 标记是否爆仓并终止

        # 遍历所有信号区间的开始点
        for start_idx in segment_starts:
            if liquidated:
                break

            if start_idx not in df.index:
                continue

            # 获取信号开始的行
            start_row = df.loc[start_idx]
            start_date = start_row['date_cn']  # UTC+8 的日

            # 同一天已经交易过，跳过（保证每个 UTC+8 日最多一次交易）
            if start_date in processed_dates:
                continue

            # 提取当日所有记录
            day_rows = df[df['date_cn'] == start_date]
            if day_rows.empty:
                # 极端情况：没有找到当日数据，跳过
                continue

            # 开仓信息 - 使用当日第一条记录而不是segment start的时间点
            open_row = day_rows.iloc[0]  # 当日首条记录
            open_price = float(open_row['open'])
            open_time = open_row['datetime_cn']

            # 计算止盈止损价格
            if TRADE_DIRECTION == "多":
                stop_loss_price = open_price * (1 - STOP_LOSS_PERCENT)  # 止损价格
                take_profit_price = open_price * (1 + TAKE_PROFIT_PERCENT)  # 止盈价格
            else:  # 空单
                stop_loss_price = open_price * (1 + STOP_LOSS_PERCENT)  # 止损价格（价格上涨）
                take_profit_price = open_price * (1 - TAKE_PROFIT_PERCENT)  # 止盈价格（价格下跌）

            # 查找是否触发止盈止损
            exit_type = "正常平仓"  # 默认为正常平仓（当日收盘）
            exit_row = day_rows.iloc[-1]  # 默认平仓行为当日收盘

            # 遍历当天数据，检查是否触发止盈止损
            for idx, row in day_rows.iterrows():
                # 只检查开仓时间之后的数据
                if row["datetime"] <= open_time:
                    continue

                high_price = float(row["high"])
                low_price = float(row["low"])

                # 检查是否触发止盈或止损
                if TRADE_DIRECTION == "多":
                    # 多单：检查是否触发止盈（价格上涨）或止损（价格下跌）
                    if high_price >= take_profit_price:
                        exit_type = "止盈平仓"
                        exit_row = row
                        break

                    if low_price <= stop_loss_price:
                        exit_type = "止损平仓"
                        exit_row = row
                        break
                else:  # 空单
                    # 空单：检查是否触发止盈（价格下跌）或止损（价格上涨）
                    if low_price <= take_profit_price:
                        exit_type = "止盈平仓"
                        exit_row = row
                        break

                    if high_price >= stop_loss_price:
                        exit_type = "止损平仓"
                        exit_row = row
                        break

            # 平仓：根据是否触发止盈止损决定平仓时间点
            close_price = float(exit_row["close"])
            close_time = exit_row["datetime_cn"]

            # 计算考虑滑点后的实际成交价
            if TRADE_DIRECTION == "多":
                open_price_slippage = open_price * (1 + SLIPPAGE)  # 开仓时的滑点（买入时价格稍高）
                close_price_slippage = close_price * (1 - SLIPPAGE)  # 平仓时的滑点（卖出时价格稍低）
            else:  # 空单
                open_price_slippage = open_price * (1 - SLIPPAGE)  # 开仓时的滑点（卖出时价格稍低）
                close_price_slippage = close_price * (1 + SLIPPAGE)  # 平仓时的滑点（买入时价格稍高）

            # 名义仓位 = 历史日结后的最高资金 * PEAK_PERCENT
            # 如果 daily_closes 为空（即第一笔交易），则使用初始资金
            peak_close = max(daily_closes) if daily_closes else INITIAL_CAPITAL
            nominal_size = peak_close * PEAK_PERCENT
            trade_size = float(nominal_size)  # 允许杠杆：不限制 trade_size 相对于当前 capital

            # 计算单位
            units = trade_size / open_price_slippage

            # 计算手续费和资金费
            open_fee = trade_size * TAKER_FEE  # 开仓手续费（吃单）
            funding_fee = trade_size * FUNDING_RATE  # 资金费
            close_value = units * close_price_slippage  # 平仓价值
            close_fee = close_value * TAKER_FEE  # 平仓手续费（吃单）

            # 总费用
            total_fees = open_fee + funding_fee + close_fee

            # 计算净收益
            if TRADE_DIRECTION == "多":
                pnl = units * (close_price_slippage - open_price_slippage) - total_fees
            else:  # 空单
                pnl = units * (open_price_slippage - close_price_slippage) - total_fees

            prev_capital = capital
            new_capital = capital + pnl

            # 记录交易
            is_liquidated = False
            if new_capital <= 0:
                # 爆仓：资金归零，记录并停止后续交易
                is_liquidated = True
                new_capital = 0.0
                liquidated = True

            capital = new_capital
            daily_closes.append(capital)  # 把当日结算后的资金写入历史峰值计算集合（用于之后天数）
            processed_dates.add(start_date)  # 标记该日期已处理

            # 计算实际涨跌幅
            if TRADE_DIRECTION == "多":
                price_change = (close_price - open_price) / open_price
            else:  # 空单
                price_change = (open_price - close_price) / open_price

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
            # 每完成一笔交易（代表一个交易日结束），就将当前 capital 加入 capital_curve
            capital_curve.append(capital)

        # ------------------ 补齐资本曲线到所有日期 ------------------
        all_dates = sorted(df["date_cn"].unique())
        full_capital_curve = []
        last_capital = INITIAL_CAPITAL

        # 构建完整的资金曲线，包含所有日期
        j = 0  # 用于追踪capital_curve的索引
        processed_dates_list = sorted(list(processed_dates))

        for i, date in enumerate(all_dates):
            # 查找是否有在该日期交易的记录
            while j < len(processed_dates_list) and processed_dates_list[j] < date:
                # 跳过之前未处理的交易日
                j += 1

            if j < len(processed_dates_list) and processed_dates_list[j] == date:
                # 这一天有交易
                full_capital_curve.append(capital_curve[j])
                last_capital = capital_curve[j]
                j += 1  # 移动到下一个交易日
            else:
                # 这一天没有交易，使用前一个资金值
                full_capital_curve.append(last_capital)

        # 如果发生爆仓，需要将爆仓后的所有日期资金设为0
        if liquidated and trades:
            liquidated_date = trades[-1]["开仓时间"]
            for i, date in enumerate(all_dates):
                if date > liquidated_date:
                    full_capital_curve[i] = 0.0

        # ------------------ 输出 ------------------
        trades_df = pd.DataFrame(trades)
        csv_trades = OUT_DIR / f"回测_{TARGET_STAR}开盘平仓_peakClose20pct_leverage_{TRADE_DIRECTION}.csv"
        trades_df.to_csv(csv_trades, index=False, encoding="utf-8-sig")

        # 绘制资金曲线图
        plt.figure(figsize=(15, 8))
        plt.plot(all_dates, full_capital_curve, linewidth=2, color='#1f77b4')
        plt.title(f'回测资金曲线 - {TARGET_STAR}策略 - {TRADE_DIRECTION}单', fontsize=16)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('账户资金 (USDT)', fontsize=12)
        plt.grid(True, alpha=0.3)

        # 设置x轴日期格式，使用年度刻度避免标签过于密集
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
        plt.xticks(rotation=0)

        # 添加初始资金参考线
        plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', alpha=0.7, label=f'初始资金: {INITIAL_CAPITAL:.2f} USDT')

        # 标注最终资金
        final_capital = full_capital_curve[-1] if full_capital_curve else INITIAL_CAPITAL
        plt.annotate(f'最终资金: {final_capital:.2f} USDT',
                     xy=(all_dates[-1] if all_dates else 0, final_capital),
                     xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

        plt.legend()
        plt.tight_layout()
        capital_curve_path = OUT_DIR / f"回测资金曲线_{TARGET_STAR}策略_{TRADE_DIRECTION}单.png"
        plt.savefig(capital_curve_path, dpi=300, bbox_inches='tight')
        plt.close()

        # ------------------ 统计指标 ------------------
        if not trades_df.empty:
            # 计算关键指标
            final_capital = full_capital_curve[-1] if full_capital_curve else INITIAL_CAPITAL
            total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL

            # 计算年化收益率
            start_date = all_dates[0] if all_dates else None
            end_date = all_dates[-1] if all_dates else None
            if start_date is not None and end_date is not None:
                days = (end_date - start_date).days + 1
                years = days / 365.0 if days > 0 else 1.0
                if final_capital <= 0:
                    annualized_return = -1.0  # 归零即 -100%
                else:
                    annualized_return = (final_capital / INITIAL_CAPITAL) ** (1.0 / years) - 1.0
            else:
                annualized_return = 0.0

            # 计算夏普比率（假设无风险收益率为0）
            returns = [t["收益_USD"] / INITIAL_CAPITAL for t in trades]
            if len(returns) > 1 and np.std(returns) != 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(len(returns))
            else:
                sharpe_ratio = 0

            # 计算胜率和盈亏比
            pnl_arr = [t["收益_USD"] for t in trades]
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
            equity = np.array(full_capital_curve, dtype=float)
            if len(equity) > 0:
                cummax = np.maximum.accumulate(equity)
                drawdown = (cummax - equity) / np.where(cummax == 0, 1, cummax)
                max_dd = float(np.nanmax(drawdown)) if len(drawdown) > 0 else 0.0

                # 计算最大回撤时长（从峰值到恢复到峰值的时间）
                max_drawdown_duration = 0

                # 从后向前查找每个峰值及其恢复时间
                i = len(equity) - 1
                while i >= 0:
                    # 当前点是峰值点（不小于之前任何点）
                    if equity[i] == cummax[i]:
                        peak_value = equity[i]
                        peak_date = all_dates[i]

                        # 向后查找恢复点（恢复到峰值或创新高）
                        recovery_date = None
                        for j in range(i + 1, len(equity)):
                            if equity[j] >= peak_value:
                                recovery_date = all_dates[j]
                                break

                        # 计算回撤持续时间
                        if recovery_date is not None:
                            # 在回测结束前恢复了
                            drawdown_duration = (recovery_date - peak_date).days
                        else:
                            # 到回测结束都未恢复
                            drawdown_duration = (all_dates[-1] - peak_date).days

                        max_drawdown_duration = max(max_drawdown_duration, drawdown_duration)

                    i -= 1
            else:
                max_dd = 0.0
                max_drawdown_duration = 0

            # 美化输出
            print("=" * 50)
            print(Fore.YELLOW + f"           回测结果统计 - {TARGET_STAR}" + Style.RESET_ALL)
            print("=" * 50)
            print(Fore.WHITE + f"开始时间:     {min(t['开仓时间'] for t in trades).date()}" + Style.RESET_ALL)
            print(Fore.WHITE + f"结束时间:     {max(t['平仓时间'] for t in trades).date()}" + Style.RESET_ALL)
            if liquidated:
                print(Fore.RED + "已爆仓" + Style.RESET_ALL)
            else:
                print(Fore.GREEN + f"初始资金:     {INITIAL_CAPITAL:.2f} U" + Style.RESET_ALL)
                print(Fore.GREEN + f"结束资金:     {final_capital:.2f} U" + Style.RESET_ALL)
                print(Fore.GREEN + f"累计收益率:   {total_return * 100:.2f}%" + Style.RESET_ALL)
                print()
                print(Fore.GREEN + f"年化收益率:   {annualized_return * 100:.2f}%" + Style.RESET_ALL)
                print(Fore.GREEN + f"最大回撤:     {max_dd * 100:.2f}%" + Style.RESET_ALL)
                print(Fore.GREEN + f"最大回撤时长: {max_drawdown_duration} 天" + Style.RESET_ALL)

            print()
            print(Fore.MAGENTA + f"夏普比率:     {sharpe_ratio:.2f}" + Style.RESET_ALL)
            print(Fore.MAGENTA + f"总交易次数:   {len(trades)}" + Style.RESET_ALL)
            print(Fore.MAGENTA + f"盈利/亏损:    {wins}/{len(pnl_arr) - wins}" + Style.RESET_ALL)
            print(Fore.MAGENTA + f"胜率:         {win_rate * 100:.2f}%" + Style.RESET_ALL)
            print(Fore.MAGENTA + f"盈亏比:       {profit_loss_ratio:.2f}" + Style.RESET_ALL)
            print("=" * 50)

        else:
            print("\n" + "=" * 50)
            print(Fore.YELLOW + f"           回测结果统计 - {TARGET_STAR}" + Style.RESET_ALL)
            print("=" * 50)
            print(Fore.RED + "无有效交易记录" + Style.RESET_ALL)
            print("=" * 50)

        # 保存回测汇总到列表中
        summary = {
            "触发信号": TARGET_STAR,
            "初始资金": round(INITIAL_CAPITAL, 2),
            "结束资金": round(final_capital, 2),
            "累计收益率": f"{total_return * 100:.2f}%",
            "年化收益率": f"{annualized_return * 100:.2f}%",
            "夏普比率": round(sharpe_ratio, 2),
            "覆盖天数": int(days) if 'days' in locals() else 0,
            "覆盖年数": round(years, 2) if 'years' in locals() else 0.0,
            "交易次数": len(trades),
            "盈利次数": int(wins),
            "亏损次数": int(len(pnl_arr) - wins),
            "胜率": f"{win_rate * 100:.2f}%",
            "盈亏比": round(profit_loss_ratio, 2),
            "最大回撤": f"{max_dd * 100:.2f}%",
            "最大回撤时长": max_drawdown_duration,
            "是否爆仓": bool(liquidated)
        }
        all_summaries.append(summary)

    # 保存所有的汇总结果到一个CSV文件中
    summary_df = pd.DataFrame(all_summaries)
    summary_csv_path = OUT_DIR / f"回测_summary_批量.csv"
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    print(Fore.GREEN + f"已保存汇总结果: {summary_csv_path}" + Style.RESET_ALL)

    print(Fore.CYAN + "回测完成，结果目录：" + Style.RESET_ALL, OUT_DIR)

except KeyboardInterrupt:
    print(f"\n{Fore.YELLOW}程序被用户中断，正在退出...{Style.RESET_ALL}")
    sys.exit(0)
except Exception as e:
    print(Fore.RED + f"程序执行过程中发生错误: {e}" + Style.RESET_ALL)
    sys.exit(1)