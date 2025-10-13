import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob

# 星宿配置（二十八宿）
STAR_LODGES = {
    '毕宿': {'start': '03-21', 'end': '04-19'},
    '氐宿': {'start': '04-20', 'end': '05-20'},
    '参宿': {'start': '05-21', 'end': '06-20'},
    '尾宿': {'start': '06-21', 'end': '07-22'},
    '轸宿': {'start': '07-23', 'end': '08-22'}
}

# 建星配置（十二建星）
JIAN_XING = {
    '除': {'cycle': 12, 'position': 3}  # 除日在建星循环中的位置
}

# 目标筛选条件
TARGET_STAR_LODGES = ['毕宿', '氐宿', '参宿', '尾宿', '轸宿']  # 五个星宿
TARGET_JIAN_XING = ['除']  # 建星中的除日


def get_star_lodge(date):
    """根据日期获取星宿"""
    month_day = date.strftime('%m-%d')

    for lodge, dates in STAR_LODGES.items():
        current_year = date.year

        # 解析开始日期
        start_month, start_day = map(int, dates['start'].split('-'))
        start_date = datetime(current_year, start_month, start_day).date()

        # 解析结束日期
        end_month, end_day = map(int, dates['end'].split('-'))
        end_date = datetime(current_year, end_month, end_day).date()

        # 处理跨年情况
        if start_date > end_date:
            if date.date() >= start_date or date.date() <= end_date:
                return lodge
        else:
            if start_date <= date.date() <= end_date:
                return lodge

    return None


def get_jian_xing(date, base_date=datetime(2000, 1, 1)):
    """根据日期获取建星"""
    days_diff = (date - base_date).days

    for jx, config in JIAN_XING.items():
        cycle = config['cycle']
        position = config['position']

        # 计算当前日期在建星循环中的位置
        current_position = (days_diff % cycle)

        # 如果位置匹配，返回对应的建星
        if current_position == position:
            return jx

    return None


def load_all_data(data_folder):
    """加载文件夹中的所有parquet文件数据"""
    parquet_files = glob.glob(os.path.join(data_folder, "*.parquet"))

    if not parquet_files:
        print(f"在文件夹 {data_folder} 中未找到parquet文件")
        return None

    print(f"找到 {len(parquet_files)} 个parquet文件:")
    for file in parquet_files:
        print(f"  - {os.path.basename(file)}")

    all_dataframes = []
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            print(f"成功读取: {os.path.basename(file)}, 数据量: {len(df)} 行")
            all_dataframes.append(df)
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")

    if not all_dataframes:
        print("未能成功读取任何文件")
        return None

    combined_df = pd.concat(all_dataframes, ignore_index=True)

    if 'datetime' in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=['datetime'])
        combined_df = combined_df.sort_values('datetime')
    elif 'timestamp' in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=['timestamp'])
        combined_df = combined_df.sort_values('timestamp')

    print(f"合并后总数据量: {len(combined_df)} 行")
    return combined_df


def filter_daily_conditions(df):
    """按整天筛选：星宿和建星"""
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    elif 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    else:
        print("数据框中未找到datetime或timestamp列")
        return pd.DataFrame()

    # 提取日期（去掉时间部分）
    df['date'] = df['datetime'].dt.date

    # 按日期分组，获取每天的星宿和建星
    daily_data = df.groupby('date').agg({
        'datetime': 'first'  # 取每天的第一个时间点来计算星宿和建星
    }).reset_index()

    # 添加星宿列（基于每天的日期）
    daily_data['star_lodge'] = daily_data['date'].apply(
        lambda x: get_star_lodge(datetime.combine(x, datetime.min.time()))
    )

    # 添加建星列（基于每天的日期）
    daily_data['jian_xing'] = daily_data['date'].apply(
        lambda x: get_jian_xing(datetime.combine(x, datetime.min.time()))
    )

    # 多条件筛选：星宿在目标列表中且建星为除日
    filtered_dates = daily_data[
        (daily_data['star_lodge'].isin(TARGET_STAR_LODGES)) &
        (daily_data['jian_xing'] == '除')
        ]

    return filtered_dates


def main():
    """主函数"""
    data_folder = "E:/project/AstroQuant/data/merged/btc/btc_5m"

    if not os.path.exists(data_folder):
        print(f"数据文件夹不存在: {data_folder}")
        return

    print("开始按整天多条件筛选...")
    print(f"目标星宿: {', '.join(TARGET_STAR_LODGES)}")
    print(f"目标建星: {', '.join(TARGET_JIAN_XING)}")
    print("=" * 60)

    # 加载所有数据
    df = load_all_data(data_folder)
    if df is None:
        return

    # 按整天筛选
    filtered_dates = filter_daily_conditions(df)

    if not filtered_dates.empty:
        print(f"\n按整天筛选结果统计:")
        print("=" * 60)
        print(f"符合条件的日期数: {len(filtered_dates)}")

        # 按年份统计
        filtered_dates['year'] = filtered_dates['date'].apply(lambda x: x.year)
        year_stats = filtered_dates['year'].value_counts().sort_index()

        print(f"\n按年份统计:")
        print("=" * 40)
        for year, count in year_stats.items():
            print(f"{year}年: {count} 天")

        # 按星宿统计
        lodge_stats = filtered_dates['star_lodge'].value_counts()
        print(f"\n按星宿统计:")
        print("=" * 40)
        for lodge, count in lodge_stats.items():
            print(f"{lodge}: {count} 天")

        # 打印详细的日期列表
        print(f"\n符合条件的详细日期:")
        print("=" * 40)
        for _, row in filtered_dates.iterrows():
            print(f"日期: {row['date'].strftime('%Y-%m-%d')}")
            print(f"星宿: {row['star_lodge']}")
            print(f"建星: {row['jian_xing']}")
            print("-" * 30)

        # 保存筛选结果
        result_file = "E:/project/AstroQuant/data/daily_filtered_results.csv"
        filtered_dates.to_csv(result_file, index=False)
        print(f"\n筛选结果已保存到: {result_file}")

        # 显示数据基本信息
        if 'datetime' in df.columns:
            min_date = df['datetime'].min().date()
            max_date = df['datetime'].max().date()
            total_days = (max_date - min_date).days
            print(f"\n数据时间范围: {min_date} 到 {max_date} (共{total_days}天)")

            # 计算筛选比例
            total_unique_days = df['datetime'].dt.date.nunique()
            filtered_ratio = len(filtered_dates) / total_unique_days * 100
            print(f"总天数: {total_unique_days} 天")
            print(f"筛选比例: {filtered_ratio:.2f}%")

    else:
        print("未找到符合条件的日期")
        print("可能的原因:")
        print("1. 星宿日期范围配置有误")
        print("2. 建星循环周期配置有误")
        print("3. 数据时间范围内没有符合条件的日期")


if __name__ == "__main__":
    main()