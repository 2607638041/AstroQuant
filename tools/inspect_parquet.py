#!/usr/bin/env python3
"""
inspect_3parquet.py

交互式查看目录下的 parquet 文件：
- 列出目录下所有 parquet 文件（递归）
- 选择一个文件，随机查看 20 条记录
- 支持继续查看（再随机 20 条）、查看 head/tail、显示 schema、返回上一级、退出

用法:
  python inspect_parquet.py
或
  python inspect_parquet.py --data-dir data    # 指定数据目录

依赖:
  pip install pandas pyarrow
(若未安装 pyarrow，pandas 读取 parquet 会报错——脚本会提示并建议安装)
"""
import os
import sys
import argparse
import random
import signal
from textwrap import dedent


def signal_handler(sig, frame):
    """
    信号处理函数，用于捕获Ctrl+C中断信号

    Args:
        sig: 信号编号
        frame: 当前堆栈帧
    """
    print('\n\n程序已被用户中断，正在退出...')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

try:
    import pandas as pd

    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', None)  # 不限制显示宽度
    pd.set_option('display.max_colwidth', None)  # 不限制列宽
except Exception as e:
    print("需要 pandas，请先安装：pip install pandas pyarrow")
    raise

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")  # 默认数据目录


def find_parquet_files(root_dir):
    """
    递归查找指定目录下的所有parquet文件

    Args:
        root_dir (str): 要搜索的根目录路径

    Returns:
        list: 包含所有找到的parquet文件路径的排序列表
    """
    files = []
    for dirpath, dirs, filenames in os.walk(root_dir):  # 递归遍历目录
        for fn in filenames:
            if fn.lower().endswith(".parquet"):  # 筛选parquet文件
                files.append(os.path.join(dirpath, fn))
    files.sort()  # 排序文件列表
    return files


def safe_read_parquet(path, max_rows_warn=500000):
    """
    尝试安全地读取 parquet 文件为 pandas.DataFrame。
    如果文件很大（行数未知或超过 max_rows_warn），返回 df=None 并提示。

    Args:
        path (str): parquet文件路径
        max_rows_warn (int): 行数警告阈值，默认为500000

    Returns:
        tuple: (DataFrame或None, 行数或错误信息)
    """
    try:
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            num_rows = pf.metadata.num_rows  # 获取文件行数元数据
        except Exception:
            num_rows = None

        if num_rows is not None and num_rows > max_rows_warn:  # 检查文件是否过大
            return None, num_rows
        df = pd.read_parquet(path)  # 读取文件
        return df, len(df)
    except Exception as e:
        return None, str(e)


def sample_rows_from_df(df, n=20, seen_idx=None):
    """
    从DataFrame中随机采样n行数据，避免重复采样已查看过的行

    Args:
        df (pandas.DataFrame): 源数据DataFrame
        n (int): 采样行数，默认20
        seen_idx (set): 已经采样过的行索引集合

    Returns:
        tuple: (采样的DataFrame, 更新后的已查看索引集合)
    """
    if seen_idx is None:
        seen_idx = set()
    if df is None or df.empty:
        return pd.DataFrame(), seen_idx
    if len(df) <= n:  # 如果数据行数小于等于采样数，返回全部数据
        return df.sample(frac=1).reset_index(drop=False), set(range(len(df)))
    available_idx = set(range(len(df))) - seen_idx  # 计算未查看的行索引
    if len(available_idx) < n:  # 如果未查看的行数不足，重置查看记录
        seen_idx = set()
        available_idx = set(range(len(df)))
    pick = random.sample(list(available_idx), n)  # 随机采样
    seen_idx.update(pick)  # 更新已查看索引集合
    return df.iloc[pick].reset_index(drop=False), seen_idx


def show_dataframe_full(df):
    """
    完整显示 DataFrame，确保所有列都显示出来

    Args:
        df (pandas.DataFrame): 要显示的DataFrame
    """
    if df is None:
        print("DataFrame 为空")
        return

    # 保存当前显示设置
    original_settings = {
        'display.max_columns': pd.get_option('display.max_columns'),
        'display.width': pd.get_option('display.width'),
        'display.max_colwidth': pd.get_option('display.max_colwidth')
    }

    # 设置无限制显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print(df)

    # 恢复原始显示设置
    pd.set_option('display.max_columns', original_settings['display.max_columns'])
    pd.set_option('display.width', original_settings['display.width'])
    pd.set_option('display.max_colwidth', original_settings['display.max_colwidth'])


def show_menu_file_list(files):
    """
    显示文件列表菜单

    Args:
        files (list): 文件路径列表
    """
    print("\n找到以下 parquet 文件：")
    for i, f in enumerate(files, 1):  # 从1开始编号显示
        basename = os.path.basename(f)
        print(f"  [{i}] {basename}")
    print("\n选项:")
    print("  r  刷新列表")
    print("  b  返回上一级")
    print("  q  退出程序")
    print("请输入文件编号或文件名：")


def show_file_menu(path):
    """
    显示文件操作菜单

    Args:
        path (str): 当前查看的文件路径
    """
    basename = os.path.basename(path)
    print(dedent(f"""
    ------------------------------------------------------------
    当前选择: {basename}
    文件大小: {os.path.getsize(path) / 1024 / 1024:.2f} MB
    操作选项:
      1) 随机查看 20 条记录
      2) 查看头部 20 条
      3) 查看尾部 20 条
      4) 查看下 20 条记录
      5) 查看上 20 条记录
      6) 显示列信息
      b) 返回上一级
      q) 退出程序
    请输入选项:
    """))


def interactive_inspect(data_dir):
    """
    交互式检查指定目录中的parquet文件

    Args:
        data_dir (str): 要检查的目录路径

    Returns:
        str: "quit"表示退出程序，"back"表示返回上一级
    """
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):  # 检查目录是否存在
        print(f"数据目录不存在: {data_dir}")
        return

    files = find_parquet_files(data_dir)
    if not files:  # 检查是否有parquet文件
        print(f"未在 {data_dir} 下找到任何 .parquet 文件。")
        return

    while True:
        show_menu_file_list(files)
        choice = input(">>> ").strip()
        if choice.lower() == "q":
            print("退出程序。")
            return "quit"
        if choice.lower() == "r":  # 刷新文件列表
            files = find_parquet_files(data_dir)
            continue
        if choice.lower() == "b":
            return "back"
        if not choice:  # 检查输入是否为空
            print("输入不能为空，请重试。")
            continue

        selected = None
        if choice.isdigit():  # 按编号选择文件
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                selected = files[idx]
            else:
                print("编号超出范围，请重试。")
                continue
        else:  # 按文件名选择文件
            matches = [f for f in files if os.path.basename(f) == choice]
            if not matches:
                print("未找到匹配的文件名，请重试。")
                continue
            elif len(matches) > 1:  # 处理同名文件
                print("找到多个同名文件，请使用编号选择：")
                for i, f in enumerate(matches, 1):
                    print(f"  [{i}] {os.path.relpath(f)}")
                sub_choice = input("请选择编号: ").strip()
                if sub_choice.isdigit() and 1 <= int(sub_choice) <= len(matches):
                    selected = matches[int(sub_choice) - 1]
                else:
                    print("无效选择。")
                    continue
            else:
                selected = matches[0]

        if not selected:
            print("未选择文件，请重试。")
            continue

        # 初始化文件检查状态
        df = None
        df_len = None
        loaded_full = False
        seen_idx = set()
        current_position = 0

        while True:
            show_file_menu(selected)
            try:
                sub = input(">>> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n输入中断，返回文件列表。")
                break

            if sub == "b":
                break
            if sub == "q":
                print("退出程序。")
                return "quit"

            if sub in ("1", "2", "3", "4", "5", "6"):  # 处理文件查看操作
                if df is None:  # 按需加载数据
                    df, info = safe_read_parquet(selected)
                    if df is None:
                        if isinstance(info, int):  # 处理大文件
                            print(f"[warn] 文件行数估计为 {info} 行，直接读取可能耗费大量内存/时间。")
                            print("你可以选择：")
                            print("  h  读取头部 10 条（推荐）")
                            print("  t  读取尾部 10 条（推荐）")
                            print("  f  强制读取全部（按 y 确认）")
                            print("  b  返回上一级")
                            try:
                                ans = input("请选择 (h/t/f/b): ").strip().lower()
                            except (EOFError, KeyboardInterrupt):
                                print("\n输入中断，返回文件菜单。")
                                continue

                            if ans == "h":  # 读取头部数据
                                try:
                                    df_head = pd.read_parquet(selected, engine="pyarrow", columns=None).head(10)
                                    show_dataframe_full(df_head)
                                except Exception as e:
                                    print(f"[error] 读取头部失败: {e}")
                                continue
                            elif ans == "t":  # 读取尾部数据
                                try:
                                    df_tail = pd.read_parquet(selected, engine="pyarrow", columns=None).tail(10)
                                    show_dataframe_full(df_tail)
                                except Exception as e:
                                    print(f"[error] 读取尾部失败: {e}")
                                continue
                            elif ans == "f":  # 强制读取全部数据
                                try:
                                    confirm = input("确定要强制读取整个文件到内存吗？(y/N): ").strip().lower()
                                except (EOFError, KeyboardInterrupt):
                                    print("\n输入中断，返回文件菜单。")
                                    continue

                                if confirm != "y":
                                    continue
                                try:
                                    print("[info] 正在读取整个 parquet 文件，请稍候...")
                                    df = pd.read_parquet(selected)
                                    df_len = len(df)
                                    loaded_full = True
                                except Exception as e:
                                    print(f"[error] 读取失败: {e}")
                                    df = None
                                    df_len = None
                                    continue
                            elif ans == "b":
                                continue
                            else:
                                continue
                        else:
                            print(f"[error] 读取 parquet 文件失败: {info}")
                            try:
                                sub2 = input("按 b 返回上一级，按 q 退出: ").strip().lower()
                            except (EOFError, KeyboardInterrupt):
                                print("\n输入中断，返回文件菜单。")
                                continue

                            if sub2 == "b":
                                break
                            if sub2 == "q":
                                return "quit"
                            continue
                    else:
                        df_len = len(df)
                        loaded_full = True

                # 根据用户选择执行不同操作
                if sub == "1":  # 随机查看
                    if df is None:
                        print("[error] Dataframe 未加载，无法随机采样。")
                        continue
                    sample, seen_idx = sample_rows_from_df(df, n=20, seen_idx=seen_idx)
                    show_dataframe_full(sample)
                    print(f"已查看样本数量: {len(seen_idx)} / {df_len}")
                elif sub == "2":  # 查看头部
                    try:
                        if loaded_full:
                            show_dataframe_full(df.head(20))
                        else:
                            show_dataframe_full(pd.read_parquet(selected).head(20))
                        # 修复：查看头部后应将当前位置重置为0
                        current_position = 0
                        print("[info] 当前位置: 0 (头部)")
                    except Exception as e:
                        print(f"[error] 读取头部失败: {e}")
                elif sub == "3":  # 查看尾部
                    try:
                        if loaded_full:
                            show_dataframe_full(df.tail(20))
                            current_position = len(df)
                        else:
                            # 在未完全加载时，我们不知道确切的总行数，但可以安全地将指针设为一个大于或等于总行数的值。
                            # 这里先尝试通过 pyarrow 获取元数据中的行数。
                            try:
                                import pyarrow.parquet as pq
                                pf = pq.ParquetFile(selected)
                                total_rows = pf.metadata.num_rows
                            except Exception:
                                # 如果无法获取元数据，则保守地将位置设为一个很大的数，确保"下20条"会提示已到末尾。
                                total_rows = float('inf')

                            df_tail = pd.read_parquet(selected).tail(20)
                            show_dataframe_full(df_tail)
                            current_position = total_rows
                            print(f"[info] 当前位置: {current_position}{' (尾部)' if total_rows != float('inf') else ''}")
                        print(f"[info] 当前位置: {current_position} (尾部)")
                    except Exception as e:
                        print(f"[error] 读取尾部失败: {e}")
                elif sub == "4":  # 查看下20条
                    try:
                        if df is None:
                            print("[error] Dataframe 未加载，无法查看下二十条记录。")
                            continue

                        # 特殊处理：如果在头部位置（位置0），则显示前20条记录后将位置更新为20
                        if current_position == 0:
                            next_slice = df.iloc[current_position:current_position + 20]
                            show_dataframe_full(next_slice)
                            current_position = min(current_position + 20, len(df))
                        elif current_position >= len(df):  # 检查是否到达文件末尾
                            print("[info] 已经到达文件末尾，无法查看下二十条记录。")
                            continue
                        else:
                            next_slice = df.iloc[current_position:current_position + 20]
                            show_dataframe_full(next_slice)
                            current_position = min(current_position + 20, len(df))
                    
                        position_info = f"{current_position}/{len(df)}"
                        if current_position == 0:
                            position_info += " (头部)"
                        elif current_position >= len(df):
                            position_info += " (尾部)"
                        print(f"[info] 当前位置: {position_info}")
                    except Exception as e:
                        print(f"[error] 查看下二十条记录失败: {e}")
                elif sub == "5":  # 查看上20条
                    try:
                        if df is None:
                            print("[error] Dataframe 未加载，无法查看上二十条记录。")
                            continue

                        if current_position >= len(df):  # 如果在文件末尾，调整位置
                            current_position = len(df)

                        if current_position <= 0:  # 检查是否到达文件开头
                            print("[info] 已经到达文件开头，无法查看上二十条记录。")
                            continue

                        prev_position = max(current_position - 20, 0)
                        prev_slice = df.iloc[prev_position:current_position]
                        show_dataframe_full(prev_slice)
                        current_position = prev_position
                        print(f"[info] 当前位置: {current_position}/{len(df)}")
                    except Exception as e:
                        print(f"[error] 查看上二十条记录失败: {e}")
                elif sub == "6":  # 显示列信息
                    try:
                        if loaded_full:
                            print("Columns:", list(df.columns))
                            print("Index dtype:", df.index.dtype)
                            print("Rows:", df_len)
                        else:
                            try:
                                import pyarrow.parquet as pq
                                pf = pq.ParquetFile(selected)
                                schema = pf.schema
                                print("pyarrow schema:")
                                print(schema)
                                try:
                                    num_rows = pf.metadata.num_rows
                                    print("Num rows (pyarrow metadata):", num_rows)
                                except Exception:
                                    pass
                            except Exception:
                                df_tmp = pd.read_parquet(selected, engine="pyarrow", columns=None)
                                print("Columns:", list(df_tmp.columns))
                                print("Rows:", len(df_tmp))
                    except Exception as e:
                        print(f"[error] 获取 schema 失败: {e}")
            else:
                print("无效输入，请输入菜单中的选项。")


def interactive_inspect_file(filepath):
    """
    交互式检查单个parquet文件

    Args:
        filepath (str): 要检查的文件路径
    """
    # 初始化文件检查状态
    df = None
    df_len = None
    loaded_full = False
    seen_idx = set()
    current_position = 0

    while True:
        show_file_menu(filepath)
        try:
            sub = input(">>> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n输入中断，返回文件列表。")
            break

        if sub == "b":
            break
        if sub == "q":
            print("退出程序。")
            sys.exit(0)

        if sub in ("1", "2", "3", "4", "5", "6"):
            if df is None:  # 按需加载数据
                df, info = safe_read_parquet(filepath)
                if df is None:
                    if isinstance(info, int):  # 处理大文件
                        print(f"[warn] 文件行数估计为 {info} 行，直接读取可能耗费大量内存/时间。")
                        print("你可以选择：")
                        print("  h  读取头部 10 条（推荐）")
                        print("  t  读取尾部 10 条（推荐）")
                        print("  f  强制读取全部（按 y 确认）")
                        print("  b  返回上一级")
                        try:
                            ans = input("请选择 (h/t/f/b): ").strip().lower()
                        except (EOFError, KeyboardInterrupt):
                            print("\n输入中断，返回文件菜单。")
                            continue

                        if ans == "h":  # 读取头部数据
                            try:
                                df_head = pd.read_parquet(filepath, engine="pyarrow", columns=None).head(10)
                                show_dataframe_full(df_head)
                            except Exception as e:
                                print(f"[error] 读取头部失败: {e}")
                            continue
                        elif ans == "t":  # 读取尾部数据
                            try:
                                df_tail = pd.read_parquet(filepath, engine="pyarrow", columns=None).tail(10)
                                show_dataframe_full(df_tail)
                            except Exception as e:
                                print(f"[error] 读取尾部失败: {e}")
                            continue
                        elif ans == "f":  # 强制读取全部数据
                            try:
                                confirm = input("确定要强制读取整个文件到内存吗？(y/N): ").strip().lower()
                            except (EOFError, KeyboardInterrupt):
                                print("\n输入中断，返回文件菜单。")
                                continue

                            if confirm != "y":
                                continue
                            try:
                                print("[info] 正在读取整个 parquet 文件，请稍候...")
                                df = pd.read_parquet(filepath)
                                df_len = len(df)
                                loaded_full = True
                            except Exception as e:
                                print(f"[error] 读取失败: {e}")
                                df = None
                                df_len = None
                                continue
                        elif ans == "b":
                            continue
                        else:
                            continue
                    else:
                        print(f"[error] 读取 parquet 文件失败: {info}")
                        try:
                            sub2 = input("按 b 返回上一级，按 q 退出: ").strip().lower()
                        except (EOFError, KeyboardInterrupt):
                            print("\n输入中断，返回文件菜单。")
                            continue

                        if sub2 == "b":
                            break
                        if sub2 == "q":
                            sys.exit(0)
                        continue
                else:
                    df_len = len(df)
                    loaded_full = True

            # 根据用户选择执行不同操作
            if sub == "1":  # 随机查看
                if df is None:
                    print("[error] Dataframe 未加载，无法随机采样。")
                    continue
                sample, seen_idx = sample_rows_from_df(df, n=20, seen_idx=seen_idx)
                show_dataframe_full(sample)
                print(f"已查看样本数量: {len(seen_idx)} / {df_len}")
            elif sub == "2":  # 查看头部
                try:
                    if loaded_full:
                        show_dataframe_full(df.head(20))
                    else:
                        show_dataframe_full(pd.read_parquet(filepath).head(20))
                    # 修复：查看头部后应将当前位置重置为0
                    current_position = 0
                    print("[info] 当前位置: 0 (头部)")
                except Exception as e:
                    print(f"[error] 读取头部失败: {e}")
            elif sub == "3":  # 查看尾部
                try:
                    if loaded_full:
                        show_dataframe_full(df.tail(20))
                        current_position = len(df)
                    else:
                        # 在未完全加载时，我们不知道确切的总行数，但可以安全地将指针设为一个大于或等于总行数的值。
                        # 这里先尝试通过 pyarrow 获取元数据中的行数。
                        try:
                            import pyarrow.parquet as pq
                            pf = pq.ParquetFile(filepath)
                            total_rows = pf.metadata.num_rows
                        except Exception:
                            # 如果无法获取元数据，则保守地将位置设为一个很大的数，确保“下20条”会提示已到末尾。
                            total_rows = float('inf')

                        df_tail = pd.read_parquet(filepath).tail(20)
                        show_dataframe_full(df_tail)
                        current_position = total_rows
                        print(f"[info] 当前位置: {current_position}{' (尾部)' if total_rows != float('inf') else ''}")
                    print(f"[info] 当前位置: {current_position} (尾部)")
                except Exception as e:
                    print(f"[error] 读取尾部失败: {e}")
            elif sub == "4":  # 查看下20条
                try:
                    if df is None:
                        print("[error] Dataframe 未加载，无法查看下二十条记录。")
                        continue

                    if current_position >= len(df):  # 检查是否到达文件末尾
                        print("[info] 已经到达文件末尾，无法查看下二十条记录。")
                        continue

                    next_slice = df.iloc[current_position:current_position + 20]
                    show_dataframe_full(next_slice)
                    current_position = min(current_position + 20, len(df))
                    print(f"[info] 当前位置: {current_position}/{len(df)}")
                except Exception as e:
                    print(f"[error] 查看下二十条记录失败: {e}")
            elif sub == "5":  # 查看上20条
                try:
                    if df is None:
                        print("[error] Dataframe 未加载，无法查看上二十条记录。")
                        continue

                    if current_position >= len(df):  # 如果在文件末尾，调整位置
                        current_position = len(df)

                    if current_position <= 0:  # 检查是否到达文件开头
                        print("[info] 已经到达文件开头，无法查看上二十条记录。")
                        continue

                    prev_position = max(current_position - 20, 0)
                    prev_slice = df.iloc[prev_position:current_position]
                    show_dataframe_full(prev_slice)
                    current_position = prev_position
                    print(f"[info] 当前位置: {current_position}/{len(df)}")
                except Exception as e:
                    print(f"[error] 查看上二十条记录失败: {e}")
            elif sub == "6":  # 显示列信息
                try:
                    if loaded_full:
                        print("Columns:", list(df.columns))
                        print("Index dtype:", df.index.dtype)
                        print("Rows:", df_len)
                    else:
                        try:
                            import pyarrow.parquet as pq
                            pf = pq.ParquetFile(filepath)
                            schema = pf.schema
                            print("pyarrow schema:")
                            print(schema)
                            try:
                                num_rows = pf.metadata.num_rows
                                print("Num rows (pyarrow metadata):", num_rows)
                            except Exception:
                                pass
                        except Exception:
                            df_tmp = pd.read_parquet(filepath, engine="pyarrow", columns=None)
                            print("Columns:", list(df_tmp.columns))
                            print("Rows:", len(df_tmp))
                except Exception as e:
                    print(f"[error] 获取 schema 失败: {e}")
        else:
            print("无效输入，请输入菜单中的选项。")


class DirectoryBrowser:
    """
    目录浏览器类，用于层级式浏览包含parquet文件的目录结构
    """

    def __init__(self, data_root_dir):
        """
        初始化目录浏览器

        Args:
            data_root_dir (str): 数据根目录路径
        """
        self.data_root_dir = os.path.abspath(data_root_dir)
        self.current_dir = self.data_root_dir
        self.dir_history = []  # 目录浏览历史栈

    def list_directories_and_files(self):
        """
        列出当前目录下的所有子目录和parquet文件

        Returns:
            tuple: (子目录列表, parquet文件列表)
        """
        dirs = []
        files = []

        try:
            items = os.listdir(self.current_dir)  # 获取当前目录内容
            for item in items:
                item_path = os.path.join(self.current_dir, item)
                if os.path.isdir(item_path):  # 处理子目录
                    has_parquet = False
                    for root, _, filenames in os.walk(item_path):  # 检查子目录是否包含parquet文件
                        if any(fn.lower().endswith('.parquet') for fn in filenames):
                            has_parquet = True
                            break
                    if has_parquet:
                        dirs.append(item)
                elif os.path.isfile(item_path) and item.lower().endswith('.parquet'):  # 处理parquet文件
                    files.append(item)
        except Exception as e:
            print(f"读取目录内容失败: {e}")

        return sorted(dirs), sorted(files)  # 返回排序后的结果

    def browse(self):
        """
        执行目录浏览，返回选择的文件路径或None（表示退出）

        Returns:
            str or None: 选择的文件路径，或None表示退出目录选择
        """
        while True:
            # 显示当前目录路径
            print(
                f"\n当前目录: {os.path.relpath(self.current_dir, self.data_root_dir) if self.current_dir != self.data_root_dir else '[根目录]'}")

            subdirs, parquet_files = self.list_directories_and_files()

            options = []
            index = 1

            # 添加子目录选项
            for dirname in subdirs:
                dir_path = os.path.join(self.current_dir, dirname)
                file_count = 0
                try:
                    for root, _, filenames in os.walk(dir_path):
                        file_count += len([f for f in filenames if f.lower().endswith('.parquet')])
                except Exception:
                    pass

                options.append((index, 'dir', dirname, dir_path, file_count))
                print(f"  [{index}] {dirname}/ ({file_count} 个文件)")
                index += 1

            # 添加文件选项
            for filename in parquet_files:
                filepath = os.path.join(self.current_dir, filename)
                try:
                    size = os.path.getsize(filepath)
                    size_mb = size / (1024 * 1024)
                    options.append((index, 'file', filename, filepath, size_mb))
                    print(f"  [{index}] {filename} ({size_mb:.2f} MB)")
                    index += 1
                except Exception:
                    pass

            print("  [b] 返回上一级")
            print("  [q] 退出程序")
            print("  [*] 手动输入路径")

            choice = input("请输入选项编号: ").strip()

            if choice.lower() == "b":
                if self.dir_history:
                    self.current_dir = self.dir_history.pop()
                    continue
                else:
                    return None

            if choice.lower() == "q":
                return None

            if choice.lower() == "*":
                manual_path = input("请输入文件或目录的完整路径: ").strip()
                if not manual_path:
                    print("路径不能为空。")
                    continue

                if not os.path.exists(manual_path):
                    print("路径不存在。")
                    continue

                if os.path.isfile(manual_path):
                    if manual_path.lower().endswith('.parquet'):
                        print(f"选定文件: {manual_path}")
                        return manual_path
                    else:
                        print("不是有效的 Parquet 文件 (.parquet)。")
                        continue
                elif os.path.isdir(manual_path):
                    self.dir_history.append(self.current_dir)
                    self.current_dir = manual_path
                    continue
                else:
                    print("既不是文件也不是目录。")
                    continue

            if choice.isdigit():
                choice_idx = int(choice)
                selected = None
                for opt in options:
                    if opt[0] == choice_idx:
                        selected = opt
                        break

                if selected:
                    if selected[1] == 'dir':
                        self.dir_history.append(self.current_dir)
                        self.current_dir = selected[3]
                        continue
                    elif selected[1] == 'file':
                        print(f"选定文件: {selected[3]}")
                        return selected[3]

                print("无效选择，请重试。")
            else:
                print("无效输入，请输入数字编号、'b'、'q' 或 'm'。")


def main():
    """
    主函数，程序入口点
    """
    parser = argparse.ArgumentParser(description="交互式 Parquet 文件查看工具")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="数据根目录，用于查找子目录")
    args = parser.parse_args()
    print("Parquet 文件交互式查看工具")
    print("数据根目录:", args.data_dir)

    try:
        browser = DirectoryBrowser(args.data_dir) if os.path.isdir(args.data_dir) else None
        if not browser:
            print(f"数据根目录不存在: {args.data_dir}")
            return

        while True:
            selected_path = browser.browse()
            if not selected_path:
                print("程序退出。")
                break

            if os.path.isfile(selected_path) and selected_path.lower().endswith('.parquet'):
                print(f"选定文件: {selected_path}")
                interactive_inspect_file(selected_path)
                continue
            else:
                print("无效路径。")
                break

    except Exception as e:
        print(f"程序运行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    """
    程序入口点
    """
    try:
        main()
    except KeyboardInterrupt:
        print('\n\n程序已被用户中断，正在退出...')
        sys.exit(0)
    except Exception as e:
        print(f"程序发生未预期的错误: {e}")
        sys.exit(1)