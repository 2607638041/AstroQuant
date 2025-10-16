#!/usr/bin/env python3
"""
 一个简单的 parquet 文件浏览器
"""
import os
import sys
import argparse
import random
import signal

try:
    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
except:
    print("需要 pandas，请先安装：pip install pandas pyarrow")
    sys.exit(1)

signal.signal(signal.SIGINT, lambda s, f: (print('\n\n程序已被用户中断'), sys.exit(0)))

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MAX_ROWS = 5000000

# 获取所有 parquet 文件
def find_parquets(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        files.extend(os.path.join(dirpath, f) for f in filenames if f.lower().endswith(".parquet"))
    return sorted(files)

# 获取行数
def get_row_count(path):
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    except:
        return None

# 显示 DataFrame
def show_df(df):
    print(df if df is not None and not df.empty else "DataFrame 为空")

# 文件读取器
class FileReader:
    # 初始化
    def __init__(self, path):
        self.path = path
        self.df = None
        self.size = 0
        self.seen = set()
        self.pos = 0
    # 读取文件
    def load(self):
        if self.df is not None:
            return True

        try:
            rows = get_row_count(self.path)
            if rows and rows > MAX_ROWS:
                return self._handle_large(rows)

            self.df = pd.read_parquet(self.path)
            self.size = len(self.df)
            return True
        except Exception as e:
            print(f"[error] {e}")
            return False
    # 处理大型文件
    def _handle_large(self, rows):
        print(f"[warn] 文件约 {rows} 行，可能占用大量内存")
        print("  h-头部10条  t-尾部10条  f-强制全读  b-返回")
        choice = input(">>> ").strip().lower()

        if choice == "h":
            show_df(pd.read_parquet(self.path).head(10))
        elif choice == "t":
            show_df(pd.read_parquet(self.path).tail(10))
        elif choice == "f" and input("确认? (y/N): ").lower() == "y":
            print("[info] 读取中...")
            self.df = pd.read_parquet(self.path)
            self.size = len(self.df)
            return True
        return False
    # 随机查看
    def sample(self, n=20):
        if not self.load() or self.df is None:
            return

        if self.size <= n:
            show_df(self.df.sample(frac=1))
            self.seen = set(range(self.size))
        else:
            avail = set(range(self.size)) - self.seen
            if len(avail) < n:
                self.seen.clear()
                avail = set(range(self.size))
            pick = random.sample(list(avail), n)
            self.seen.update(pick)
            show_df(self.df.iloc[pick].reset_index(drop=False))
        print(f"已查看: {len(self.seen)}/{self.size}")
    # 显示头部
    def head(self):
        if self.df is not None:
            show_df(self.df.head(20))
            self.pos = 0
        else:
            show_df(pd.read_parquet(self.path).head(20))
        print("[info] 位置: 0 (头部)")
    # 显示尾部
    def tail(self):
        if self.df is not None:
            show_df(self.df.tail(20))
            self.pos = self.size
            print(f"[info] 位置: {self.pos} (尾部)")
        else:
            show_df(pd.read_parquet(self.path).tail(20))
            rows = get_row_count(self.path)
            print(f"[info] 位置: {rows if rows else '?'} (尾部)")
    # 显示下一页
    def next_page(self):
        if not self.load():
            return
        if self.pos >= self.size:
            print("[info] 已到末尾")
            return
        show_df(self.df.iloc[self.pos:self.pos+20])
        self.pos = min(self.pos + 20, self.size)
        print(f"[info] 位置: {self.pos}/{self.size}")
    # 显示上一页
    def prev_page(self):
        if not self.load():
            return
        if self.pos <= 0:
            print("[info] 已到开头")
            return
        new_pos = max(0, self.pos - 20)
        show_df(self.df.iloc[new_pos:self.pos])
        self.pos = new_pos
        print(f"[info] 位置: {self.pos}/{self.size}")
    # 显示模式
    def schema(self):
        if self.df is not None:
            print(f"Columns: {list(self.df.columns)}\nRows: {self.size}")
        else:
            try:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(self.path)
                print(f"{pf.schema}\nRows: {pf.metadata.num_rows}")
            except:
                df = pd.read_parquet(self.path)
                print(f"Columns: {list(df.columns)}\nRows: {len(df)}")


class Browser:
    # 初始化
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.curr = self.root
        self.hist = []
    # 判断目录下是否有 parquet 文件
    def _has_parquet(self, path):
        for _, _, files in os.walk(path):
            if any(f.lower().endswith('.parquet') for f in files):
                return True
        return False
    # 统计目录下 parquet 文件数
    def _count_parquet(self, path):
        count = 0
        for _, _, files in os.walk(path):
            count += sum(1 for f in files if f.lower().endswith('.parquet'))
        return count
    # 显示目录内容
    def show(self):
        rel = os.path.relpath(self.curr, self.root) if self.curr != self.root else "[根]"
        print(f"\n当前: {rel}")

        opts = []
        idx = 1

        for item in sorted(os.listdir(self.curr)):
            full = os.path.join(self.curr, item)
            if os.path.isdir(full) and self._has_parquet(full):
                cnt = self._count_parquet(full)
                opts.append((idx, 'd', item, full))
                print(f"  [{idx}] {item}/ ({cnt})")
                idx += 1
            elif item.lower().endswith('.parquet'):
                mb = os.path.getsize(full) / 1024 / 1024
                opts.append((idx, 'f', item, full))
                print(f"  [{idx}] {item} ({mb:.2f}MB)")
                idx += 1

        print("  [b]返回 [q]退出 [*]手动路径")
        return opts
    # 导航
    def nav(self):
        while True:
            opts = self.show()
            choice = input(">>> ").strip()

            if choice == 'b':
                if self.hist:
                    self.curr = self.hist.pop()
                else:
                    return None
            elif choice == 'q':
                return None
            elif choice == '*':
                path = input("路径: ").strip()
                if os.path.isfile(path) and path.lower().endswith('.parquet'):
                    return path
                elif os.path.isdir(path):
                    self.hist.append(self.curr)
                    self.curr = path
            elif choice.isdigit():
                for i, t, n, p in opts:
                    if i == int(choice):
                        if t == 'd':
                            self.hist.append(self.curr)
                            self.curr = p
                            break
                        else:
                            return p

# 文件查看器
def inspect_file(path):
    reader = FileReader(path)
    name = os.path.basename(path)
    size = os.path.getsize(path) / 1024 / 1024

    ops = {
        '1': ('随机20条', reader.sample),
        '2': ('头部20条', reader.head),
        '3': ('尾部20条', reader.tail),
        '4': ('下20条', reader.next_page),
        '5': ('上20条', reader.prev_page),
        '6': ('列信息', reader.schema)
    }

    while True:
        print(f"\n{'='*60}\n{name} ({size:.2f}MB)")
        for k, (desc, _) in ops.items():
            print(f"  [{k}] {desc}")
        print("  [b]返回 [q]退出")

        choice = input(">>> ").strip().lower()

        if choice == 'b':
            return False
        elif choice == 'q':
            return True
        elif choice in ops:
            ops[choice][1]()

# 主程序
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DIR)
    args = parser.parse_args()

    print(f"Parquet 查看工具\n数据目录: {args.data_dir}")

    if not os.path.isdir(args.data_dir):
        print("目录不存在")
        return

    browser = Browser(args.data_dir)

    while True:
        path = browser.nav()
        if not path:
            break
        if inspect_file(path):
            break

    print("退出")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\n中断退出')
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)