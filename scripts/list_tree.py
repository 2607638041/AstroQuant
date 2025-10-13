# scripts/list_tree.py
import os
import argparse  # 命令行参数解析
from pathlib import Path    # Python 3.4+

# 需要排除的目录/文件
EXCLUDE_DIRS = {'.idea', 'venv', '__pycache__'}  # 排除IDEA、虚拟环境、Python缓存目录
EXCLUDE_FILES = {'.DS_Store'}           # 排除Mac系统文件

def build_tree(root: Path, max_depth: int = 3) -> str:  # 定义函数
    """构建目录树文本"""
    lines = []  # 目录树文本行

    def _walk(path: Path, depth: int):
        if depth > max_depth:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            return
        for entry in entries:
            if entry.name in EXCLUDE_DIRS or entry.name in EXCLUDE_FILES:
                continue
            indent = '    ' * depth
            lines.append(f"{indent}- {entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                _walk(entry, depth + 1)

    _walk(root, 0)
    return '\n'.join(lines)  # 目录树文本


def main():
    parser = argparse.ArgumentParser(description="生成目录树")
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="项目根目录 (默认：脚本上级目录)",
    )
    parser.add_argument(
        "--depth", type=int, default=3, help="最大递归深度 (默认: 3)"
    )
    args = parser.parse_args()

    tree_text = build_tree(args.root, args.depth)
    print(tree_text)


if __name__ == "__main__":  # 脚本运行入口
    main()