import argparse
import os
import sys
from datetime import datetime


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    i = 0
    while size >= 1024 and i < len(units) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.2f} {units[i]}"


def list_dir(path: str, max_depth: int = 2):
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return
    print(f"Root: {path}")
    print("")

    def _walk(cur: str, depth: int):
        try:
            entries = sorted(os.scandir(cur), key=lambda e: e.name)
        except PermissionError:
            print(f"[Permission denied] {cur}")
            return
        except FileNotFoundError:
            print(f"[Missing] {cur}")
            return
        for e in entries:
            try:
                stat = e.stat(follow_symlinks=False)
                size = format_size(stat.st_size)
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                prefix = "  " * depth
                if e.is_dir(follow_symlinks=False):
                    print(f"{prefix}[DIR] {e.name}  (modified {mtime})")
                    if depth < max_depth:
                        _walk(e.path, depth + 1)
                else:
                    print(f"{prefix}{e.name}  {size}  (modified {mtime})")
            except Exception as exc:
                print(f"Error reading {e.path}: {exc}")

    _walk(path, 0)


def parquet_summary(path: str):
    try:
        import pyarrow.parquet as pq
    except Exception:
        print("pyarrow not available; skipping parquet summary.")
        return

    if not os.path.isdir(path):
        print(f"Not a directory: {path}")
        return

    print("")
    print("Parquet files summary:")
    for name in sorted(os.listdir(path)):
        if name.endswith(".parquet"):
            fpath = os.path.join(path, name)
            try:
                pf = pq.ParquetFile(fpath)
                rows = pf.metadata.num_rows
                cols = pf.metadata.num_columns
                created_by = pf.metadata.created_by or ""
                print(f"- {name}: rows={rows}, cols={cols}, created_by={created_by}")
            except Exception as exc:
                print(f"- {name}: failed to read metadata ({exc})")


def main():
    parser = argparse.ArgumentParser(description="List dataset directory contents and optional parquet summaries.")
    parser.add_argument(
        "--path",
        default="./",
        help="Dataset root path",
    )
    parser.add_argument("--max-depth", type=int, default=2, help="Max directory depth to display")
    parser.add_argument("--show-parquet", action="store_true", help="Show parquet metadata in root directory")
    args = parser.parse_args()

    list_dir(args.path, max_depth=args.max_depth)
    if args.show_parquet:
        parquet_summary(args.path)


if __name__ == "__main__":
    sys.exit(main())
