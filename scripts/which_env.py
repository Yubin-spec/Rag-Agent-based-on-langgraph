#!/usr/bin/env python3
"""打印当前 Python 解释器路径与 prefix，用于确认运行/开发时使用的是哪个环境。"""
import sys
from pathlib import Path

def main():
    exe = getattr(sys, "executable", "?")
    prefix = getattr(sys, "prefix", "?")
    root = Path(__file__).resolve().parent.parent
    venv_python = None
    for name in ("bin/python3", "bin/python", "Scripts/python.exe", "Scripts/python3.exe"):
        p = root / ".venv" / name
        if p.exists():
            venv_python = p
            break
    using_venv = bool(venv_python and exe != "?" and Path(exe).resolve() == venv_python.resolve())
    print("executable:", exe)
    print("prefix:   ", prefix)
    print("using .venv:", using_venv)
    if not using_venv and (root / ".venv").exists():
        print("(tip: run `source .venv/bin/activate` then run again to use project .venv)")

if __name__ == "__main__":
    main()
