#!/usr/bin/env python3
"""
初始化 Text2SQL 使用的 SQLite 示例表（可选）。
表结构需与 src/kb/text2sql.py 中 DEFAULT_SCHEMA 一致，便于自然语言转 SQL 查询。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, text
from config import get_settings


def main():
    """创建 import_export_stats、aeoc_enterprises 表（若不存在）。"""
    settings = get_settings()
    engine = create_engine(settings.text2sql_database_uri)
    ROOT.joinpath("data").mkdir(exist_ok=True)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS import_export_stats (
                year INT, port VARCHAR(64), category VARCHAR(128), volume REAL, unit VARCHAR(32)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS aeoc_enterprises (
                name VARCHAR(256), level VARCHAR(64), region VARCHAR(64)
            )
        """))
        conn.commit()
    print("SQLite 示例表已创建。可插入示例数据以便 Text2SQL 查询。")

if __name__ == "__main__":
    main()
