from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.data_cleaning import GameDataCleaner
from src.database import PostgresConfig, PostgresWriter
from src.settings import DATA_FILE, LOGGER, OUTPUT_DIR, build_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="清理 VGChartz 原始数据并写入 PostgreSQL"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_FILE,
        help="原始 CSV 文件路径，默认使用 config.json 中的 paths.data_file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="清洗结果输出目录，默认使用 config.json 中的 paths.output_dir",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="跳过写入 PostgreSQL 的步骤",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="机器学习步骤使用的随机种子",
    )
    return parser.parse_args()


def run_cleaning(
    input_path: Path,
    output_dir: Path,
    skip_db: bool = False,
    random_state: int = 42,
) -> None:
    """
    执行数据清洗流程
    """
    if not input_path.exists():
        # 如果未在指定位置找到，尝试在输出目录中查找同名文件作为回退
        alt_path = OUTPUT_DIR / input_path.name
        if alt_path.exists():
            LOGGER.warning("未在 %s 找到原始数据，改为使用：%s", input_path, alt_path)
            input_path = alt_path
        else:
            LOGGER.error("原始数据文件不存在：%s", input_path)
            LOGGER.error("回退路径也未找到：%s", alt_path)
            sys.exit(1)

    LOGGER.info("读取原始数据：%s", input_path)
    raw_df = pd.read_csv(input_path)
    cleaner = GameDataCleaner(random_state=random_state)
    result = cleaner.clean(raw_df)

    artifacts = build_artifacts(output_dir)
    cleaner.save_clean_outputs(
        result.dataframe, artifacts.clean_csv, artifacts.clean_parquet
    )
    quality_report = cleaner.build_quality_report(result.dataframe, result.summary)
    cleaner.save_quality_report(quality_report, artifacts.quality_json)
    cleaner.log_console_report(result.summary)

    if skip_db:
        LOGGER.info("已跳过 PostgreSQL 写入 (skip-db)")
    else:
        LOGGER.info("写入 PostgreSQL 数据库")
        writer = PostgresWriter(PostgresConfig())
        writer.write(result.dataframe)

    LOGGER.info("清理程序执行完毕，产出文件目录：%s", output_dir.resolve())


def main() -> None:
    args = parse_args()
    run_cleaning(
        input_path=args.input,
        output_dir=args.output_dir,
        skip_db=args.skip_db,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
