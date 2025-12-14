"""项目主入口 - 运行数据分析管线"""

import sys
import os
os.environ["NUMEXPR_MAX_THREADS"] = "20"
os.environ["NUMEXPR_NUM_THREADS"] = "20"
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from src.scrape_vgchartz import VGChartzScraper, ScrapeOptions
from src.clean_and_store import run_cleaning
from src.analysis import main as run_analysis
from src.settings import DATA_FILE, OUTPUT_DIR


def run_scrape():
    print("\n=== 执行爬取任务 ===")
    options = ScrapeOptions(
        pages=1, page_size=10000, ignore_proxy=True, output_csv=DATA_FILE
    )
    print(f"目标文件: {DATA_FILE}")
    scraper = VGChartzScraper(options)
    scraper.run()
    print("=== 爬取任务完成 ===\n")


def run_clean():
    print("\n=== 执行清洗任务 ===")
    run_cleaning(input_path=DATA_FILE, output_dir=OUTPUT_DIR, skip_db=False)
    print("=== 清洗任务完成 ===\n")


def run_analyze():
    print("\n=== 执行分析任务 ===")
    run_analysis()
    print("=== 分析任务完成 ===\n")


def interactive_menu():
    while True:
        print("\n请选择执行模式:")
        print("0: 全流程执行 (爬取 -> 清洗 -> 分析)")
        print("1: 仅爬取")
        print("2: 仅清洗")
        print("3: 仅分析")
        print("4: 清洗 + 分析")
        print("q: 退出")

        choice = input("请输入选项: ").strip().lower()

        if choice == "0":
            run_scrape()
            run_clean()
            run_analyze()
        elif choice == "1":
            run_scrape()
        elif choice == "2":
            run_clean()
        elif choice == "3":
            run_analyze()
        elif choice == "4":
            run_clean()
            run_analyze()
        elif choice == "q":
            print("退出程序")
            break
        else:
            print("无效选项，请重试")


if __name__ == "__main__":
    interactive_menu()
