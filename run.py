import sys
import os

os.environ["NUMEXPR_MAX_THREADS"] = "20"
os.environ["NUMEXPR_NUM_THREADS"] = "20"
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from src.scrape_vgchartz import VGChartzScraper, ScrapeOptions, DEFAULT_GENRES
from src.clean_and_store import run_cleaning
from src.analysis import main as run_analysis
from src.settings import DATA_FILE, OUTPUT_DIR


def get_user_input(prompt: str, default: str) -> str:
    """获取用户输入，如果为空则使用默认值"""
    value = input(f"{prompt} [默认: {default}]: ").strip()
    return value if value else default


def get_scrape_options_interactive() -> ScrapeOptions:
    """交互式获取爬取配置"""
    print("\n--- 配置爬取参数 ---")

    # 基础参数
    pages = int(get_user_input("爬取页数", "1"))
    page_size = int(get_user_input("每页数量", "10000"))
    print(f"\n可用游戏类型: {', '.join(DEFAULT_GENRES)}")
    genres_input = input("请输入要爬取的游戏类型 (逗号分隔, 留空则爬取所有): ").strip()
    genres = [g.strip() for g in genres_input.split(",")] if genres_input else None
    ignore_proxy_str = get_user_input("忽略系统代理 (y/n)", "y").lower()
    ignore_proxy = ignore_proxy_str == "y"

    return ScrapeOptions(
        pages=pages,
        page_size=page_size,
        genres=genres,
        ignore_proxy=ignore_proxy,
        output_csv=DATA_FILE,
    )


def run_scrape():
    print("\n=== 执行爬取任务 ===")

    # 询问是否使用默认配置
    use_default = (
        input("是否使用默认配置(1页, 10000条/页)? (y/n) [默认: y]: ").strip().lower()
    )

    if use_default == "n":
        options = get_scrape_options_interactive()
    else:
        options = ScrapeOptions(
            pages=1, page_size=10000, ignore_proxy=True, output_csv=DATA_FILE
        )

    print(f"\n当前配置:")
    print(f"- 页数: {options.pages}")
    print(f"- 每页数量: {options.page_size}")
    print(f"- 游戏类型: {options.genres if options.genres else 'All'}")
    print(f"- 忽略代理: {options.ignore_proxy}")
    print(f"目标文件: {DATA_FILE}")

    confirm = input("\n确认开始爬取? (y/n) [默认: y]: ").strip().lower()
    if confirm == "n":
        print("已取消爬取任务")
        return

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
