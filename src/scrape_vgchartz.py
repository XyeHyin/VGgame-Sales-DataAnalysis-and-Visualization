from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from requests import exceptions as req_exc
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.settings import LOGGER, OUTPUT_DIR, PLATFORM_FAMILY_MAP


DEFAULT_OUTPUT = OUTPUT_DIR / "data/vgchartz_scrape.csv"
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]
DEFAULT_USER_AGENT = USER_AGENTS[0]

DEFAULT_GENRES = [
    "Action",
    "Action-Adventure",
    "Adventure",
    "Board Game",
    "Education",
    "Fighting",
    "Misc",
    "MMO",
    "Music",
    "Party",
    "Platform",
    "Puzzle",
    "Racing",
    "Role-Playing",
    "Sandbox",
    "Shooter",
    "Simulation",
    "Sports",
    "Strategy",
    "Visual Novel",
]


@dataclass(frozen=True)
class ScrapeOptions:
    pages: int = 1
    page_size: int = 10000
    platform: str = "All"
    order: str = "Sales"
    sleep_seconds: float = 3.0
    base_url: str = "https://www.vgchartz.com/games/games.php"
    output_csv: Path = DEFAULT_OUTPUT
    user_agent: str = DEFAULT_USER_AGENT
    max_retries: int = 3
    retry_backoff: float = 2.0
    timeout: float = 30.0
    verify_ssl: bool = True
    fallback_to_http: bool = True
    dump_html_dir: Path | None = None
    ownership: str = "Both"
    direction: str = "DESC"
    include_region_sales: bool = True
    include_total_sales: bool = True
    show_multiplatform: str = "Yes"
    ignore_proxy: bool = False
    genres: Optional[List[str]] = None


class VGChartzScraper:
    columns: List[str] = [
        "Rank",
        "Name",
        "Platform",
        "Platform_Family",
        "Year",
        "Genre",
        "Publisher",
        "Developer",
        "NA_Sales",
        "EU_Sales",
        "JP_Sales",
        "Other_Sales",
        "Global_Sales",
        "VGChartz_Score",
        "Critic_Score",
        "User_Score",
        "Total_Shipped",
        "Release_Date",
        "Last_Update",
    ]

    def __init__(self, options: ScrapeOptions) -> None:
        self.options = options
        self.session = requests.Session()
        if options.ignore_proxy:
            self.session.trust_env = False
        self._rotate_user_agent()
        retry_strategy = Retry(
            total=options.max_retries,
            backoff_factor=options.retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _rotate_user_agent(self):
        ua = random.choice(USER_AGENTS)
        self.session.headers.update({"User-Agent": ua})
        LOGGER.debug("切换 User-Agent: %s", ua)

    def run(self) -> Path:
        total_rows = 0
        max_consecutive_empty = 10
        max_page_retries = 3
        genres = self.options.genres or DEFAULT_GENRES

        self.options.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.options.output_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.columns)
            writer.writeheader()

            for genre in genres:
                LOGGER.info("开始抓取 VGChartz 分类：%s", genre)
                page = 1
                consecutive_empty_pages = 0
                genre_slug = (genre or "all").replace(" ", "_")

                while page <= self.options.pages:
                    LOGGER.info("分类 %s - 第 %d 页", genre, page)
                    records = []
                    for retry in range(max_page_retries):
                        try:
                            html = self._fetch_page(page, genre)
                            records = list(self._parse_table(html, page))
                            if records:
                                break

                            LOGGER.warning(
                                "分类 %s 第 %d 页未找到数据 (重试 %d/%d)",
                                genre,
                                page,
                                retry + 1,
                                max_page_retries,
                            )
                            self._dump_html(
                                html, page, suffix=f"_{genre_slug}_retry_{retry+1}"
                            )
                            sleep_time = (
                                self.options.sleep_seconds * (retry + 1)
                            ) + random.uniform(2.0, 5.0)
                            LOGGER.info("等待 %.2f 秒后重试...", sleep_time)
                            time.sleep(sleep_time)
                            self._rotate_user_agent()

                        except Exception as e:
                            LOGGER.error("分类 %s 第 %d 页抓取异常: %s", genre, page, e)
                            time.sleep(5)

                    if not records:
                        consecutive_empty_pages += 1
                        LOGGER.warning(
                            "分类 %s 第 %d 页最终未获取到数据 (连续空页数: %d/%d)",
                            genre,
                            page,
                            consecutive_empty_pages,
                            max_consecutive_empty,
                        )
                        if consecutive_empty_pages >= max_consecutive_empty:
                            LOGGER.info(
                                "分类 %s 连续 %d 页未找到数据，停止该分类抓取",
                                genre,
                                max_consecutive_empty,
                            )
                            break
                    else:
                        consecutive_empty_pages = 0
                        for record in records:
                            record["Genre"] = genre
                            writer.writerow(record)
                            total_rows += 1
                    sleep_time = self.options.sleep_seconds + random.uniform(1.0, 4.0)
                    LOGGER.debug("分类 %s 休眠 %.2f 秒...", genre, sleep_time)
                    time.sleep(sleep_time)

                    page += 1

        LOGGER.info(
            "VGChartz 抓取完成：%d 条记录，输出 %s",
            total_rows,
            self.options.output_csv,
        )
        return self.options.output_csv

    def _fetch_page(self, page: int, genre: Optional[str] = None) -> str:
        params = self._build_query_params(page, genre)
        query = urlencode(params)
        url = f"{self.options.base_url}?{query}"
        last_error: Exception | None = None
        for attempt in range(1, self.options.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    timeout=self.options.timeout,
                    verify=self.options.verify_ssl,
                )
                response.raise_for_status()
                return response.text
            except req_exc.SSLError as exc:
                last_error = exc
                LOGGER.warning(
                    "SSL 错误（尝试 %d/%d）：%s",
                    attempt,
                    self.options.max_retries,
                    exc,
                )
                if self.options.fallback_to_http and url.startswith("https://"):
                    url = url.replace("https://", "http://", 1)
                    LOGGER.info("切换为 HTTP 继续抓取 VGChartz")
                if attempt == self.options.max_retries:
                    raise
                time.sleep(self.options.retry_backoff * attempt)
            except req_exc.RequestException as exc:
                last_error = exc
                LOGGER.warning(
                    "请求失败（尝试 %d/%d）：%s",
                    attempt,
                    self.options.max_retries,
                    exc,
                )
                if attempt == self.options.max_retries:
                    raise
                time.sleep(self.options.retry_backoff * attempt)
            except KeyboardInterrupt:
                LOGGER.error(
                    "用户中断操作。如果连接卡住，请尝试使用 --skip-ssl-verify 或 --ignore-proxy 参数。"
                )
                sys.exit(1)
        raise RuntimeError(f"VGChartz 请求失败: {last_error}")

    def _build_query_params(
        self, page: int, genre: Optional[str] = None
    ) -> Dict[str, object]:
        params: Dict[str, object] = {
            "page": page,
            "results": self.options.page_size,
            "order": self.options.order,
            "platform": self.options.platform if self.options.platform != "All" else "",
            "console": self.options.platform if self.options.platform != "All" else "",
            "ownership": self.options.ownership,
            "direction": self.options.direction,
            "showpublisher": 1,
            "showdeveloper": 1,
            "showreleasedate": 1,
            "showlastupdate": 1,
            "showvgchartzscore": 1,
            "showcriticscore": 1,
            "showuserscore": 1,
            "showshipped": 1,
            "showmultiplat": self.options.show_multiplatform,
        }
        if self.options.include_region_sales:
            params.update(
                {
                    "shownasales": 1,
                    "showpalsales": 1,
                    "showjapansales": 1,
                    "showothersales": 1,
                }
            )
        if self.options.include_total_sales:
            params["showtotalsales"] = 1
        if genre:
            params["genre"] = genre
        return params

    def _parse_table(self, html: str, page: int) -> Iterable[Dict[str, object]]:
        soup = BeautifulSoup(html, "html.parser")
        table = self._locate_table(soup)
        if not table:
            LOGGER.warning("VGChartz 页面结构变化，未能找到数据表")
            self._dump_html(html, page)
            return []
        rows = table.find_all("tr")
        header_index, headers = self._extract_headers(rows)
        if header_index is None or not headers:
            LOGGER.warning("未找到包含列标题的表头，无法解析第 %d 页", page)
            self._dump_html(html, page)
            return []
        for row in rows[header_index + 1 :]:
            cells = row.find_all("td")
            if not cells:
                continue
            if len(cells) != len(headers):
                continue
            row_map = {header: cell for header, cell in zip(headers, cells)}
            record = self._build_record(row_map)
            if record:
                yield record

    def _locate_table(self, soup: BeautifulSoup):
        main_table = soup.select_one("div#generalBody table")
        if main_table:
            return main_table
        table = soup.select_one("table.chart")
        if table:
            return table
        fallbacks = [
            soup.select_one("table#chart"),
            soup.select_one("table#chart-body"),
            soup.select_one("table.chart_table"),
        ]
        for candidate in fallbacks:
            if candidate:
                return candidate
        tables = soup.find_all("table")
        for tbl in tables:
            headers = [th.get_text(strip=True).lower() for th in tbl.find_all("th")]
            if not headers:
                continue
            if any("global" in header for header in headers) and any(
                "sales" in header for header in headers
            ):
                return tbl
        return None

    def _dump_html(self, html: str, page: int, suffix: str = "") -> None:
        if not self.options.dump_html_dir:
            return
        path = self.options.dump_html_dir / f"vgchartz_page_{page}{suffix}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        LOGGER.info("已将第 %d 页原始 HTML 写入 %s 供调试", page, path)

    def _extract_headers(self, rows: List) -> tuple[Optional[int], List[str]]:
        for idx, row in enumerate(rows):
            ths = row.find_all("th")
            if not ths:
                continue
            header_text = " ".join(th.get_text(strip=True) for th in ths)
            if "Pos" not in header_text or "Game" not in header_text:
                continue
            headers: List[str] = []
            for th in ths:
                label = th.get_text(strip=True) or ""
                colspan = int(th.get("colspan", 1))
                if label == "Game" and colspan >= 2:
                    headers.append("BoxArt")
                    if colspan > 1:
                        headers.extend(["Game"] * (colspan - 1))
                    continue
                headers.extend([label or f"col_{len(headers)}"] * colspan)

            if "Console" not in headers:
                LOGGER.warning(
                    "警告：未在表头中找到 'Console' 列。当前表头: %s", headers
                )

            return idx, headers
        return None, []

    def _build_record(self, row_map: Dict[str, Tag]) -> Optional[Dict[str, object]]:
        name = self._extract_game_name(row_map.get("Game"))
        if not name:
            return None
        platform = self._extract_platform(row_map.get("Console"))
        publisher = self._get_text(row_map.get("Publisher"))
        developer = self._get_text(row_map.get("Developer"))
        release_date_raw = self._clean_text(self._get_text(row_map.get("Release Date")))
        last_update_raw = self._clean_text(self._get_text(row_map.get("Last Update")))
        year = self._parse_release_year(release_date_raw)
        raw_rank = self._get_text(row_map.get("Pos"))
        family = PLATFORM_FAMILY_MAP.get(platform, platform)

        na_sales = self._parse_sales_cell(row_map.get("NA Sales"))
        eu_sales = self._parse_sales_cell(row_map.get("PAL Sales"))
        jp_sales = self._parse_sales_cell(row_map.get("Japan Sales"))
        other_sales = self._parse_sales_cell(row_map.get("Other Sales"))
        total_sales = self._parse_sales_cell(row_map.get("Total Sales"))
        total_shipped = self._parse_sales_cell(row_map.get("Total Shipped"))
        global_sales = total_sales if total_sales > 0 else total_shipped

        record: Dict[str, object] = {
            "Rank": self._safe_int(raw_rank),
            "Name": name,
            "Platform": platform,
            "Platform_Family": family,
            "Year": year,
            "Genre": "",
            "Publisher": publisher,
            "Developer": developer,
            "NA_Sales": na_sales,
            "EU_Sales": eu_sales,
            "JP_Sales": jp_sales,
            "Other_Sales": other_sales,
            "Global_Sales": global_sales,
            "VGChartz_Score": self._safe_float(
                self._get_text(row_map.get("VGChartz Score"))
            ),
            "Critic_Score": self._safe_float(
                self._get_text(row_map.get("Critic Score"))
            ),
            "User_Score": self._safe_float(self._get_text(row_map.get("User Score"))),
            "Total_Shipped": total_shipped,
            "Release_Date": release_date_raw,
            "Last_Update": last_update_raw,
        }
        return record

    @staticmethod
    def _extract_game_name(cell: Optional[Tag]) -> str:
        if cell is None:
            return ""
        link = cell.find("a")
        if link:
            return link.get_text(strip=True)
        return cell.get_text(strip=True)

    @staticmethod
    def _extract_platform(cell: Optional[Tag]) -> str:
        if cell is None:
            return ""
        img = cell.find("img")
        if img and img.get("alt"):
            return img.get("alt").strip()
        return cell.get_text(strip=True)

    @staticmethod
    def _get_text(cell: Optional[Tag]) -> str:
        if cell is None:
            return ""
        return cell.get_text(strip=True)

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def _parse_release_year(self, text: str) -> int:
        if not text:
            return 0
        cleaned = re.sub(r"(st|nd|rd|th)", "", text)
        match = re.search(r"(\d{2,4})$", cleaned.strip())
        if not match:
            return 0
        year_value = int(match.group(1))
        if year_value >= 1900:
            return year_value
        if year_value >= 70:
            return 1900 + year_value
        return 2000 + year_value

    def _parse_sales_cell(self, cell: Optional[Tag]) -> float:
        if cell is None:
            return 0.0
        return self._parse_sales(cell.get_text(strip=True))

    @staticmethod
    def _safe_float(value: str) -> Optional[float]:
        if not value or value.upper() == "N/A":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _safe_int(value: str) -> int:
        try:
            return int(value.replace("#", ""))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _parse_sales(value: str) -> float:
        cleaned = value.replace("m", "").replace("M", "").strip()
        if cleaned in {"", "-"}:
            return 0.0
        try:
            return round(float(cleaned), 3)
        except ValueError:
            return 0.0


def parse_args() -> ScrapeOptions:
    parser = argparse.ArgumentParser(description="Scrape VGChartz game DB into CSV")
    parser.add_argument("--pages", type=int, default=1, help="Number of pages to crawl")
    parser.add_argument("--page-size", type=int, default=10000, help="Rows per page")
    parser.add_argument("--platform", default="All", help="VGChartz platform filter")
    parser.add_argument("--order", default="Sales", help="Sort order, default by Sales")
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        help="Delay between requests (seconds) to stay polite",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output CSV path (default outputs/vgchartz_scrape.csv)",
    )
    parser.add_argument(
        "--base-url",
        default="https://www.vgchartz.com/games/games.php",
        help="VGChartz listing URL (override when站点结构变化)",
    )
    parser.add_argument(
        "--dump-html-dir",
        help="Directory to dump raw HTML when parsing fails (optional)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per page when request/SSL fails (default 3)",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.5,
        help="Base seconds for exponential backoff between retries",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default 30)",
    )
    parser.add_argument(
        "--skip-ssl-verify",
        action="store_true",
        help="Disable SSL certificate verification (not recommended)",
    )
    parser.add_argument(
        "--no-http-fallback",
        action="store_true",
        help="Disable HTTPS→HTTP fallback when SSL fails",
    )
    parser.add_argument(
        "--ignore-proxy",
        action="store_true",
        help="Ignore environment proxy settings",
    )
    parser.add_argument(
        "--genres",
        nargs="+",
        help="Specific VGChartz genres to crawl (default: all site genres)",
    )
    args = parser.parse_args()
    genre_filters = (
        [genre.strip() for genre in args.genres if genre.strip()]
        if args.genres
        else None
    )

    return ScrapeOptions(
        pages=max(args.pages, 1),
        page_size=max(args.page_size, 1),
        platform=args.platform,
        order=args.order,
        sleep_seconds=max(args.sleep, 0.0),
        base_url=args.base_url,
        output_csv=Path(args.output),
        max_retries=max(args.max_retries, 1),
        retry_backoff=max(args.retry_backoff, 0.2),
        timeout=max(args.timeout, 5.0),
        verify_ssl=not args.skip_ssl_verify,
        fallback_to_http=not args.no_http_fallback,
        dump_html_dir=Path(args.dump_html_dir) if args.dump_html_dir else None,
        include_region_sales=True,
        include_total_sales=True,
        ignore_proxy=args.ignore_proxy,
        genres=genre_filters,
    )


def main() -> None:
    options = parse_args()
    scraper = VGChartzScraper(options)
    scraper.run()


if __name__ == "__main__":
    main()
