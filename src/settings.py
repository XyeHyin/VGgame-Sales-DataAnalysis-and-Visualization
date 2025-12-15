import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).parent.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
USER_CONFIG_PATH = PROJECT_ROOT / "config" / "user_config.json"
MAPPER_PATH = PROJECT_ROOT / "config" / "mapper.json"


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


APP_CONFIG = _load_config(CONFIG_PATH)
USER_CONFIG = _load_config(USER_CONFIG_PATH)
MAPPER_CONFIG = _load_config(MAPPER_PATH)
PATHS_CONFIG = APP_CONFIG.get("paths", {})
OUTPUT_NAMES = APP_CONFIG.get("outputs", {})
LOGGING_CONFIG = USER_CONFIG.get(
    "logging",
    {
        "level": "INFO",
        "format": "%(asctime)s | %(levelname)s | %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    },
)
POSTGRES_DEFAULTS = USER_CONFIG.get(
    "postgres",
    {
        "host": "localhost",
        "port": 5432,
        "database": "postgres",
        "user": "postgres",
        "password": "123400",
        "table_name": "vgchartz_clean",
    },
)
DASHBOARD_TEXT = USER_CONFIG.get(
    "dashboard",
    {
        "page_title": "全球电子游戏销量交互式仪表盘",
        "hero_title": "全球电子游戏销量交互式仪表盘",
        "hero_subtitle": "",
        "data_source": "",
    },
)
TAXONOMY = MAPPER_CONFIG.get("taxonomy", {})

DATA_FILE = PROJECT_ROOT / Path(
    PATHS_CONFIG.get("data_file", "data/vgchartz_scrape.csv")
)
OUTPUT_DIR = PROJECT_ROOT / Path(PATHS_CONFIG.get("output_dir", "outputs"))
DASHBOARD_TEMPLATE_PATH = PROJECT_ROOT / Path(
    PATHS_CONFIG.get("dashboard_template", "templates/dashboard.j2")
)
DASHBOARD_SUMMARY_TEMPLATE_PATH = PROJECT_ROOT / Path(
    PATHS_CONFIG.get("dashboard_summary_template", "templates/dashboard_summary.j2")
)
CLEAN_CSV_PATH = OUTPUT_DIR / OUTPUT_NAMES.get("clean_csv", "data/vgchartz_clean.csv")
CLEAN_PARQUET_PATH = OUTPUT_DIR / OUTPUT_NAMES.get(
    "clean_parquet", "data/vgchartz_clean.parquet"
)
QUALITY_JSON_PATH = OUTPUT_DIR / OUTPUT_NAMES.get(
    "quality_json", "metrics/data_quality.json"
)
METRICS_JSON_PATH = OUTPUT_DIR / OUTPUT_NAMES.get(
    "metrics_json", "metrics/analysis_metrics.json"
)
SUMMARY_MD_PATH = OUTPUT_DIR / OUTPUT_NAMES.get(
    "summary_md", "reports/analysis_summary.md"
)
DASHBOARD_PATH = OUTPUT_DIR / OUTPUT_NAMES.get("dashboard_html", "dashboard.html")
ML_METRICS_PATH = OUTPUT_DIR / OUTPUT_NAMES.get(
    "ml_metrics_json", "metrics/ml_metrics.json"
)
ML_CLUSTERS_PATH = OUTPUT_DIR / OUTPUT_NAMES.get(
    "ml_clusters_json", "metrics/ml_clusters.json"
)
ML_PREDICTIONS_PATH = OUTPUT_DIR / OUTPUT_NAMES.get(
    "ml_predictions_json", "metrics/ml_predictions_sample.json"
)
ML_FEATURE_PNG_PATH = OUTPUT_DIR / OUTPUT_NAMES.get(
    "ml_feature_png", "plots/ml_feature_importance.png"
)

REGION_COLS = TAXONOMY.get(
    "region_columns", ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
)
REGION_LABELS = TAXONOMY.get("region_labels", {})
GLOBAL_SALES_LABEL_CN = TAXONOMY.get("global_sales_label", "全球销量（百万套）")
PLATFORM_FAMILY_MAP = TAXONOMY.get("platform_family_map", {})
PLATFORM_LABELS = TAXONOMY.get("platform_labels", {})
GENRE_LABELS = TAXONOMY.get("genre_labels", {})


@dataclass(frozen=True)
class DashboardConfig:
    template_path: Path
    summary_template_path: Path
    page_title: str
    hero_title: str
    hero_subtitle: str
    data_source: str


@dataclass(frozen=True)
class OutputArtifacts:
    directory: Path
    clean_csv: Path
    clean_parquet: Path
    quality_json: Path
    metrics_json: Path
    summary_md: Path
    dashboard_html: Path
    ml_metrics_json: Path
    ml_clusters_json: Path
    ml_predictions_json: Path
    ml_feature_png: Path


DASHBOARD_CONFIG = DashboardConfig(
    template_path=DASHBOARD_TEMPLATE_PATH,
    summary_template_path=DASHBOARD_SUMMARY_TEMPLATE_PATH,
    page_title=DASHBOARD_TEXT["page_title"],
    hero_title=DASHBOARD_TEXT["hero_title"],
    hero_subtitle=DASHBOARD_TEXT["hero_subtitle"],
    data_source=DASHBOARD_TEXT["data_source"],
)

DEFAULT_ARTIFACTS = OutputArtifacts(
    directory=OUTPUT_DIR,
    clean_csv=CLEAN_CSV_PATH,
    clean_parquet=CLEAN_PARQUET_PATH,
    quality_json=QUALITY_JSON_PATH,
    metrics_json=METRICS_JSON_PATH,
    summary_md=SUMMARY_MD_PATH,
    dashboard_html=DASHBOARD_PATH,
    ml_metrics_json=ML_METRICS_PATH,
    ml_clusters_json=ML_CLUSTERS_PATH,
    ml_predictions_json=ML_PREDICTIONS_PATH,
    ml_feature_png=ML_FEATURE_PNG_PATH,
)

LOGGER = logging.getLogger("vgsales")
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"].upper(), logging.INFO),
    format=LOGGING_CONFIG["format"],
    datefmt=LOGGING_CONFIG["datefmt"],
)


def build_artifacts(directory: Path) -> OutputArtifacts:
    def reroot(path: Path) -> Path:
        try:
            rel = path.relative_to(OUTPUT_DIR)
        except ValueError:
            rel = Path(path.name)
        return directory / rel

    return OutputArtifacts(
        directory=directory,
        clean_csv=reroot(CLEAN_CSV_PATH),
        clean_parquet=reroot(CLEAN_PARQUET_PATH),
        quality_json=reroot(QUALITY_JSON_PATH),
        metrics_json=reroot(METRICS_JSON_PATH),
        summary_md=reroot(SUMMARY_MD_PATH),
        dashboard_html=reroot(DASHBOARD_PATH),
        ml_metrics_json=reroot(ML_METRICS_PATH),
        ml_clusters_json=reroot(ML_CLUSTERS_PATH),
        ml_predictions_json=reroot(ML_PREDICTIONS_PATH),
        ml_feature_png=reroot(ML_FEATURE_PNG_PATH),
    )
