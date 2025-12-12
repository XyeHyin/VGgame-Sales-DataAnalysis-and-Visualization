import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


CONFIG_PATH = Path("config.json")


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件 {path} 不存在")
    return json.loads(path.read_text(encoding="utf-8"))


APP_CONFIG = _load_config(CONFIG_PATH)
PATHS_CONFIG = APP_CONFIG["paths"]
OUTPUT_NAMES = APP_CONFIG["outputs"]
LOGGING_CONFIG = APP_CONFIG["logging"]
POSTGRES_DEFAULTS = APP_CONFIG["postgres"]
DASHBOARD_TEXT = APP_CONFIG["dashboard"]
TAXONOMY = APP_CONFIG["taxonomy"]

DATA_FILE = Path(PATHS_CONFIG["data_file"])
OUTPUT_DIR = Path(PATHS_CONFIG["output_dir"])
DASHBOARD_TEMPLATE_PATH = Path(PATHS_CONFIG["dashboard_template"])
DASHBOARD_SUMMARY_TEMPLATE_PATH = Path(PATHS_CONFIG["dashboard_summary_template"])
CLEAN_CSV_PATH = OUTPUT_DIR / OUTPUT_NAMES["clean_csv"]
CLEAN_PARQUET_PATH = OUTPUT_DIR / OUTPUT_NAMES["clean_parquet"]
QUALITY_JSON_PATH = OUTPUT_DIR / OUTPUT_NAMES["quality_json"]
METRICS_JSON_PATH = OUTPUT_DIR / OUTPUT_NAMES["metrics_json"]
SUMMARY_MD_PATH = OUTPUT_DIR / OUTPUT_NAMES["summary_md"]
DASHBOARD_PATH = OUTPUT_DIR / OUTPUT_NAMES["dashboard_html"]
ML_METRICS_PATH = OUTPUT_DIR / OUTPUT_NAMES["ml_metrics_json"]
ML_CLUSTERS_PATH = OUTPUT_DIR / OUTPUT_NAMES["ml_clusters_json"]
ML_PREDICTIONS_PATH = OUTPUT_DIR / OUTPUT_NAMES["ml_predictions_json"]
ML_FEATURE_PNG_PATH = OUTPUT_DIR / OUTPUT_NAMES["ml_feature_png"]

REGION_COLS = TAXONOMY["region_columns"]
REGION_LABELS = TAXONOMY["region_labels"]
GLOBAL_SALES_LABEL_CN = TAXONOMY["global_sales_label"]
PLATFORM_FAMILY_MAP = TAXONOMY["platform_family_map"]
PLATFORM_LABELS = TAXONOMY["platform_labels"]
GENRE_LABELS = TAXONOMY["genre_labels"]


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
    """Create an OutputArtifacts bundle rooted at the given directory."""
    return OutputArtifacts(
        directory=directory,
        clean_csv=directory / CLEAN_CSV_PATH.name,
        clean_parquet=directory / CLEAN_PARQUET_PATH.name,
        quality_json=directory / QUALITY_JSON_PATH.name,
        metrics_json=directory / METRICS_JSON_PATH.name,
        summary_md=directory / SUMMARY_MD_PATH.name,
        dashboard_html=directory / DASHBOARD_PATH.name,
        ml_metrics_json=directory / ML_METRICS_PATH.name,
        ml_clusters_json=directory / ML_CLUSTERS_PATH.name,
        ml_predictions_json=directory / ML_PREDICTIONS_PATH.name,
        ml_feature_png=directory / ML_FEATURE_PNG_PATH.name,
    )
