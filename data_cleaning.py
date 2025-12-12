from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from settings import (
    GENRE_LABELS,
    LOGGER,
    PLATFORM_FAMILY_MAP,
    PLATFORM_LABELS,
    REGION_COLS,
    REGION_LABELS,
)


@dataclass
class CleaningResult:
    dataframe: pd.DataFrame
    summary: Dict[str, object]


class GameDataCleaner:
    """Cleans raw VGChartz data with ML-enhanced imputation and clustering."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def clean(self, raw_df: pd.DataFrame) -> CleaningResult:
        LOGGER.info("启动独立数据清理程序")
        work = raw_df.copy()
        work.columns = [col.strip() for col in work.columns]
        work.replace(
            {"": np.nan, "N/A": np.nan, "na": np.nan, "None": np.nan}, inplace=True
        )

        stats: Dict[str, object] = {
            "initial_rows": int(len(work)),
            "initial_columns": list(work.columns),
            "missing_before": work.isna().sum().to_dict(),
        }

        work = self._standardize_strings(work)
        work = self._parse_temporal_fields(work)
        work = self._coerce_numeric_fields(work)
        work = self._fill_sales_features(work)
        work, filter_stats = self._filter_records(work)
        stats.update(filter_stats)
        work = self._engineer_features(work)
        work, ml_stats = self._apply_ml_enrichment(work)
        stats.update(ml_stats)

        stats["final_rows"] = int(len(work))
        stats["missing_after"] = work.isna().sum().to_dict()
        stats["deduplication_rate"] = self._safe_ratio(
            stats.get("duplicates_removed", 0),
            stats["initial_rows"],
        )
        stats["retention_rate"] = self._safe_ratio(
            stats["final_rows"], stats["initial_rows"]
        )

        LOGGER.info(
            "数据清理完成：保留 %d/%d 行 (%.2f%%)",
            stats["final_rows"],
            stats["initial_rows"],
            100 * stats["retention_rate"],
        )
        return CleaningResult(work, stats)

    def save_clean_outputs(
        self, df: pd.DataFrame, clean_csv: Path, clean_parquet: Path
    ) -> None:
        clean_csv.parent.mkdir(parents=True, exist_ok=True)
        clean_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(clean_csv, index=False, encoding="utf-8-sig")
        df.to_parquet(clean_parquet, index=False)
        LOGGER.info("已输出清洗数据：%s, %s", clean_csv.name, clean_parquet.name)

    def build_quality_report(
        self, df: pd.DataFrame, stats: Dict[str, object]
    ) -> Dict[str, object]:
        return {
            "记录数量": int(len(df)),
            "字段类型": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "缺失值统计": df.isna().sum().to_dict(),
            "重复行数量": int(df.duplicated().sum()),
            "时间范围": {
                "最早年份": int(df["Year"].min()),
                "最晚年份": int(df["Year"].max()),
            },
            "清洗摘要": stats,
        }

    def save_quality_report(self, report: Dict[str, object], destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info("已生成数据质量报告：%s", destination.name)

    def log_console_report(self, stats: Dict[str, object]) -> None:
        LOGGER.info("===== 数据清理报告 =====")
        LOGGER.info("原始行数：%d", stats.get("initial_rows", 0))
        LOGGER.info("最终行数：%d", stats.get("final_rows", 0))
        LOGGER.info("重复记录移除：%d", stats.get("duplicates_removed", 0))
        LOGGER.info("缺失关键字段移除：%d", stats.get("missing_key_rows", 0))
        LOGGER.info("销量无效移除：%d", stats.get("invalid_sales_rows", 0))
        LOGGER.info(
            "KNN 缺失值插补：%s",
            "执行" if stats.get("knn_imputer") else "跳过",
        )
        if stats.get("knn_imputer"):
            LOGGER.info("  - 使用特征：%s", ", ".join(stats["knn_imputer"]["features"]))
            LOGGER.info("  - 邻居数：%d", stats["knn_imputer"]["n_neighbors"])
        LOGGER.info(
            "区域聚类：%s",
            "执行" if stats.get("regional_clustering") else "跳过",
        )
        if stats.get("regional_clustering"):
            LOGGER.info("  - 聚类数：%d", stats["regional_clustering"]["n_clusters"])
        LOGGER.info("数据保留率：%.2f%%", 100 * stats.get("retention_rate", 0.0))
        LOGGER.info("========================")

    def _standardize_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        text_cols = [
            "Name",
            "Platform",
            "Platform_Family",
            "Genre",
            "Publisher",
            "Developer",
            "Release_Date",
            "Last_Update",
        ]
        for col in text_cols:
            if col in df.columns:
                series = df[col].astype("string").str.strip()
                df[col] = series.replace(
                    {"": pd.NA, "nan": pd.NA, "None": pd.NA, "N/A": pd.NA}
                )
        return df

    def _parse_temporal_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ("Release_Date", "Last_Update"):
            if col in df.columns:
                df[col] = self._parse_date_column(df[col])

        if "Year" in df.columns:
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        else:
            df["Year"] = np.nan

        if "Release_Date" in df.columns:
            release_year = df["Release_Date"].dt.year
            mask = df["Year"].isna() & release_year.notna()
            df.loc[mask, "Year"] = release_year[mask]

        if "Last_Update" in df.columns:
            update_year = df["Last_Update"].dt.year
            mask = df["Year"].isna() & update_year.notna()
            df.loc[mask, "Year"] = update_year[mask]

        current_year = datetime.now().year
        df.dropna(subset=["Year"], inplace=True)
        df["Year"] = df["Year"].clip(1970, current_year).astype(int)
        return df

    def _coerce_numeric_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = [
            "Rank",
            "NA_Sales",
            "EU_Sales",
            "JP_Sales",
            "Other_Sales",
            "Global_Sales",
            "Total_Shipped",
            "VGChartz_Score",
            "Critic_Score",
            "User_Score",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _fill_sales_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Platform" in df.columns:
            df["Platform"] = (
                df["Platform"].astype("string").str.replace(r"\s+", " ", regex=True)
            )

        if "Platform_Family" not in df.columns:
            df["Platform_Family"] = df["Platform"].map(PLATFORM_FAMILY_MAP)
        else:
            df["Platform_Family"] = df["Platform_Family"].fillna(
                df["Platform"].map(PLATFORM_FAMILY_MAP)
            )
        df["Platform_Family"] = df["Platform_Family"].fillna("Other")

        if "Genre" in df.columns:
            df["Genre"] = df["Genre"].fillna("其他")
        else:
            df["Genre"] = "其他"

        region_cols = [col for col in REGION_COLS if col in df.columns]
        for region in region_cols:
            df[region] = df[region].clip(lower=0)

        global_sales = (
            pd.to_numeric(df["Global_Sales"], errors="coerce")
            if "Global_Sales" in df.columns
            else pd.Series(np.nan, index=df.index, dtype="float64")
        )
        df["Global_Sales"] = global_sales

        if "Total_Shipped" in df.columns:
            total_shipped = pd.to_numeric(df["Total_Shipped"], errors="coerce")
            df["Total_Shipped"] = total_shipped
            fill_mask = (
                (df["Global_Sales"].isna() | (df["Global_Sales"] == 0))
                & total_shipped.notna()
                & (total_shipped != 0)
            )
            df.loc[fill_mask, "Global_Sales"] = total_shipped[fill_mask]

        if region_cols:
            df["Regional_Sales_Sum"] = df[region_cols].sum(axis=1)
        else:
            df["Regional_Sales_Sum"] = 0.0
        return df

    def _filter_records(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        stats = {
            "duplicates_removed": 0,
            "missing_key_rows": 0,
            "invalid_sales_rows": 0,
        }

        if "Platform" in df.columns:
            platform_series = df["Platform"].astype("string").str.lower()
            mask = platform_series.ne("series") | platform_series.isna()
            before = len(df)
            df = df[mask]
            stats["series_rows_removed"] = before - len(df)

        before = len(df)
        df = df.dropna(subset=["Name", "Platform", "Global_Sales"])
        stats["missing_key_rows"] = before - len(df)

        before = len(df)
        df = df[df["Global_Sales"] > 0]
        stats["invalid_sales_rows"] = before - len(df)

        before = len(df)
        current_year = datetime.now().year
        df = df[df["Year"].between(1970, current_year)]
        stats["out_of_range_years"] = before - len(df)

        before = len(df)
        df = df.drop_duplicates(
            subset=["Name", "Platform", "Year", "Publisher"], keep="first"
        )
        stats["duplicates_removed"] = before - len(df)

        return df, stats

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Genre" in df.columns:
            df["Genre"] = df["Genre"].map(GENRE_LABELS).fillna(df["Genre"])

        region_cols = [col for col in REGION_COLS if col in df.columns]
        if region_cols:
            df["Top_Region"] = df[region_cols].idxmax(axis=1)
            df["Top_Region_CN"] = df["Top_Region"].map(REGION_LABELS)
            denominator = df["Global_Sales"].replace(0, np.nan)
            for region in region_cols:
                share_col = f"{region}_Share"
                df[share_col] = (df[region] / denominator).round(4)

        df["Platform_Family_CN"] = (
            df["Platform_Family"].map(PLATFORM_LABELS).fillna("其他平台")
        )
        df["Decade"] = (df["Year"] // 10) * 10
        df["Age_Years"] = (datetime.now().year - df["Year"]).clip(lower=0)
        if "Release_Date" in df.columns:
            df["Release_Quarter"] = (
                df["Release_Date"].dt.to_period("Q").astype("string")
            )

        df["Composite_Score"] = df[["Critic_Score", "User_Score"]].mean(axis=1)
        df["Score_Gap"] = df["Critic_Score"] - df["User_Score"]
        df["Is_Modern_Title"] = df["Year"] >= 2015

        sales_bins = [-np.inf, 1, 5, 10, 20, np.inf]
        sales_labels = ["<1M", "1-5M", "5-10M", "10-20M", "20M+"]
        df["Sales_Tier"] = pd.cut(
            df["Global_Sales"], bins=sales_bins, labels=sales_labels
        )

        score_bins = [-np.inf, 6, 8, np.inf]
        score_labels = ["<=6", "6_to_8", ">=8"]
        df["Score_Tier"] = pd.cut(
            df["Composite_Score"], bins=score_bins, labels=score_labels
        )
        df["Has_Critic_Score"] = df["Critic_Score"].notna()
        df["Has_User_Score"] = df["User_Score"].notna()

        df["Canonical_Name"] = (
            df["Name"]
            .astype("string")
            .str.replace(r"[®™©]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        return df

    def _apply_ml_enrichment(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        stats: Dict[str, object] = {}
        row_count = len(df)
        if row_count < 5:
            return df, stats

        score_cols = [
            col
            for col in ["VGChartz_Score", "Critic_Score", "User_Score"]
            if col in df.columns
        ]
        helper_cols = [col for col in ["Global_Sales", "Year"] if col in df.columns]

        if score_cols and helper_cols and df[score_cols].isna().any().any():
            feature_columns = score_cols + helper_cols
            n_neighbors = min(8, max(2, row_count - 1))
            imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
            imputed = imputer.fit_transform(df[feature_columns])
            df.loc[:, score_cols] = imputed[:, : len(score_cols)]
            for col in score_cols:
                df[col] = df[col].clip(lower=0).round(2)
            stats["knn_imputer"] = {
                "features": feature_columns,
                "n_neighbors": n_neighbors,
            }

        region_cols = [col for col in REGION_COLS if col in df.columns]
        if region_cols and row_count >= 20:
            share = df[region_cols].copy()
            total = share.sum(axis=1).replace(0, np.nan)
            share = share.divide(total, axis=0).fillna(0)
            unique_rows = share.drop_duplicates()
            if len(unique_rows) >= 2:
                if row_count >= 400:
                    cluster_count = 4
                elif row_count >= 150:
                    cluster_count = 3
                else:
                    cluster_count = 2
                cluster_count = min(cluster_count, len(unique_rows))
                if cluster_count >= 2:
                    scaler = StandardScaler()
                    scaled = scaler.fit_transform(share)
                    kmeans = KMeans(
                        n_clusters=cluster_count,
                        n_init=25,
                        random_state=self.random_state,
                    )
                    labels = kmeans.fit_predict(scaled)
                    df["Regional_Cluster"] = labels
                    centers = pd.DataFrame(kmeans.cluster_centers_, columns=region_cols)
                    label_map: Dict[int, str] = {}
                    for idx, center in centers.iterrows():
                        top_region = REGION_LABELS.get(center.idxmax(), center.idxmax())
                        label_map[idx] = (
                            f"{top_region}主导" if top_region else f"Cluster {idx}"
                        )
                    df["Regional_Cluster_Label"] = df["Regional_Cluster"].map(label_map)
                    stats["regional_clustering"] = {
                        "n_clusters": cluster_count,
                        "label_map": label_map,
                    }
        return df, stats

    def _parse_date_column(self, series: pd.Series) -> pd.Series:
        if series.empty:
            return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

        normalized = series.astype("string").str.strip()
        normalized = normalized.replace(
            {"": pd.NA, "nan": pd.NA, "None": pd.NA, "N/A": pd.NA}
        )
        normalized = normalized.str.replace(
            r"(?i)(\d{1,2})(st|nd|rd|th)", r"\1", regex=True
        )
        normalized = normalized.str.replace(",", "", regex=False)

        parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        date_formats = [
            "%d %b %Y",
            "%d %b %y",
            "%d %B %Y",
            "%d %B %y",
            "%Y-%m-%d",
            "%d-%b-%Y",
        ]

        for fmt in date_formats:
            pending = parsed.isna()
            if not pending.any():
                break
            parsed_values = pd.to_datetime(
                normalized[pending], format=fmt, errors="coerce"
            )
            parsed.loc[pending] = parsed_values

        pending = parsed.isna()
        if pending.any():
            parsed.loc[pending] = pd.to_datetime(normalized[pending], errors="coerce")

        return parsed

    @staticmethod
    def _safe_ratio(numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0
        return float(numerator) / float(denominator)
