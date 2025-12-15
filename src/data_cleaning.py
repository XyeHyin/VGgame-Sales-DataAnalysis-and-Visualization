from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

from src.settings import (
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
            "MICE 缺失值插补：%s",
            "执行" if stats.get("iterative_imputer") else "跳过",
        )
        if stats.get("iterative_imputer"):
            LOGGER.info(
                "  - 使用特征：%s", ", ".join(stats["iterative_imputer"]["features"])
            )
            LOGGER.info("  - 算法：%s", stats["iterative_imputer"]["algorithm"])
        LOGGER.info(
            "异常检测 (Isolation Forest)：%s",
            "执行" if stats.get("isolation_forest") else "跳过",
        )
        if stats.get("isolation_forest"):
            LOGGER.info(
                "  - 检测到异常点：%d", stats["isolation_forest"]["outliers_found"]
            )
        LOGGER.info(
            "区域聚类：%s",
            "执行" if stats.get("regional_clustering") else "跳过",
        )
        if stats.get("regional_clustering"):
            LOGGER.info("  - 聚类数：%d", stats["regional_clustering"]["n_clusters"])
            LOGGER.info(
                "  - 算法：%s", stats["regional_clustering"].get("algorithm", "K-Means")
            )
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
            "outliers_detected": 0,
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
        df, outlier_stats = self._detect_outliers(df)
        stats.update(outlier_stats)

        return df, stats

    def _detect_outliers(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """使用 Isolation Forest 检测异常数据点"""
        stats: Dict[str, object] = {}

        if len(df) < 50:
            df["Is_Outlier"] = False
            df["Outlier_Score"] = 0.0
            return df, stats

        detection_cols = ["Global_Sales", "Year"]
        score_cols = [
            col for col in ["Critic_Score", "User_Score"] if col in df.columns
        ]
        detection_cols.extend(
            [col for col in score_cols if df[col].notna().sum() > len(df) * 0.3]
        )
        region_cols = [col for col in REGION_COLS if col in df.columns]
        detection_cols.extend(region_cols[:2])

        feature_df = df[detection_cols].copy()
        feature_df = feature_df.fillna(feature_df.median())

        if len(feature_df.columns) < 2:
            df["Is_Outlier"] = False
            df["Outlier_Score"] = 0.0
            return df, stats
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_df)
        iso_forest = IsolationForest(
            contamination=0.05, 
            random_state=self.random_state,
            n_jobs=-1,
        )
        outlier_labels = iso_forest.fit_predict(scaled_features)
        outlier_scores = iso_forest.decision_function(scaled_features)

        df["Is_Outlier"] = outlier_labels == -1
        df["Outlier_Score"] = outlier_scores

        n_outliers = int((outlier_labels == -1).sum())
        stats["outliers_detected"] = n_outliers
        stats["isolation_forest"] = {
            "features": detection_cols,
            "contamination": 0.05,
            "outliers_found": n_outliers,
            "outlier_ratio": round(n_outliers / len(df), 4) if len(df) > 0 else 0.0,
        }

        LOGGER.info(
            "Isolation Forest 检测到 %d 个异常点 (%.2f%%)",
            n_outliers,
            100 * n_outliers / len(df),
        )

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
            # IterativeImputer: 基于贝叶斯岭回归的多元链式插补
            imputer = IterativeImputer(
                max_iter=10,
                random_state=self.random_state,
                initial_strategy="median",
                skip_complete=True,
            )
            imputed = imputer.fit_transform(df[feature_columns])
            df.loc[:, score_cols] = imputed[:, : len(score_cols)]
            for col in score_cols:
                df[col] = df[col].clip(lower=0).round(2)
            stats["iterative_imputer"] = {
                "features": feature_columns,
                "max_iter": 10,
                "algorithm": "MICE (Multivariate Imputation by Chained Equations)",
            }
            LOGGER.info("MICE 插补完成，特征: %s", ", ".join(feature_columns))

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
                    # 使用 PCA 降维，便于可视化和加速聚类
                    pca = PCA(
                        n_components=min(2, len(region_cols)),
                        random_state=self.random_state,
                    )
                    pca_features = pca.fit_transform(scaled)
                    df["PCA_Region_1"] = pca_features[:, 0]
                    if pca_features.shape[1] > 1:
                        df["PCA_Region_2"] = pca_features[:, 1]
                    # K-Means++ 聚类
                    kmeans = KMeans(
                        n_clusters=cluster_count,
                        n_init=25,
                        random_state=self.random_state,
                        init="k-means++",
                    )
                    labels = kmeans.fit_predict(pca_features)
                    df["Regional_Cluster"] = labels
                    centers = pd.DataFrame(
                        kmeans.cluster_centers_,
                        columns=[f"PC{i+1}" for i in range(pca_features.shape[1])],
                    )
                    # 根据原始特征空间确定主导区域
                    original_centers = scaler.inverse_transform(
                        pca.inverse_transform(kmeans.cluster_centers_)
                    )
                    original_centers_df = pd.DataFrame(
                        original_centers, columns=region_cols
                    )
                    label_map: Dict[int, str] = {}
                    for idx, center in original_centers_df.iterrows():
                        top_region = REGION_LABELS.get(center.idxmax(), center.idxmax())
                        label_map[idx] = (
                            f"{top_region}主导" if top_region else f"Cluster {idx}"
                        )
                    df["Regional_Cluster_Label"] = df["Regional_Cluster"].map(label_map)
                    stats["regional_clustering"] = {
                        "n_clusters": cluster_count,
                        "label_map": label_map,
                        "algorithm": "K-Means++ with PCA",
                        "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
                    }
                    LOGGER.info(
                        "K-Means++ (with PCA) 区域聚类完成，聚类数: %d", cluster_count
                    )
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
