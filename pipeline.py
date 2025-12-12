import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dashboard import DashboardBuilder
from database import PostgresConfig, PostgresReader
from ml import MLArtifacts, SalesMLAnalyzer
from plots import FigureGenerator
from settings import (
    DASHBOARD_CONFIG,
    LOGGER,
    OutputArtifacts,
    REGION_COLS,
    REGION_LABELS,
    build_artifacts,
)


class VGSalesPipeline:
    def __init__(
        self,
        data_file: Path,
        output_dir: Path,
        *,
        artifacts: Optional[OutputArtifacts] = None,
        dashboard_builder: Optional[DashboardBuilder] = None,
        figure_generator: Optional[FigureGenerator] = None,
        ml_analyzer: Optional[SalesMLAnalyzer] = None,
        db_config: Optional[PostgresConfig] = None,
    ) -> None:
        self.data_file = data_file
        self.artifacts = artifacts or build_artifacts(output_dir)
        self.output_dir = self.artifacts.directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create gallery directory
        self.gallery_dir = self.output_dir / "gallery"
        self.gallery_dir.mkdir(exist_ok=True)

        self.dashboard_builder = dashboard_builder or DashboardBuilder(
            self.artifacts.dashboard_html,
            config=DASHBOARD_CONFIG,
        )
        # Pass gallery_dir to FigureGenerator
        self.figure_generator = figure_generator or FigureGenerator(self.gallery_dir)
        default_ml_artifacts = MLArtifacts(
            metrics_json=self.artifacts.ml_metrics_json,
            clusters_json=self.artifacts.ml_clusters_json,
            predictions_json=self.artifacts.ml_predictions_json,
            feature_importance_png=self.artifacts.ml_feature_png,
        )
        self.ml_analyzer = ml_analyzer or SalesMLAnalyzer(default_ml_artifacts)
        self.db_reader = PostgresReader(db_config or PostgresConfig())
        self.random_state = 42
        LOGGER.info("输出目录：%s", self.output_dir.resolve())

    def run(self) -> None:
        LOGGER.info("开始执行数据分析管线")
        df = self._load_data()
        LOGGER.info("输入已清洗数据：%d 行 %d 列", df.shape[0], df.shape[1])
        metrics = self._compute_summary_metrics(df)
        ml_results = self._run_ml_analysis(df)
        if ml_results:
            metrics["ml"] = ml_results
        self._write_metrics_json(metrics)
        self._write_summary(metrics)
        figure_paths = self.figure_generator.generate_all(df)

        # 检查 SHAP 蜂群图是否存在，移动到 gallery 并添加到图表列表
        shap_summary_path = self.output_dir / "shap_summary.png"
        if shap_summary_path.exists():
            new_shap_path = self.gallery_dir / "shap_summary.png"
            # 如果目标文件存在，先删除
            if new_shap_path.exists():
                new_shap_path.unlink()
            shap_summary_path.rename(new_shap_path)
            figure_paths.append(new_shap_path)
            LOGGER.info("已添加 SHAP 蜂群图到画廊: %s", new_shap_path.name)

        self._generate_interactive_dashboard(df, metrics, figure_paths)
        LOGGER.info("流程完成，共生成 %d 张静态图表", len(figure_paths))

    def _load_data(self) -> pd.DataFrame:
        try:
            df = self.db_reader.read()
            df.columns = df.columns.map(str)
            LOGGER.info("已从数据库读取清洗数据：%d 行 %d 列", df.shape[0], df.shape[1])
            return df
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("数据库读取失败，回退至 CSV：%s", exc)

        if self.data_file and self.data_file.exists():
            df = pd.read_csv(self.data_file)
            df.columns = df.columns.map(str)
            LOGGER.info("回退读取 CSV：%d 行 %d 列", df.shape[0], df.shape[1])
            return df

        raise FileNotFoundError("无法从数据库或 CSV 读取清洗数据")

    def _compute_summary_metrics(self, df: pd.DataFrame) -> Dict[str, object]:
        region_totals = df[REGION_COLS].sum()
        region_share = (
            (region_totals / region_totals.sum()).fillna(0).to_dict()
            if region_totals.sum() > 0
            else {col: 0.0 for col in REGION_COLS}
        )
        genre_sales = (
            df.groupby("Genre")["Global_Sales"].sum().sort_values(ascending=False)
        )
        publisher_sales = (
            df.groupby("Publisher")["Global_Sales"].sum().sort_values(ascending=False)
        )
        platform_sales = (
            df.groupby("Platform_Family_CN")["Global_Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        metrics: Dict[str, object] = {
            "row_count": int(len(df)),
            "time_span": (
                int(df["Year"].min()),
                int(df["Year"].max()),
            ),
            "global_sales_total": float(df["Global_Sales"].sum()),
            "region_share": {
                REGION_LABELS[col]: float(region_share.get(col, 0.0))
                for col in REGION_COLS
            },
            "top_genres": list(genre_sales.head(10).round(2).items()),
            "top_publishers": list(publisher_sales.head(10).round(2).items()),
            "top_platforms": list(platform_sales.head(10).round(2).items()),
        }

        metrics["time_series"] = self._compute_temporal_metrics(df)
        metrics["tier_contributions"] = self._compute_tier_contributions(df)
        metrics["platform_lifecycle"] = self._compute_platform_lifecycle(df)
        metrics["regional_resilience"] = self._compute_regional_resilience(df)
        metrics["quality_gap"] = self._compute_quality_gap(df)
        metrics["cluster_insights"] = self._summarize_clusters(df)
        metrics.update(self._compute_innovation_metrics(df))
        return metrics

    def _compute_innovation_metrics(self, df: pd.DataFrame) -> Dict[str, object]:
        region_totals = df[REGION_COLS].sum()
        genre_sales = (
            df.groupby("Genre")["Global_Sales"].sum().sort_values(ascending=False)
        )
        platform_sales = (
            df.groupby("Platform_Family")["Global_Sales"]
            .sum()
            .sort_values(ascending=False)
        )
        global_sales_total = df["Global_Sales"].sum()

        genre_region = df.groupby("Genre")[REGION_COLS].sum()
        genre_region_pct = genre_region.divide(genre_region.sum(axis=1), axis=0).fillna(
            0
        )
        global_region_pct = region_totals / region_totals.sum()
        preference_matrix = genre_region_pct.subtract(global_region_pct, axis=1)
        preference_records = [
            {
                "Genre": genre,
                "Region": region,
                "Preference": float(preference_matrix.loc[genre, region]),
            }
            for genre in preference_matrix.index
            for region in preference_matrix.columns
        ]
        preference_records.sort(key=lambda r: r["Preference"], reverse=True)

        platform_diversity = self._shannon_entropy(platform_sales.values)
        publisher_stats = df.groupby("Publisher").agg(
            global_sales=("Global_Sales", "sum"), decades=("Decade", "nunique")
        )
        if publisher_stats.empty:
            moat = []
        else:
            publisher_stats["sales_norm"] = (
                publisher_stats["global_sales"] / publisher_stats["global_sales"].max()
            )
            publisher_stats["decade_norm"] = (
                publisher_stats["decades"] / publisher_stats["decades"].max()
            )
            publisher_stats["moat_score"] = (
                0.7 * publisher_stats["sales_norm"]
                + 0.3 * publisher_stats["decade_norm"]
            )
            moat = (
                publisher_stats.sort_values("moat_score", ascending=False)
                .head(5)
                .reset_index()[["Publisher", "moat_score"]]
                .to_dict("records")
            )

        gini = self._gini_coefficient(df["Global_Sales"].values)

        return {
            "innovation": {
                "region_share": region_totals.to_dict(),
                "top_genres": list(genre_sales.head(10).items()),
                "platform_share": [
                    (name, value / global_sales_total if global_sales_total else 0.0)
                    for name, value in platform_sales.items()
                ],
                "region_preference": preference_records,
                "platform_diversity": platform_diversity,
                "publisher_moat": moat,
                "gini": gini,
            }
        }

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return float(numerator) / float(denominator)

    def _compute_temporal_metrics(self, df: pd.DataFrame) -> Dict[str, object]:
        if "Year" not in df.columns or df["Year"].empty:
            return {}
        yearly = (
            df.groupby("Year")
            .agg(sales=("Global_Sales", "sum"), releases=("Name", "count"))
            .sort_index()
        )
        yearly_sales = yearly["sales"].replace(0, np.nan)
        yoy = yearly_sales.pct_change().replace([np.inf, -np.inf], np.nan)
        cagr = 0.0
        if len(yearly_sales.dropna()) >= 2:
            first = yearly_sales.dropna().iloc[0]
            last = yearly_sales.dropna().iloc[-1]
            periods = len(yearly_sales.dropna()) - 1
            if first > 0 and periods > 0:
                cagr = float((last / first) ** (1 / periods) - 1)
        volatility = float(yoy.dropna().std()) if yoy.dropna().size else 0.0
        boom = [
            {"year": int(year), "yoy": float(value)}
            for year, value in yoy.nlargest(3).dropna().items()
        ]
        bust = [
            {"year": int(year), "yoy": float(value)}
            for year, value in yoy.nsmallest(3).dropna().items()
        ]
        recent_density = (
            float(yearly["releases"].tail(5).mean()) if not yearly.empty else 0.0
        )
        return {
            "yearly_sales": yearly["sales"].round(2).to_dict(),
            "yearly_releases": yearly["releases"].astype(int).to_dict(),
            "cagr": cagr,
            "volatility": volatility,
            "boom_periods": boom,
            "bust_periods": bust,
            "recent_release_density": recent_density,
        }

    def _compute_tier_contributions(self, df: pd.DataFrame) -> Dict[str, object]:
        region_cols = [col for col in REGION_COLS if col in df.columns]
        result = {
            "sales_by_tier": [],
            "score_sales_matrix": [],
            "tier_lift": [],
        }
        if "Sales_Tier" not in df.columns:
            return result

        tier_group = (
            df.dropna(subset=["Sales_Tier"])
            .groupby("Sales_Tier")[["Global_Sales"] + region_cols]
            .sum()
        )
        for tier, row in tier_group.iterrows():
            regional_total = row[region_cols].sum() if region_cols else 0.0
            region_share = {
                REGION_LABELS.get(col, col): self._safe_divide(row[col], regional_total)
                for col in region_cols
            }
            result["sales_by_tier"].append(
                {
                    "tier": str(tier),
                    "global_sales": float(row["Global_Sales"]),
                    "region_share": region_share,
                }
            )

        overall_mean = df["Global_Sales"].mean()
        tier_mean = (
            df.dropna(subset=["Sales_Tier"])
            .groupby("Sales_Tier")["Global_Sales"]
            .mean()
            .round(4)
        )
        for tier, value in tier_mean.items():
            result["tier_lift"].append(
                {
                    "tier": str(tier),
                    "avg_sales": float(value),
                    "lift": self._safe_divide(value, overall_mean),
                }
            )

        if "Score_Tier" in df.columns:
            matrix = (
                df.dropna(subset=["Sales_Tier", "Score_Tier"])
                .groupby(["Score_Tier", "Sales_Tier"])["Global_Sales"]
                .sum()
                .round(2)
            )
            result["score_sales_matrix"] = [
                {
                    "score_tier": str(score),
                    "sales_tier": str(sales_tier),
                    "global_sales": float(value),
                }
                for (score, sales_tier), value in matrix.items()
            ]

        return result

    def _compute_platform_lifecycle(self, df: pd.DataFrame) -> List[Dict[str, object]]:
        if "Platform_Family_CN" not in df.columns or "Year" not in df.columns:
            return []
        platform_year = (
            df.groupby(["Platform_Family_CN", "Year"])["Global_Sales"]
            .sum()
            .sort_index()
        )
        lifecycle: List[Dict[str, object]] = []
        for platform, series in platform_year.groupby(level=0):
            yearly = series.droplevel(0)
            if yearly.empty:
                continue
            max_year = int(yearly.index.max())
            recent_start = max(max_year - 2, int(yearly.index.min()))
            prior_end = recent_start - 1
            prior_start = prior_end - 2
            recent_sum = yearly[yearly.index >= recent_start].sum()
            prior_mask = (yearly.index >= prior_start) & (yearly.index <= prior_end)
            prior_sum = yearly[prior_mask].sum()
            total_sales = yearly.sum()
            retention_window_start = max(max_year - 4, int(yearly.index.min()))
            retention_share = self._safe_divide(
                yearly[yearly.index >= retention_window_start].sum(),
                total_sales,
            )
            lifecycle.append(
                {
                    "platform": platform,
                    "year_span": (int(yearly.index.min()), max_year),
                    "momentum": self._safe_divide(recent_sum, prior_sum + 1e-6),
                    "recent_sales": float(recent_sum),
                    "total_sales": float(total_sales),
                    "retention_share": retention_share,
                }
            )
        lifecycle.sort(key=lambda item: item["momentum"], reverse=True)
        return lifecycle

    def _compute_regional_resilience(self, df: pd.DataFrame) -> Dict[str, object]:
        region_cols = [col for col in REGION_COLS if col in df.columns]
        if "Platform_Family_CN" not in df.columns or not region_cols:
            return {}
        platform_regions = (
            df.groupby("Platform_Family_CN")[region_cols].sum().replace(0, np.nan)
        )
        records: List[Dict[str, object]] = []
        for platform, row in platform_regions.iterrows():
            total = row.sum()
            if total == 0 or pd.isna(total):
                continue
            shares = (row / total).fillna(0)
            hhi = float((shares**2).sum())
            entropy = float(
                -(shares * np.log2(shares.replace(0, np.nan))).sum(skipna=True)
            )
            dominant_region = REGION_LABELS.get(shares.idxmax(), shares.idxmax())
            records.append(
                {
                    "platform": platform,
                    "hhi": hhi,
                    "entropy": entropy,
                    "dominant_region": dominant_region,
                }
            )
        diversified = sorted(records, key=lambda item: item["hhi"])[:3]
        concentrated = sorted(records, key=lambda item: item["hhi"], reverse=True)[:3]
        return {
            "platform_resilience": records,
            "most_diversified": diversified,
            "most_concentrated": concentrated,
        }

    def _compute_quality_gap(self, df: pd.DataFrame) -> Dict[str, object]:
        if "Composite_Score" not in df.columns:
            return {}
        composite = df["Composite_Score"].dropna()
        if composite.empty:
            return {}
        quartiles = composite.quantile([0.25, 0.5, 0.75]).round(2).to_dict()
        correlation = float(df["Composite_Score"].corr(df["Global_Sales"]))
        gap_summary = {}
        disagreements: List[Dict[str, object]] = []
        if "Score_Gap" in df.columns:
            gap = df["Score_Gap"].dropna()
            if not gap.empty:
                gap_summary = {
                    "mean_gap": float(gap.mean()),
                    "std_gap": float(gap.std()),
                    "share_positive": float((gap > 0).mean()),
                }
                top_gap = df.dropna(subset=["Score_Gap"]).copy()
                top_gap["abs_gap"] = top_gap["Score_Gap"].abs()
                disagreements = [
                    {
                        "name": row.get("Name"),
                        "platform": row.get("Platform_Family_CN"),
                        "score_gap": float(row.get("Score_Gap", 0.0)),
                        "global_sales": float(row.get("Global_Sales", 0.0)),
                    }
                    for _, row in top_gap.nlargest(5, "abs_gap").iterrows()
                ]
        return {
            "quartiles": quartiles,
            "correlation": correlation,
            "score_gap": gap_summary,
            "largest_disagreements": disagreements,
        }

    def _summarize_clusters(self, df: pd.DataFrame) -> List[Dict[str, object]]:
        label_col = None
        if "Regional_Cluster_Label" in df.columns:
            label_col = "Regional_Cluster_Label"
        elif "Regional_Cluster" in df.columns:
            label_col = "Regional_Cluster"
        if label_col is None:
            return []
        clusters: List[Dict[str, object]] = []
        for label, group in df.groupby(label_col):
            clusters.append(
                {
                    "label": str(label) if label is not None else "未知",
                    "size": int(len(group)),
                    "avg_sales": float(group["Global_Sales"].mean()),
                    "top_genres": group["Genre"].value_counts().head(3).index.tolist(),
                    "top_platforms": group["Platform_Family_CN"]
                    .value_counts()
                    .head(3)
                    .index.tolist(),
                }
            )
        clusters.sort(key=lambda item: item["avg_sales"], reverse=True)
        return clusters

    @staticmethod
    def _shannon_entropy(values: np.ndarray) -> float:
        total = values.sum()
        if total == 0:
            return 0.0
        probs = values / total
        probs = probs[probs > 0]
        return float(-(probs * np.log2(probs)).sum())

    @staticmethod
    def _gini_coefficient(values: np.ndarray) -> float:
        values = np.sort(values[values >= 0])
        if len(values) == 0:
            return 0.0
        cumulative = np.cumsum(values)
        sum_values = cumulative[-1]
        if sum_values == 0:
            return 0.0
        gini = (len(values) + 1 - 2 * np.sum(cumulative) / sum_values) / len(values)
        return float(gini)

    def _write_metrics_json(self, metrics: Dict[str, object]) -> None:
        self.artifacts.metrics_json.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info("指标字典已写入 %s", self.artifacts.metrics_json.name)

    def _write_summary(self, metrics: Dict[str, object]) -> None:
        lines = [
            "# 全球电子游戏销量分析简报",
            "",
            f"- 数据记录数：{metrics['row_count']:,}",
            f"- 时间范围：{metrics['time_span'][0]} - {metrics['time_span'][1]}",
            f"- 累计销量：{metrics['global_sales_total']:.2f} 百万套",
            f"- 销量基尼系数：{metrics['innovation']['gini']:.3f}",
            "",
            "## 区域销量占比",
        ]
        for region, share in metrics["region_share"].items():
            lines.append(f"- {region}：{share:.2%}")

        time_series = metrics.get("time_series", {})
        if time_series:
            lines.extend(
                [
                    "",
                    "## 市场动能",
                    f"- CAGR：{time_series.get('cagr', 0.0):.2%}",
                    f"- 波动率：{time_series.get('volatility', 0.0):.2%}",
                    f"- 近五年年度发布密度：{time_series.get('recent_release_density', 0.0):.1f} 款/年",
                ]
            )
            boom = time_series.get("boom_periods", [])
            if boom:
                boom_desc = ", ".join(
                    f"{item['year']}: {item['yoy']:.1%}" for item in boom
                )
                lines.append(f"- 高光年份：{boom_desc}")

        tier_info = metrics.get("tier_contributions", {})
        if tier_info.get("tier_lift"):
            lines.extend(["", "## 销量分层表现"])
            for entry in tier_info["tier_lift"][:5]:
                lines.append(
                    f"- {entry['tier']}：平均 {entry['avg_sales']:.2f} 百万套，lift {entry['lift']:.2f}"
                )

        lines.extend(["", "## 热门游戏类型（前五）"])
        for genre, value in metrics["top_genres"][:5]:
            lines.append(f"- {genre}：{value:.2f} 百万套")

        lines.extend(["", "## 发行商综合竞争力（前五）"])
        for entry in metrics["innovation"]["publisher_moat"]:
            lines.append(
                f"- {entry['Publisher']}：竞争力得分 {entry['moat_score']:.2f}"
            )

        quality = metrics.get("quality_gap", {})
        if quality:
            lines.extend(["", "## 媒体 VS 玩家评分差异"])
            lines.append(f"- 评分与销量相关性：{quality.get('correlation', 0.0):.2f}")
            gap = quality.get("score_gap", {})
            if gap:
                lines.append(
                    f"- 平均评分差：{gap.get('mean_gap', 0.0):.2f}，标准差 {gap.get('std_gap', 0.0):.2f}"
                )

        clusters = metrics.get("cluster_insights", [])
        if clusters:
            lines.extend(["", "## 区域偏好集群（Top3）"])
            for cluster in clusters[:3]:
                lines.append(
                    f"- {cluster['label']}：样本 {cluster['size']}，均值 {cluster['avg_sales']:.2f} 百万套"
                )

        self.artifacts.summary_md.write_text("\n".join(lines), encoding="utf-8")
        LOGGER.info("Markdown 摘要已写入 %s", self.artifacts.summary_md.name)

    def _generate_interactive_dashboard(
        self,
        df: pd.DataFrame,
        metrics: Dict[str, object],
        figure_paths: List[Path] = None,
    ) -> None:
        if self.dashboard_builder is None:
            LOGGER.info("未配置仪表盘构建器，跳过交互式输出")
            return
        self.dashboard_builder.build(df, metrics, figure_paths)

    def _run_ml_analysis(self, df: pd.DataFrame) -> Dict[str, object]:
        if self.ml_analyzer is None:
            LOGGER.info("未配置机器学习分析器，跳过该步骤")
            return {}
        LOGGER.info("启动机器学习建模与聚类分析阶段")
        return self.ml_analyzer.run(df)
