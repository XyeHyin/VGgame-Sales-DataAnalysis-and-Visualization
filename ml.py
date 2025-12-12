import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from settings import LOGGER, REGION_COLS, REGION_LABELS


@dataclass(frozen=True)
class MLArtifacts:
    metrics_json: Path
    clusters_json: Path
    predictions_json: Path
    feature_importance_png: Path


class SalesMLAnalyzer:
    def __init__(
        self,
        artifacts: MLArtifacts,
        *,
        hit_threshold: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.artifacts = artifacts
        self.hit_threshold = hit_threshold
        self.random_state = random_state
        self.numeric_features = ["Year", "Decade"]
        self.categorical_features = [
            "Platform_Family",
            "Genre",
            "Publisher_Grouped",
        ]
        self._ensure_directories()

    def run(self, df: pd.DataFrame) -> Dict[str, object]:
        if df.empty:
            LOGGER.warning("ML 模块：输入数据为空，跳过建模阶段")
            return {}

        work = self._prepare_features(df)
        regression = self._train_regression(work)
        classification = self._train_classification(work)
        clustering = self._cluster_regions(work)
        behavior_clusters = self._cluster_behavior_segments(work)
        similarity = self._build_similarity_samples(work)

        summary = {
            "regression": regression["metrics"] if regression else None,
            "regression_quantiles": (
                regression.get("quantiles", {}).get("bands") if regression else None
            ),
            "regression_alternative": (
                regression.get("alternative_model") if regression else None
            ),
            "classification": classification["metrics"] if classification else None,
            "calibration": (
                classification.get("calibration") if classification else None
            ),
            "clustering": clustering["segments"][:3] if clustering else [],
            "behavior_clusters": (
                behavior_clusters["segments"][:3] if behavior_clusters else []
            ),
            "top_features": (
                regression["feature_importance"][:5] if regression else []
            ),
            "permutation_features": (
                regression.get("permutation_importance", [])[:5] if regression else []
            ),
            "similar_titles": similarity[:5] if similarity else [],
        }

        self._persist_artifacts(
            regression,
            classification,
            clustering,
            behavior_clusters,
            similarity,
        )
        return summary

    def _ensure_directories(self) -> None:
        paths = [
            self.artifacts.metrics_json,
            self.artifacts.clusters_json,
            self.artifacts.predictions_json,
            self.artifacts.feature_importance_png,
        ]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        # 将低频发行商合并到 OtherPublisher，避免稀疏矩阵膨胀
        top_publishers = work["Publisher"].value_counts().head(30).index
        work["Publisher_Grouped"] = work["Publisher"].where(
            work["Publisher"].isin(top_publishers), "OtherPublisher"
        )
        work["Publisher_Grouped"] = work["Publisher_Grouped"].fillna("OtherPublisher")
        work["Genre"] = work["Genre"].fillna("其他")
        work["Platform_Family"] = work["Platform_Family"].fillna("Other")
        work["Decade"] = work["Decade"].fillna(work["Decade"].median())
        work["Year"] = work["Year"].fillna(work["Year"].median())
        return work

    def _build_regression_pipeline(self, estimator) -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            (
                                "num",
                                Pipeline(
                                    steps=[
                                        ("imputer", SimpleImputer(strategy="median")),
                                        ("scaler", StandardScaler()),
                                    ]
                                ),
                                self.numeric_features,
                            ),
                            (
                                "cat",
                                Pipeline(
                                    steps=[
                                        (
                                            "imputer",
                                            SimpleImputer(strategy="most_frequent"),
                                        ),
                                        (
                                            "encoder",
                                            OneHotEncoder(
                                                handle_unknown="ignore",
                                                sparse_output=False,
                                            ),
                                        ),
                                    ]
                                ),
                                self.categorical_features,
                            ),
                        ]
                    ),
                ),
                ("model", estimator),
            ]
        )

    def _train_regression(self, df: pd.DataFrame) -> Optional[Dict[str, object]]:
        target = df["Global_Sales"]
        if target.nunique() <= 1 or len(df) < 50:
            LOGGER.warning("ML 模块：样本不足或目标列缺乏变化，跳过回归建模")
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            df[self.numeric_features + self.categorical_features],
            target,
            test_size=0.2,
            random_state=self.random_state,
        )

        pipeline = self._build_regression_pipeline(
            RandomForestRegressor(
                n_estimators=400,
                max_depth=14,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
            )
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = {
            "model": "RandomForestRegressor",
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "r2": float(r2_score(y_test, predictions)),
        }

        feature_importance = self._collect_feature_importance(pipeline)
        permutation_features = self._permutation_importance(pipeline, X_test, y_test)
        quantiles = self._train_quantile_models(X_train, y_train, X_test, y_test)
        alternative_model = self._train_hist_gradient_regressor(
            X_train, X_test, y_train, y_test
        )
        samples = self._build_regression_samples(
            df.loc[y_test.index],
            predictions,
            quantiles.get("per_sample") if quantiles else None,
        )
        self._plot_feature_importance(feature_importance)
        result: Dict[str, object] = {
            "metrics": metrics,
            "feature_importance": feature_importance,
            "samples": samples,
        }
        if permutation_features:
            result["permutation_importance"] = permutation_features
        if quantiles:
            result["quantiles"] = quantiles
        if alternative_model:
            result["alternative_model"] = alternative_model
        return result

    def _train_classification(self, df: pd.DataFrame) -> Optional[Dict[str, object]]:
        labels = (df["Global_Sales"] >= self.hit_threshold).astype(int)
        if labels.nunique() <= 1:
            LOGGER.warning("ML 模块：没有足够的命中样本，跳过分类建模")
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            df[self.numeric_features + self.categorical_features],
            labels,
            test_size=0.2,
            random_state=self.random_state,
            stratify=labels,
        )

        base_pipeline = Pipeline(
            steps=[
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            (
                                "num",
                                Pipeline(
                                    steps=[
                                        ("imputer", SimpleImputer(strategy="median")),
                                        ("scaler", StandardScaler()),
                                    ]
                                ),
                                self.numeric_features,
                            ),
                            (
                                "cat",
                                Pipeline(
                                    steps=[
                                        (
                                            "imputer",
                                            SimpleImputer(strategy="most_frequent"),
                                        ),
                                        (
                                            "encoder",
                                            OneHotEncoder(
                                                handle_unknown="ignore",
                                                sparse_output=False,
                                            ),
                                        ),
                                    ]
                                ),
                                self.categorical_features,
                            ),
                        ]
                    ),
                ),
                (
                    "model",
                    GradientBoostingClassifier(random_state=self.random_state),
                ),
            ]
        )

        calibrator_kwargs = {"method": "isotonic", "cv": 3}
        try:
            calibrated = CalibratedClassifierCV(
                estimator=base_pipeline,
                **calibrator_kwargs,
            )
        except TypeError:
            # Older sklearn versions use base_estimator instead of estimator
            calibrated = CalibratedClassifierCV(
                base_estimator=base_pipeline,
                **calibrator_kwargs,
            )
        calibrated.fit(X_train, y_train)
        predictions = calibrated.predict(X_test)
        probabilities = calibrated.predict_proba(X_test)[:, 1]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="binary", zero_division=0
        )
        metrics = {
            "model": "GradientBoostingClassifier",
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "hit_threshold": self.hit_threshold,
            "support": int(len(y_test)),
            "brier": float(brier_score_loss(y_test, probabilities)),
        }
        report = classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        )
        metrics["classification_report"] = report
        frac_pos, mean_pred = calibration_curve(y_test, probabilities, n_bins=10)
        calibration_points = [
            {"predicted": float(p), "observed": float(o)}
            for p, o in zip(mean_pred, frac_pos)
        ]
        samples = self._build_classification_samples(
            df.loc[y_test.index], probabilities, predictions
        )
        return {
            "metrics": metrics,
            "samples": samples,
            "calibration": {"curve": calibration_points},
        }

    def _cluster_regions(self, df: pd.DataFrame) -> Optional[Dict[str, object]]:
        if not REGION_COLS:
            return None
        region_cols = [col for col in REGION_COLS if col in df.columns]
        if not region_cols:
            return None
        share = df[region_cols].copy()
        share_sum = share.sum(axis=1).replace(0, np.nan)
        share = share.divide(share_sum, axis=0).fillna(0)
        if len(share) < 10:
            LOGGER.warning("ML 模块：样本过少，聚类结果仅供参考")
        cluster_count = 4 if len(share) >= 4 else len(share)
        if cluster_count < 2:
            return None

        model = KMeans(
            n_clusters=cluster_count, n_init=20, random_state=self.random_state
        )
        labels = model.fit_predict(share)
        df_with_cluster = df.copy()
        df_with_cluster["cluster"] = labels

        segments: List[Dict[str, object]] = []
        for cluster_id, group in df_with_cluster.groupby("cluster"):
            centroid = share.loc[group.index].mean().to_dict()
            dominant_region = max(centroid, key=centroid.get)
            segments.append(
                {
                    "cluster": int(cluster_id),
                    "size": int(len(group)),
                    "avg_global_sales": float(group["Global_Sales"].mean()),
                    "top_genres": group["Genre"].value_counts().head(2).index.tolist(),
                    "top_platforms": group["Platform_Family"]
                    .value_counts()
                    .head(2)
                    .index.tolist(),
                    "dominant_region": REGION_LABELS.get(
                        dominant_region, dominant_region
                    ),
                    "region_profile": {
                        REGION_LABELS.get(col, col): round(float(val), 4)
                        for col, val in centroid.items()
                    },
                }
            )

        segments.sort(key=lambda seg: seg["avg_global_sales"], reverse=True)
        return {"segments": segments}

    def _build_similarity_samples(self, df: pd.DataFrame) -> List[Dict[str, object]]:
        region_cols = [col for col in REGION_COLS if col in df.columns]
        feature_cols = region_cols.copy()
        for extra in ["Composite_Score", "Age_Years", "Global_Sales"]:
            if extra in df.columns and extra not in feature_cols:
                feature_cols.append(extra)
        if len(feature_cols) < 3 or len(df) < 50:
            return []
        work = df[feature_cols].copy().fillna(0)
        scaler = StandardScaler()
        matrix = scaler.fit_transform(work)
        neighbors = NearestNeighbors(metric="cosine", n_neighbors=6)
        neighbors.fit(matrix)
        index_positions = {idx: pos for pos, idx in enumerate(df.index)}
        anchors = df.nlargest(10, "Global_Sales")
        recommendations: List[Dict[str, object]] = []
        for anchor_idx, anchor_row in anchors.iterrows():
            position = index_positions.get(anchor_idx)
            if position is None:
                continue
            distances, indices = neighbors.kneighbors(matrix[position].reshape(1, -1))
            for distance, neighbor_pos in zip(distances[0][1:], indices[0][1:]):
                neighbor_row = df.iloc[int(neighbor_pos)]
                recommendations.append(
                    {
                        "anchor": anchor_row.get("Name"),
                        "anchor_platform": anchor_row.get("Platform_Family"),
                        "candidate": neighbor_row.get("Name"),
                        "candidate_platform": neighbor_row.get("Platform_Family"),
                        "similarity": float(1 - distance),
                        "anchor_sales": float(anchor_row.get("Global_Sales", 0.0)),
                        "candidate_sales": float(neighbor_row.get("Global_Sales", 0.0)),
                    }
                )
            if len(recommendations) >= 30:
                break
        return recommendations

    def _cluster_behavior_segments(
        self, df: pd.DataFrame
    ) -> Optional[Dict[str, object]]:
        region_cols = [col for col in REGION_COLS if col in df.columns]
        feature_cols = region_cols.copy()
        for extra in ["Age_Years", "Composite_Score", "Global_Sales"]:
            if extra in df.columns and extra not in feature_cols:
                feature_cols.append(extra)
        if len(feature_cols) < 3 or len(df) < 60:
            return None
        work = df[feature_cols].copy().fillna(0)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(work)
        cluster_count = min(5, max(2, len(work) // 250))
        if cluster_count < 2:
            cluster_count = 2
        model = GaussianMixture(
            n_components=cluster_count, random_state=self.random_state
        )
        labels = model.fit_predict(scaled)
        df_copy = df.copy()
        df_copy["behavior_cluster"] = labels
        segments: List[Dict[str, object]] = []
        for cluster_id, group in df_copy.groupby("behavior_cluster"):
            region_profile = {}
            if region_cols:
                region_totals = group[region_cols].sum()
                total = region_totals.sum()
                if total > 0:
                    region_profile = {
                        REGION_LABELS.get(col, col): round(float(val / total), 4)
                        for col, val in region_totals.items()
                    }
            avg_age = 0.0
            if "Age_Years" in group and not group["Age_Years"].isna().all():
                avg_val = group["Age_Years"].mean()
                if not np.isnan(avg_val):
                    avg_age = float(avg_val)
            score_median = 0.0
            if "Composite_Score" in group and not group["Composite_Score"].isna().all():
                med_val = group["Composite_Score"].median()
                if not np.isnan(med_val):
                    score_median = float(med_val)
            segments.append(
                {
                    "cluster": int(cluster_id),
                    "size": int(len(group)),
                    "avg_global_sales": float(group["Global_Sales"].mean()),
                    "avg_age": avg_age,
                    "score_median": score_median,
                    "top_genres": group["Genre"].value_counts().head(3).index.tolist(),
                    "top_platforms": group["Platform_Family"]
                    .value_counts()
                    .head(2)
                    .index.tolist(),
                    "region_profile": region_profile,
                }
            )
        segments.sort(key=lambda seg: seg["avg_global_sales"], reverse=True)
        return {"segments": segments}

    def _collect_feature_importance(self, pipeline: Pipeline) -> List[Dict[str, float]]:
        model: RandomForestRegressor = pipeline.named_steps["model"]
        preprocess: ColumnTransformer = pipeline.named_steps["preprocess"]
        raw_names = preprocess.get_feature_names_out()
        importances = model.feature_importances_
        pairs = sorted(
            zip(raw_names, importances), key=lambda item: item[1], reverse=True
        )[:15]
        return [
            {"feature": self._prettify_feature_name(name), "importance": float(value)}
            for name, value in pairs
        ]

    def _permutation_importance(
        self, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
    ) -> List[Dict[str, float]]:
        try:
            preprocess: ColumnTransformer = pipeline.named_steps["preprocess"]
            model = pipeline.named_steps["model"]
        except KeyError:
            return []
        try:
            transformed = preprocess.transform(X_test)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Permutation importance 预处理失败：%s", exc)
            return []
        result = permutation_importance(
            model,
            transformed,
            y_test,
            n_repeats=8,
            random_state=self.random_state,
            n_jobs=-1,
        )
        feature_names = preprocess.get_feature_names_out()
        pairs = sorted(
            zip(feature_names, result.importances_mean),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
        return [
            {"feature": self._prettify_feature_name(name), "importance": float(value)}
            for name, value in pairs
        ]

    def _train_quantile_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, object]:
        alphas = [0.1, 0.5, 0.9]
        predictions: Dict[str, np.ndarray] = {}
        for alpha in alphas:
            estimator = GradientBoostingRegressor(
                loss="quantile",
                alpha=alpha,
                random_state=self.random_state,
            )
            pipeline = self._build_regression_pipeline(estimator)
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            predictions[f"p{int(alpha * 100)}"] = preds
        if not predictions:
            return {}
        per_sample: Dict[int, Dict[str, float]] = {}
        for idx_pos, idx in enumerate(y_test.index):
            per_sample[idx] = {
                key: float(pred[idx_pos]) for key, pred in predictions.items()
            }
        bands = {
            f"{key}_mean": float(np.mean(values)) for key, values in predictions.items()
        }
        bands.update(
            {
                f"{key}_median": float(np.median(values))
                for key, values in predictions.items()
            }
        )
        return {"per_sample": per_sample, "bands": bands}

    def _train_hist_gradient_regressor(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Optional[Dict[str, object]]:
        if len(X_train) < 50:
            return None
        estimator = HistGradientBoostingRegressor(
            max_depth=12,
            learning_rate=0.08,
            random_state=self.random_state,
        )
        pipeline = self._build_regression_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        return {
            "model": "HistGradientBoostingRegressor",
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "r2": float(r2_score(y_test, preds)),
        }

    def _prettify_feature_name(self, raw_name: str) -> str:
        name = raw_name.split("__", maxsplit=1)[-1]
        replacements = {
            "Platform_Family_": "平台:",
            "Genre_": "类型:",
            "Publisher_Grouped_": "发行商:",
        }
        for prefix, label in replacements.items():
            if name.startswith(prefix):
                return label + name[len(prefix) :]
        return name

    def _build_regression_samples(
        self,
        reference: pd.DataFrame,
        predictions: np.ndarray,
        quantiles: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> List[Dict[str, object]]:
        records = []
        for idx, pred in zip(reference.index, predictions):
            row = reference.loc[idx]
            entry = {
                "Name": row.get("Name"),
                "Platform": row.get("Platform_Family"),
                "Genre": row.get("Genre"),
                "Year": int(row.get("Year", 0)),
                "actual_sales": float(row.get("Global_Sales", 0.0)),
                "predicted_sales": float(pred),
            }
            if quantiles and idx in quantiles:
                entry.update(quantiles[idx])
            records.append(entry)
        return records[:25]

    def _build_classification_samples(
        self,
        reference: pd.DataFrame,
        probabilities: np.ndarray,
        predictions: np.ndarray,
    ) -> List[Dict[str, object]]:
        df_samples = reference.copy()
        df_samples["hit_probability"] = probabilities
        df_samples["predicted_label"] = predictions
        df_samples.sort_values("hit_probability", ascending=False, inplace=True)
        top_rows = df_samples.head(25)
        return [
            {
                "Name": row.get("Name"),
                "Platform": row.get("Platform_Family"),
                "Genre": row.get("Genre"),
                "Year": int(row.get("Year", 0)),
                "hit_probability": float(row.get("hit_probability", 0.0)),
                "predicted_label": int(row.get("predicted_label", 0)),
            }
            for _, row in top_rows.iterrows()
        ]

    def _plot_feature_importance(self, features: List[Dict[str, float]]) -> None:
        if not features:
            return
        top_features = features[:10]
        names = [item["feature"] for item in reversed(top_features)]
        values = [item["importance"] for item in reversed(top_features)]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(names, values, color="#00f3ff")
        ax.set_xlabel("重要度")
        ax.set_title("随机森林特征重要度（Top 10）")
        fig.tight_layout()
        fig.savefig(self.artifacts.feature_importance_png, dpi=300)
        plt.close(fig)

    def _persist_artifacts(
        self,
        regression: Optional[Dict[str, object]],
        classification: Optional[Dict[str, object]],
        clustering: Optional[Dict[str, object]],
        behavior_clusters: Optional[Dict[str, object]] = None,
        similarity: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        payload = {
            "regression": regression["metrics"] if regression else None,
            "regression_quantiles": (
                regression.get("quantiles", {}).get("bands") if regression else None
            ),
            "regression_alternative": (
                regression.get("alternative_model") if regression else None
            ),
            "classification": classification["metrics"] if classification else None,
            "calibration": (
                classification.get("calibration") if classification else None
            ),
            "clustering": clustering["segments"] if clustering else None,
            "behavior_clusters": (
                behavior_clusters["segments"] if behavior_clusters else None
            ),
            "top_features": (regression["feature_importance"] if regression else []),
            "permutation_features": (
                regression.get("permutation_importance") if regression else []
            ),
            "similar_titles": similarity or [],
        }
        self.artifacts.metrics_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        clusters_payload: List[Dict[str, object]] = []
        if clustering:
            clusters_payload.extend(
                [{"type": "regional", **segment} for segment in clustering["segments"]]
            )
        if behavior_clusters:
            clusters_payload.extend(
                [
                    {"type": "behavioral", **segment}
                    for segment in behavior_clusters["segments"]
                ]
            )
        self.artifacts.clusters_json.write_text(
            json.dumps(clusters_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        prediction_payload = {
            "regression_samples": regression["samples"] if regression else [],
            "classification_samples": (
                classification["samples"] if classification else []
            ),
            "similar_titles": similarity or [],
        }
        self.artifacts.predictions_json.write_text(
            json.dumps(prediction_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("机器学习指标、聚类与样本预测已写入 JSON，并输出特征重要度图")
