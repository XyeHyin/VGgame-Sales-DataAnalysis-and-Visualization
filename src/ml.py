import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
from sklearn.preprocessing import StandardScaler

from src.settings import LOGGER, REGION_COLS, REGION_LABELS


@dataclass(frozen=True)
class MLArtifacts:
    metrics_json: Path
    clusters_json: Path
    predictions_json: Path
    feature_importance_png: Path


class SalesMLAnalyzer:
    """
    使用 LightGBM 进行销量预测和爆款分类的 ML 分析器。
    """

    def __init__(
        self,
        artifacts: MLArtifacts,
        *,
        hit_threshold: float = 1.0,
        random_state: int = 230010202,
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
            "shap_features": (
                regression.get("shap_importance", [])[:5] if regression else []
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
        top_publishers = work["Publisher"].value_counts().head(30).index
        work["Publisher_Grouped"] = work["Publisher"].where(
            work["Publisher"].isin(top_publishers), "OtherPublisher"
        )
        work["Publisher_Grouped"] = work["Publisher_Grouped"].fillna("OtherPublisher")
        work["Genre"] = work["Genre"].fillna("其他")
        work["Platform_Family"] = work["Platform_Family"].fillna("Other")
        work["Decade"] = work["Decade"].fillna(work["Decade"].median())
        work["Year"] = work["Year"].fillna(work["Year"].median())
        for col in self.categorical_features:
            if col in work.columns:
                work[col] = work[col].astype("category")

        return work

    def _train_regression(self, df: pd.DataFrame) -> Optional[Dict[str, object]]:
        target = df["Global_Sales"]
        if target.nunique() <= 1 or len(df) < 50:
            LOGGER.warning("ML 模块：样本不足或目标列缺乏变化，跳过回归建模")
            return None

        feature_cols = self.numeric_features + self.categorical_features
        X = df[feature_cols].copy()
        y = target.copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        # LightGBM 回归模型
        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=12,
            min_child_samples=20,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        predictions = model.predict(X_test)
        metrics = {
            "model": "LGBMRegressor",
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "r2": float(r2_score(y_test, predictions)),
            "n_estimators_used": (
                model.best_iteration_ if model.best_iteration_ else model.n_estimators
            ),
        }

        # 获取 LightGBM 内置特征重要性
        feature_importance = self._collect_lgb_feature_importance(model, feature_cols)

        # 使用 SHAP 进行特征重要性分析
        shap_importance = self._compute_shap_importance(model, X_test, feature_cols)

        # 分位数回归 - 给出预测置信区间
        quantiles = self._train_quantile_models(
            X_train, y_train, X_test, y_test, feature_cols
        )

        samples = self._build_regression_samples(
            df.loc[y_test.index],
            predictions,
            quantiles.get("per_sample") if quantiles else None,
        )

        self._plot_feature_importance(feature_importance)
        self._plot_shap_summary(model, X_test, feature_cols)

        result: Dict[str, object] = {
            "metrics": metrics,
            "feature_importance": feature_importance,
            "samples": samples,
            "model": model,
        }
        if shap_importance:
            result["shap_importance"] = shap_importance
        if quantiles:
            result["quantiles"] = quantiles

        LOGGER.info(
            "LightGBM 回归完成 - MAE: %.4f, RMSE: %.4f, R²: %.4f",
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
        )
        return result

    def _train_classification(self, df: pd.DataFrame) -> Optional[Dict[str, object]]:
        """使用 LightGBM 进行分类建模，判断游戏是否为爆款"""
        labels = (df["Global_Sales"] >= self.hit_threshold).astype(int)
        if labels.nunique() <= 1:
            LOGGER.warning("ML 模块：没有足够的命中样本，跳过分类建模")
            return None

        feature_cols = self.numeric_features + self.categorical_features
        X = df[feature_cols].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=self.random_state, stratify=labels
        )

        # LightGBM 分类器
        base_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1,
        )

        base_model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        # 校准分类器以获得更可靠的概率估计
        calibrator_kwargs = {"method": "isotonic", "cv": 3}
        try:
            calibrated = CalibratedClassifierCV(
                estimator=base_model,
                **calibrator_kwargs,
            )
        except TypeError:
            calibrated = CalibratedClassifierCV(
                base_estimator=base_model,
                **calibrator_kwargs,
            )

        calibrated.fit(X_train, y_train)
        predictions = calibrated.predict(X_test)
        probabilities = calibrated.predict_proba(X_test)[:, 1]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="binary", zero_division=0
        )
        metrics = {
            "model": "LGBMClassifier",
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

        LOGGER.info(
            "LightGBM 分类完成 - Accuracy: %.4f, F1: %.4f",
            metrics["accuracy"],
            metrics["f1"],
        )

        return {
            "metrics": metrics,
            "samples": samples,
            "calibration": {"curve": calibration_points},
        }

    def _cluster_regions(self, df: pd.DataFrame) -> Optional[Dict[str, object]]:
        """使用 K-Means++ 配合 PCA 进行区域偏好聚类"""
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

        # 标准化 + PCA 降维
        scaler = StandardScaler()
        scaled = scaler.fit_transform(share)

        n_components = min(2, len(region_cols))
        pca = PCA(n_components=n_components, random_state=self.random_state)
        pca_features = pca.fit_transform(scaled)

        # K-Means++ 聚类
        model = KMeans(
            n_clusters=cluster_count,
            n_init=20,
            init="k-means++",
            random_state=self.random_state,
        )
        labels = model.fit_predict(pca_features)
        df_with_cluster = df.copy()
        df_with_cluster["cluster"] = labels
        df_with_cluster["pca_1"] = pca_features[:, 0]
        if pca_features.shape[1] > 1:
            df_with_cluster["pca_2"] = pca_features[:, 1]

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

        LOGGER.info(
            "K-Means++ 区域聚类完成 (PCA 方差解释率: %.2f%%)",
            100 * sum(pca.explained_variance_ratio_),
        )

        return {
            "segments": segments,
            "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "algorithm": "K-Means++ with PCA",
        }

    def _build_similarity_samples(self, df: pd.DataFrame) -> List[Dict[str, object]]:
        """
        构建相似性推荐样本
        """
        # 1. 特征选择
        cat_cols = ["Genre", "Platform_Family", "Publisher_Grouped"]
        valid_cat = [c for c in cat_cols if c in df.columns]
        num_cols = ["Composite_Score", "Year"]
        valid_num = [c for c in num_cols if c in df.columns]
        if len(df) < 10:
            return []
        # 2. 特征预处理
        work = df.copy()
        feature_parts = []
        if valid_num:
            for col in valid_num:
                work[col] = work[col].fillna(work[col].median())

            scaler = StandardScaler()
            scaled_num = scaler.fit_transform(work[valid_num])
            df_num = pd.DataFrame(scaled_num, index=work.index, columns=valid_num)
            feature_parts.append(df_num)

        # 处理分类特征：One-Hot 编码
        if valid_cat:
            df_cat = pd.get_dummies(work[valid_cat], dummy_na=False)
            genre_cols = [c for c in df_cat.columns if c.startswith("Genre_")]
            if genre_cols:
                df_cat[genre_cols] *= 2.0

            feature_parts.append(df_cat)

        if not feature_parts:
            return []

        matrix_df = pd.concat(feature_parts, axis=1)
        matrix_df = matrix_df.fillna(0)

        # 3. 计算相似度
        n_neighbors = min(11, len(df))
        neighbors = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors)
        neighbors.fit(matrix_df.values)

        # 4. 生成推荐列表
        anchors = (
            df.sort_values("Global_Sales", ascending=False)
            .drop_duplicates(subset=["Name"])
            .head(10)
        )
        recommendations: List[Dict[str, object]] = []

        # 获取矩阵值用于查询
        matrix_values = matrix_df.values
        index_positions = {idx: pos for pos, idx in enumerate(df.index)}

        for anchor_idx, anchor_row in anchors.iterrows():
            position = index_positions.get(anchor_idx)
            if position is None:
                continue

            distances, indices = neighbors.kneighbors(
                matrix_values[position].reshape(1, -1)
            )

            for dist, neighbor_pos in zip(distances[0], indices[0]):
                if dist < 1e-5:
                    continue

                neighbor_row = df.iloc[int(neighbor_pos)]

                if neighbor_row.get("Name") == anchor_row.get("Name"):
                    continue

                similarity_score = max(0.0, 1.0 - dist)

                recommendations.append(
                    {
                        "anchor": anchor_row.get("Name"),
                        "anchor_platform": anchor_row.get("Platform_Family"),
                        "candidate": neighbor_row.get("Name"),
                        "candidate_platform": neighbor_row.get("Platform_Family"),
                        "similarity": float(similarity_score),
                        "anchor_sales": float(anchor_row.get("Global_Sales", 0.0)),
                        "candidate_sales": float(neighbor_row.get("Global_Sales", 0.0)),
                        "genre_match": anchor_row.get("Genre")
                        == neighbor_row.get("Genre"),
                    }
                )

            if len(recommendations) >= 50:
                break

        return recommendations

    def _cluster_behavior_segments(
        self, df: pd.DataFrame
    ) -> Optional[Dict[str, object]]:
        """使用 GMM (高斯混合模型) 进行软聚类，识别游戏市场定位"""
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

        # PCA 降维提升聚类效果
        n_components = min(3, len(feature_cols))
        pca = PCA(n_components=n_components, random_state=self.random_state)
        pca_features = pca.fit_transform(scaled)

        cluster_count = min(5, max(2, len(work) // 250))
        if cluster_count < 2:
            cluster_count = 2

        # GMM 软聚类 - 给出概率而非硬分配
        model = GaussianMixture(
            n_components=cluster_count,
            covariance_type="full",
            random_state=self.random_state,
            n_init=3,
        )
        labels = model.fit_predict(pca_features)
        probabilities = model.predict_proba(pca_features)

        df_copy = df.copy()
        df_copy["behavior_cluster"] = labels
        # 保存每个样本属于各聚类的概率
        for i in range(cluster_count):
            df_copy[f"cluster_prob_{i}"] = probabilities[:, i]

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

            # 计算该聚类的平均归属概率（聚类纯度指标）
            avg_prob = float(group[f"cluster_prob_{cluster_id}"].mean())

            segments.append(
                {
                    "cluster": int(cluster_id),
                    "size": int(len(group)),
                    "avg_global_sales": float(group["Global_Sales"].mean()),
                    "avg_age": avg_age,
                    "score_median": score_median,
                    "cluster_purity": avg_prob,
                    "top_genres": group["Genre"].value_counts().head(3).index.tolist(),
                    "top_platforms": group["Platform_Family"]
                    .value_counts()
                    .head(2)
                    .index.tolist(),
                    "region_profile": region_profile,
                }
            )
        segments.sort(key=lambda seg: seg["avg_global_sales"], reverse=True)

        LOGGER.info(
            "GMM 行为聚类完成 - %d 个聚类, BIC: %.2f",
            cluster_count,
            model.bic(pca_features),
        )

        return {
            "segments": segments,
            "algorithm": "GMM (Gaussian Mixture Model) with PCA",
            "bic": float(model.bic(pca_features)),
            "aic": float(model.aic(pca_features)),
        }

    def _collect_lgb_feature_importance(
        self, model: lgb.LGBMRegressor, feature_names: List[str]
    ) -> List[Dict[str, float]]:
        """获取 LightGBM 内置特征重要性 (基于 gain)"""
        importances = model.feature_importances_
        pairs = sorted(
            zip(feature_names, importances), key=lambda item: item[1], reverse=True
        )[:15]
        return [
            {"feature": self._prettify_feature_name(name), "importance": float(value)}
            for name, value in pairs
        ]

    def _compute_shap_importance(
        self, model: lgb.LGBMRegressor, X_test: pd.DataFrame, feature_names: List[str]
    ) -> List[Dict[str, float]]:
        """使用 SHAP 计算特征重要性 - 基于博弈论的精确归因"""
        try:
            # TreeExplainer 针对树模型优化，速度极快
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            # 计算每个特征的平均绝对 SHAP 值
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            pairs = sorted(
                zip(feature_names, mean_abs_shap),
                key=lambda item: item[1],
                reverse=True,
            )[:15]

            return [
                {
                    "feature": self._prettify_feature_name(name),
                    "shap_importance": float(value),
                }
                for name, value in pairs
            ]
        except Exception as exc:
            LOGGER.warning("SHAP 计算失败：%s", exc)
            return []

    def _plot_shap_summary(
        self, model: lgb.LGBMRegressor, X_test: pd.DataFrame, feature_names: List[str]
    ) -> None:
        """生成 SHAP 蜂群图"""
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            shap_plot_path = (
                self.artifacts.feature_importance_png.parent / "shap_summary.png"
            )
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                X_test,
                feature_names=[self._prettify_feature_name(n) for n in feature_names],
                show=False,
            )
            plt.tight_layout()
            plt.savefig(shap_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            LOGGER.info("SHAP 蜂群图已保存: %s", shap_plot_path.name)
        except Exception as exc:
            LOGGER.warning("SHAP 图表生成失败：%s", exc)

    def _train_quantile_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_cols: List[str],
    ) -> Dict[str, object]:
        """使用 LightGBM 分位数回归给出预测置信区间"""
        alphas = [0.1, 0.5, 0.9]
        predictions: Dict[str, np.ndarray] = {}

        for alpha in alphas:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )
            preds = model.predict(X_test)
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

        LOGGER.info("LightGBM 分位数回归完成 - 10%/50%/90% 分位")
        return {"per_sample": per_sample, "bands": bands}

    def _prettify_feature_name(self, raw_name: str) -> str:
        name = raw_name.split("__", maxsplit=1)[-1]
        replacements = {
            "Platform_Family": "平台",
            "Genre": "类型",
            "Publisher_Grouped": "发行商",
            "Year": "年份",
            "Decade": "年代",
        }
        for key, label in replacements.items():
            if name == key:
                return label
            if name.startswith(key + "_"):
                return label + ":" + name[len(key) + 1 :]
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
        ax.set_xlabel("重要度 (Gain)")
        ax.set_title("LightGBM 特征重要度（Top 10）")
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
        regression_copy = None
        if regression:
            regression_copy = {k: v for k, v in regression.items() if k != "model"}

        payload = {
            "regression": regression_copy["metrics"] if regression_copy else None,
            "regression_quantiles": (
                regression_copy.get("quantiles", {}).get("bands")
                if regression_copy
                else None
            ),
            "classification": classification["metrics"] if classification else None,
            "calibration": (
                classification.get("calibration") if classification else None
            ),
            "clustering": clustering["segments"] if clustering else None,
            "clustering_algorithm": clustering.get("algorithm") if clustering else None,
            "behavior_clusters": (
                behavior_clusters["segments"] if behavior_clusters else None
            ),
            "behavior_clusters_algorithm": (
                behavior_clusters.get("algorithm") if behavior_clusters else None
            ),
            "top_features": (
                regression_copy["feature_importance"] if regression_copy else []
            ),
            "shap_features": (
                regression_copy.get("shap_importance") if regression_copy else []
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
            "regression_samples": regression_copy["samples"] if regression_copy else [],
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
