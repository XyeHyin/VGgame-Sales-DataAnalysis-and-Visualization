from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from settings import (
    GLOBAL_SALES_LABEL_CN,
    LOGGER,
    REGION_COLS,
    REGION_LABELS,
)

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Noto Sans CJK SC",
]
plt.rcParams["axes.unicode_minus"] = False
plt.style.use("dark_background")
sns.set_theme(
    style="darkgrid",
    font="Microsoft YaHei",
    rc={
        "axes.facecolor": "#1e1e2e",
        "figure.facecolor": "#0f172a",
        "grid.color": "#444455",
        "text.color": "#e0e0e0",
        "axes.labelcolor": "#00f3ff",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "axes.edgecolor": "#00f3ff",
        "axes.titlecolor": "#ffffff",
    },
)


class FigureGenerator:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, df: pd.DataFrame) -> List[Path]:
        charts = [
            self._plot_global_sales_hist(df),
            self._plot_genre_boxplot(df),
            self._plot_platform_family_boxplot(df),
            self._plot_pareto_curve(df),
            # self._plot_yearly_sales_line(df),  # Covered by Pyecharts
            # self._plot_region_area(df),        # Covered by Pyecharts
            # self._plot_platform_genre_heatmap(df), # Covered by Pyecharts
            self._plot_top_publishers(df),
            self._plot_region_genre_panels(df),
            # self._plot_genre_radar(df),        # Covered by Pyecharts
            self._plot_correlation_heatmap(df),
            self._plot_lorenz_curve(df),
            # self._plot_decade_distributions(df),
            self._plot_region_scatter(df),
            self._plot_top_games_bubble(df),
            self._plot_calendar_heatmap(df),
            self._plot_score_sales_hexbin(df),
            self._plot_platform_lifecycle(df),
            self._plot_sankey_flow(df),
            self._plot_region_diffusion(df),
            self._plot_cohort_waterfall(df),
            # 新增高级可视化图表
            self._plot_console_war_bump(df),
            self._plot_genre_evolution_stream(df),
            self._plot_quality_sales_quadrant(df),
        ]
        return [path for path in charts if path]

    def _plot_global_sales_hist(self, df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(
            df["Global_Sales"], bins=40, log_scale=True, ax=ax, color="#4c72b0"
        )
        ax.set_title("全球销量分布（对数刻度）")
        ax.set_xlabel("全球销量（百万套）")
        ax.set_ylabel("游戏数量")
        return self._save_fig(fig, "01_全球销量分布直方图.png")

    def _plot_genre_boxplot(self, df: pd.DataFrame) -> Path:
        order = (
            df.groupby("Genre")["Global_Sales"]
            .median()
            .sort_values(ascending=False)
            .index
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x="Genre", y="Global_Sales", order=order, ax=ax)
        ax.set_title("不同类型的全球销量")
        ax.set_ylabel("全球销量（百万套）")
        ax.tick_params(axis="x", rotation=45)
        return self._save_fig(fig, "02_游戏类型箱线图.png")

    def _plot_platform_family_boxplot(self, df: pd.DataFrame) -> Path:
        order = (
            df.groupby("Platform_Family_CN")["Global_Sales"]
            .median()
            .sort_values(ascending=False)
            .index
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(
            data=df, x="Platform_Family_CN", y="Global_Sales", order=order, ax=ax
        )
        ax.set_title("不同平台家族的全球销量")
        ax.set_xlabel("平台家族")
        ax.set_ylabel("全球销量（百万套）")
        ax.tick_params(axis="x", rotation=30)
        return self._save_fig(fig, "03_平台家族箱线图.png")

    def _plot_calendar_heatmap(self, df: pd.DataFrame) -> Optional[Path]:
        if "Year" not in df.columns or "Release_Quarter" not in df.columns:
            return None
        calendar = df.groupby(["Year", "Release_Quarter"])["Name"].count().reset_index()
        if calendar.empty:
            return None
        calendar["Release_Quarter"] = calendar["Release_Quarter"].astype(str)
        pivot = calendar.pivot(
            index="Year", columns="Release_Quarter", values="Name"
        ).fillna(0)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot, cmap="mako", ax=ax)
        ax.set_title("季度发布密度热力图")
        ax.set_xlabel("季度")
        ax.set_ylabel("年份")
        return self._save_fig(fig, "16_季度发布热力图.png")

    def _plot_score_sales_hexbin(self, df: pd.DataFrame) -> Optional[Path]:
        if "Composite_Score" not in df.columns:
            return None
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hexbin(
            df["Composite_Score"],
            df["Global_Sales"],
            gridsize=35,
            cmap="viridis",
            mincnt=5,
        )
        ax.set_title("评分 vs 销量 六边形密度图")
        ax.set_xlabel("综合评分")
        ax.set_ylabel("全球销量（百万套）")
        cb = fig.colorbar(ax.collections[0], ax=ax)
        cb.set_label("样本数")
        return self._save_fig(fig, "17_评分销量六边形图.png")

    def _plot_platform_lifecycle(self, df: pd.DataFrame) -> Optional[Path]:
        if "Platform_Family_CN" not in df.columns or "Year" not in df.columns:
            return None
        platform_year = df.groupby(["Platform_Family_CN", "Year"])["Global_Sales"].sum()
        top_platforms = platform_year.groupby(level=0).sum().nlargest(6).index
        filtered = platform_year.loc[
            platform_year.index.get_level_values(0).isin(top_platforms)
        ]
        if filtered.empty:
            return None
        fig, ax = plt.subplots(figsize=(12, 6))
        for platform in top_platforms:
            series = (
                filtered.loc[platform]
                if platform in filtered.index.get_level_values(0)
                else None
            )
            if series is None:
                continue
            ax.plot(series.index, series.values, label=platform)
        ax.set_title("平台生命周期动能曲线")
        ax.set_xlabel("年份")
        ax.set_ylabel("全球销量（百万套）")
        ax.legend(ncol=2)
        return self._save_fig(fig, "18_平台生命周期.png")

    def _plot_sankey_flow(self, df: pd.DataFrame) -> Optional[Path]:
        if "Platform_Family_CN" not in df.columns or "Genre" not in df.columns:
            return None
        try:
            import plotly.graph_objects as go
        except ImportError:
            LOGGER.warning("Plotly 未安装，跳过桑基图生成")
            return None
        platform_genre_region = (
            df.groupby(["Platform_Family_CN", "Genre", "Top_Region_CN"])["Global_Sales"]
            .sum()
            .reset_index()
        )
        if platform_genre_region.empty:
            return None
        platforms = platform_genre_region["Platform_Family_CN"].unique().tolist()
        genres = platform_genre_region["Genre"].unique().tolist()
        regions = platform_genre_region["Top_Region_CN"].unique().tolist()
        nodes = platforms + genres + regions
        node_index = {name: idx for idx, name in enumerate(nodes)}
        sources = []
        targets = []
        values = []
        for _, row in platform_genre_region.iterrows():
            sources.append(node_index[row["Platform_Family_CN"]])
            targets.append(node_index[row["Genre"]])
            values.append(float(row["Global_Sales"]))
            sources.append(node_index[row["Genre"]])
            targets.append(node_index[row["Top_Region_CN"]])
            values.append(float(row["Global_Sales"]))
        fig = go.Figure(
            go.Sankey(
                arrangement="snap",
                node=dict(label=nodes, pad=18, thickness=18),
                link=dict(source=sources, target=targets, value=values),
            )
        )
        fig.update_layout(
            title_text="平台→类型→区域 桑基流",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
        )
        output = self.output_dir / "19_sankey_flow.html"
        fig.write_html(output)
        LOGGER.info("已保存图表 %s", output.name)
        return None

    def _plot_region_diffusion(self, df: pd.DataFrame) -> Optional[Path]:
        if "Regional_Cluster_Label" not in df.columns or "Year" not in df.columns:
            return None
        cluster_year = (
            df.groupby(["Year", "Regional_Cluster_Label"])["Name"].count().reset_index()
        )
        if cluster_year.empty:
            return None
        pivot = cluster_year.pivot(
            index="Year", columns="Regional_Cluster_Label", values="Name"
        ).fillna(0)
        pivot = pivot.sort_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.stackplot(pivot.index, pivot.values.T, labels=pivot.columns, alpha=0.85)
        ax.set_title("区域偏好集群扩散轨迹")
        ax.set_xlabel("年份")
        ax.set_ylabel("发行数量")
        ax.legend(loc="upper left", ncol=2)
        return self._save_fig(fig, "20_区域扩散轨迹.png")

    def _plot_cohort_waterfall(self, df: pd.DataFrame) -> Optional[Path]:
        if "Age_Years" not in df.columns:
            return None
        buckets = {
            "首发年": df[df["Age_Years"] <= 1],
            "1-3 年": df[(df["Age_Years"] > 1) & (df["Age_Years"] <= 3)],
            "4-7 年": df[(df["Age_Years"] > 3) & (df["Age_Years"] <= 7)],
            "7 年以上": df[df["Age_Years"] > 7],
        }
        contributions = []
        total = float(df["Global_Sales"].sum())
        if total == 0:
            return None
        for label, subset in buckets.items():
            contributions.append((label, float(subset["Global_Sales"].sum())))
        fig, ax = plt.subplots(figsize=(10, 6))
        cum = 0.0
        widths = 0.6
        for idx, (label, value) in enumerate(contributions):
            ax.bar(idx, value, width=widths, color="#4c72b0")
            cum += value
            ax.text(idx, value + total * 0.01, f"{value:.1f}", ha="center")
        ax.hlines(
            total,
            -0.5,
            len(contributions) - 0.5,
            colors="#c44e52",
            linestyles="--",
            label="总销量",
        )
        ax.set_xticks(range(len(contributions)))
        ax.set_xticklabels([label for label, _ in contributions])
        ax.set_ylabel("全球销量（百万套）")
        ax.set_title("生命周期销量瀑布图")
        ax.legend()
        return self._save_fig(fig, "21_生命周期瀑布图.png")

    def _plot_pareto_curve(self, df: pd.DataFrame) -> Path:
        sorted_sales = df.sort_values("Global_Sales", ascending=False).reset_index(
            drop=True
        )
        sorted_sales["CumShare"] = (
            sorted_sales["Global_Sales"].cumsum() / sorted_sales["Global_Sales"].sum()
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sorted_sales.index + 1, sorted_sales["CumShare"], color="#55a868")
        ax.axhline(0.8, color="r", linestyle="--", label="80% 分界线")
        ax.set_xscale("log")
        ax.set_xlabel("排名（对数刻度）")
        ax.set_ylabel("累积销量占比")
        ax.set_title("销量帕累托曲线")
        ax.legend()
        return self._save_fig(fig, "04_帕累托曲线.png")

    def _plot_yearly_sales_line(self, df: pd.DataFrame) -> Path:
        yearly = df.groupby("Year")["Global_Sales"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=yearly, x="Year", y="Global_Sales", ax=ax, marker="o")
        ax.set_title("全球销量年度趋势")
        ax.set_xlabel("年份")
        ax.set_ylabel("销量（百万套）")
        return self._save_fig(fig, "05_yearly_sales.png")

    def _plot_region_area(self, df: pd.DataFrame) -> Path:
        region_year = df.groupby("Year")[REGION_COLS].sum().reset_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        region_labels = [REGION_LABELS[col] for col in REGION_COLS]
        ax.stackplot(
            region_year["Year"],
            [region_year[col] for col in REGION_COLS],
            labels=region_labels,
            alpha=0.8,
        )
        ax.legend(loc="upper left")
        ax.set_title("各区域销量结构随时间变化")
        ax.set_xlabel("年份")
        ax.set_ylabel("销量（百万套）")
        return self._save_fig(fig, "06_region_stackplot.png")

    def _plot_platform_genre_heatmap(self, df: pd.DataFrame) -> Path:
        pivot = df.pivot_table(
            values="Global_Sales",
            index="Platform_Family_CN",
            columns="Genre",
            aggfunc="sum",
            fill_value=0,
        )
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot, cmap="YlGnBu", ax=ax)
        ax.set_title("平台家族与类型的销量热力图")
        ax.set_xlabel("游戏类型")
        ax.set_ylabel("平台家族")
        return self._save_fig(fig, "07_platform_genre_heatmap.png")

    def _plot_top_publishers(self, df: pd.DataFrame) -> Path:
        top_pub = (
            df.groupby("Publisher")["Global_Sales"].sum().nlargest(15).sort_values()
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        top_pub.plot(kind="barh", ax=ax, color="#c44e52")
        ax.set_title("全球销量前 15 名发行商")
        ax.set_xlabel("销量（百万套）")
        return self._save_fig(fig, "08_top_publishers.png")

    def _plot_region_genre_panels(self, df: pd.DataFrame) -> Path:
        melted = df.melt(
            id_vars=["Genre"],
            value_vars=REGION_COLS,
            var_name="Region",
            value_name="Sales",
        )
        top_genres = df.groupby("Genre")["Global_Sales"].sum().nlargest(10).index
        melted = melted[melted["Genre"].isin(top_genres)]
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
        axes = axes.flatten()
        for ax, region in zip(axes, REGION_COLS):
            data = melted[melted["Region"] == region]
            sns.barplot(data=data, x="Sales", y="Genre", ax=ax)
            ax.set_title(REGION_LABELS.get(region, region))
            ax.set_xlabel("销量（百万套）")
            ax.set_ylabel("游戏类型")
        fig.suptitle("各区域对热门类型的偏好")
        fig.tight_layout()
        return self._save_fig(fig, "09_region_genre_panels.png")

    def _plot_genre_radar(self, df: pd.DataFrame) -> Path:
        top_genres = df.groupby("Genre")["Global_Sales"].sum().nlargest(4).index
        genre_region = (
            df[df["Genre"].isin(top_genres)].groupby("Genre")[REGION_COLS].sum()
        )
        genre_region_pct = genre_region.divide(genre_region.sum(axis=1), axis=0)
        labels = [REGION_LABELS[col] for col in REGION_COLS]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        for genre in genre_region_pct.index:
            values = genre_region_pct.loc[genre].values
            values = np.concatenate((values, [values[0]]))
            ax.plot(angles, values, label=genre)
            ax.fill(angles, values, alpha=0.1)
        ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
        ax.set_title("热门类型的区域销量雷达图")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        return self._save_fig(fig, "10_genre_radar.png")

    def _plot_correlation_heatmap(self, df: pd.DataFrame) -> Path:
        corr = df[REGION_COLS + ["Global_Sales"]].corr()
        label_map = {**REGION_LABELS, "Global_Sales": GLOBAL_SALES_LABEL_CN}
        corr = corr.rename(index=label_map, columns=label_map)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, vmin=0, vmax=1)
        ax.set_title("销量相关系数矩阵")
        return self._save_fig(fig, "11_correlation_heatmap.png")

    def _plot_lorenz_curve(self, df: pd.DataFrame) -> Path:
        sales = np.sort(df["Global_Sales"].values)
        cumulative = np.cumsum(sales)
        total = cumulative[-1]
        lorenz = np.insert(cumulative / total, 0, 0)
        x_vals = np.linspace(0, 1, len(lorenz))
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_vals, lorenz, label="洛伦兹曲线")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="完全均等")
        ax.set_title("全球销量的洛伦兹曲线")
        ax.set_xlabel("累计游戏占比")
        ax.set_ylabel("累计销量占比")
        ax.legend()
        return self._save_fig(fig, "12_lorenz_curve.png")

    # def _plot_decade_distributions(self, df: pd.DataFrame) -> Path:
    #     g = sns.displot(
    #         data=df,
    #         x="Global_Sales",
    #         col="Decade",
    #         kind="hist",
    #         col_wrap=4,
    #         height=3,
    #         facet_kws={"sharex": False},
    #     )
    #     g.set_titles("{col_name} 年代")
    #     g.set(xlabel="全球销量（百万套）", ylabel="游戏数量")
    #     out_path = self.output_dir / "13_decade_distributions.png"
    #     g.fig.savefig(out_path, bbox_inches="tight")
    #     plt.close(g.fig)
    #     return out_path

    def _plot_region_scatter(self, df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.regplot(
            data=df, x="NA_Sales", y="EU_Sales", scatter_kws={"alpha": 0.3}, ax=ax
        )
        ax.set_title("北美与欧洲销量关系")
        ax.set_xlabel("北美销量（百万套）")
        ax.set_ylabel("欧洲销量（百万套）")
        return self._save_fig(fig, "14_na_eu_scatter.png")

    def _plot_top_games_bubble(self, df: pd.DataFrame) -> Path:
        sample = df.nlargest(100, "Global_Sales")
        platform_cat = sample["Platform_Family_CN"].astype("category")
        platform_colors = platform_cat.cat.codes
        fig, ax = plt.subplots(figsize=(12, 6))
        scatter = ax.scatter(
            sample["Year"],
            sample["Global_Sales"],
            s=sample["Global_Sales"] * 20,
            c=platform_colors,
            alpha=0.6,
            cmap="tab20",
        )
        ax.set_title("前 100 款游戏：年份与销量（气泡大小表示销量）")
        ax.set_xlabel("年份")
        ax.set_ylabel("全球销量（百万套）")
        categories = platform_cat.cat.categories
        norm = plt.Normalize(vmin=platform_colors.min(), vmax=platform_colors.max())
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=scatter.cmap(norm(idx)),
                label=label,
            )
            for idx, label in enumerate(categories)
        ]
        ax.legend(
            handles=handles,
            title="平台家族",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
        return self._save_fig(fig, "15_top_games_bubble.png")

    def _save_fig(self, fig: plt.Figure, filename: str) -> Path:
        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        LOGGER.info("已保存图表 %s", path.name)
        return path

    def _plot_console_war_bump(self, df: pd.DataFrame) -> Optional[Path]:
        """动态排名图 (Bump Chart) - 展示主机战争的历史演变"""
        if "Platform_Family_CN" not in df.columns or "Year" not in df.columns:
            return None

        # 统计每年各平台家族的销量
        platform_year = (
            df.groupby(["Year", "Platform_Family_CN"])["Global_Sales"]
            .sum()
            .reset_index()
        )
        if platform_year.empty:
            return None

        # 计算每年各平台的排名
        platform_year["Rank"] = platform_year.groupby("Year")["Global_Sales"].rank(
            ascending=False, method="min"
        )

        # 筛选主要平台 (至少出现 5 年)
        platform_counts = platform_year.groupby("Platform_Family_CN")["Year"].count()
        major_platforms = platform_counts[platform_counts >= 5].index.tolist()
        platform_year = platform_year[
            platform_year["Platform_Family_CN"].isin(major_platforms)
        ]

        if platform_year.empty or len(major_platforms) < 2:
            return None

        # 透视表
        pivot = platform_year.pivot(
            index="Year", columns="Platform_Family_CN", values="Rank"
        )

        fig, ax = plt.subplots(figsize=(14, 8))

        # 使用不同颜色绘制每个平台的排名曲线
        colors = plt.cm.tab10(np.linspace(0, 1, len(pivot.columns)))
        for idx, platform in enumerate(pivot.columns):
            series = pivot[platform].dropna()
            if len(series) < 2:
                continue
            ax.plot(
                series.index,
                series.values,
                marker="o",
                label=platform,
                linewidth=2.5,
                markersize=6,
                color=colors[idx],
            )
            # 在起点和终点标注平台名称
            if len(series) > 0:
                ax.annotate(
                    platform,
                    (series.index[-1], series.values[-1]),
                    xytext=(5, 0),
                    textcoords="offset points",
                    fontsize=8,
                    color=colors[idx],
                    va="center",
                )

        ax.invert_yaxis()  # 排名越小越好，所以反转Y轴
        ax.set_xlabel("年份")
        ax.set_ylabel("销量排名")
        ax.set_title("🎮 主机战争：平台销量排名演变 (Bump Chart)")
        ax.legend(bbox_to_anchor=(1.15, 1), loc="upper left", ncol=1, fontsize=8)
        ax.grid(True, alpha=0.3)

        return self._save_fig(fig, "22_console_war_bump.png")

    def _plot_genre_evolution_stream(self, df: pd.DataFrame) -> Optional[Path]:
        """游戏类型流图 (Streamgraph) - 展示玩家口味的流动"""
        if "Genre" not in df.columns or "Year" not in df.columns:
            return None

        # 统计每年各类型的销量
        genre_year = df.groupby(["Year", "Genre"])["Global_Sales"].sum().reset_index()
        if genre_year.empty:
            return None

        # 筛选主要类型 (销量前 8)
        top_genres = (
            df.groupby("Genre")["Global_Sales"].sum().nlargest(8).index.tolist()
        )
        genre_year = genre_year[genre_year["Genre"].isin(top_genres)]

        # 透视并填充缺失值
        pivot = genre_year.pivot(
            index="Year", columns="Genre", values="Global_Sales"
        ).fillna(0)
        pivot = pivot.sort_index()

        if pivot.empty or len(pivot) < 3:
            return None

        fig, ax = plt.subplots(figsize=(14, 7))

        # 使用堆叠面积图模拟流图
        colors = plt.cm.Set2(np.linspace(0, 1, len(pivot.columns)))
        ax.stackplot(
            pivot.index,
            pivot.values.T,
            labels=pivot.columns,
            colors=colors,
            alpha=0.85,
            baseline="wiggle",
        )

        ax.set_xlabel("年份")
        ax.set_ylabel("销量 (相对)")
        ax.set_title("🌊 游戏类型演变流图 (Streamgraph)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1, fontsize=9)
        ax.set_xlim(pivot.index.min(), pivot.index.max())

        # 移除Y轴刻度标签（流图更关注趋势而非绝对值）
        ax.set_yticklabels([])
        ax.axhline(0, color="white", linewidth=0.5, alpha=0.5)

        return self._save_fig(fig, "23_genre_evolution_stream.png")

    def _plot_quality_sales_quadrant(self, df: pd.DataFrame) -> Optional[Path]:
        """爆款象限图 (Quadrant Chart) - 评分 vs 销量四象限分析"""
        if "Composite_Score" not in df.columns or "Global_Sales" not in df.columns:
            return None

        # 过滤有评分的数据
        valid = df[df["Composite_Score"].notna() & (df["Global_Sales"] > 0)].copy()
        if len(valid) < 50:
            return None

        # 计算中位数作为分界线
        score_median = valid["Composite_Score"].median()
        sales_median = valid["Global_Sales"].median()

        fig, ax = plt.subplots(figsize=(12, 10))

        # 按类型着色
        if "Genre" in valid.columns:
            top_genres = (
                valid.groupby("Genre")["Global_Sales"].sum().nlargest(6).index.tolist()
            )
            genre_colors = {
                genre: plt.cm.tab10(i) for i, genre in enumerate(top_genres)
            }
            genre_colors["其他"] = "#888888"

            for genre in top_genres + ["其他"]:
                if genre == "其他":
                    subset = valid[~valid["Genre"].isin(top_genres)]
                else:
                    subset = valid[valid["Genre"] == genre]

                if len(subset) == 0:
                    continue

                ax.scatter(
                    subset["Composite_Score"],
                    subset["Global_Sales"],
                    alpha=0.5,
                    s=30,
                    label=genre,
                    color=genre_colors[genre],
                )
        else:
            ax.scatter(
                valid["Composite_Score"],
                valid["Global_Sales"],
                alpha=0.5,
                s=30,
                color="#4c72b0",
            )

        # 使用对数刻度
        ax.set_yscale("log")

        # 绘制中位数分界线
        ax.axvline(
            score_median, color="#ff6b6b", linestyle="--", linewidth=2, alpha=0.7
        )
        ax.axhline(
            sales_median, color="#ff6b6b", linestyle="--", linewidth=2, alpha=0.7
        )

        # 标注四个象限
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # 计算象限标注位置
        quadrant_labels = [
            (
                "叫好不叫座\n(高评低销)",
                score_median + (xlim[1] - score_median) * 0.5,
                ylim[0] * 3,
            ),
            (
                "叫好又叫座\n(高评高销) ⭐",
                score_median + (xlim[1] - score_median) * 0.5,
                ylim[1] * 0.3,
            ),
            (
                "口碑销量双低\n(低评低销)",
                xlim[0] + (score_median - xlim[0]) * 0.5,
                ylim[0] * 3,
            ),
            (
                "市场黑马\n(低评高销)",
                xlim[0] + (score_median - xlim[0]) * 0.5,
                ylim[1] * 0.3,
            ),
        ]

        for label, x, y in quadrant_labels:
            ax.text(
                x,
                y,
                label,
                fontsize=11,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
            )

        ax.set_xlabel("综合评分 (Composite Score)")
        ax.set_ylabel("全球销量 (百万套, 对数轴)")
        ax.set_title("💎 爆款象限图：评分 vs 销量四象限分析")

        if "Genre" in valid.columns:
            ax.legend(
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                ncol=1,
                fontsize=9,
                title="游戏类型",
            )

        # 添加分界线说明
        ax.text(
            score_median,
            ylim[1] * 0.8,
            f"评分中位数: {score_median:.1f}",
            fontsize=9,
            color="#ff6b6b",
            ha="center",
        )
        ax.text(
            xlim[1] * 0.95,
            sales_median,
            f"销量中位数: {sales_median:.2f}M",
            fontsize=9,
            color="#ff6b6b",
            ha="right",
            va="bottom",
        )

        return self._save_fig(fig, "24_quality_sales_quadrant.png")
