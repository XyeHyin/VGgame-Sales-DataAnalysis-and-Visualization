from datetime import datetime
from pathlib import Path
from string import Template
from typing import Dict, List, Optional

import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Bar, Grid, HeatMap, Line, Page, Pie, Radar, Sunburst, Gauge
from pyecharts.globals import CurrentConfig, ThemeType

from settings import (
    DashboardConfig,
    LOGGER,
    PLATFORM_LABELS,
    REGION_COLS,
    REGION_LABELS,
)


class DashboardBuilder:
    def __init__(self, output_path: Path, *, config: DashboardConfig) -> None:
        self.output_path = output_path
        self.config = config
        self._template_cache: Optional[Template] = None
        self._summary_template_cache: Optional[Template] = None

    def build(
        self,
        df: pd.DataFrame,
        metrics: Dict[str, object],
        static_charts: Optional[List[Path]] = None,
    ) -> None:
        LOGGER.info("正在生成交互式 HTML 仪表盘")
        charts = self._build_charts_dict(df, metrics)
        html = self._build_document(charts, metrics, static_charts or [])
        self.output_path.write_text(html, encoding="utf-8")
        LOGGER.info("交互式仪表盘已写入 %s", self.output_path)

    def _build_charts_dict(
        self, df: pd.DataFrame, metrics: Dict[str, object]
    ) -> Dict[str, str]:
        charts = {}

        # General Charts
        charts["yearly_line"] = self._render_chart(self._build_yearly_line_chart(df))
        charts["region_stream"] = self._render_chart(
            self._build_region_stream_chart(df)
        )
        charts["genre_bar"] = self._render_chart(self._build_genre_bar_chart(df))
        charts["platform_pie"] = self._render_chart(
            self._build_platform_pie_chart(metrics)
        )
        charts["platform_heatmap"] = self._render_chart(
            self._build_platform_genre_heatmap(df)
        )
        charts["region_radar"] = self._render_chart(self._build_region_radar_chart(df))
        charts["region_sunburst"] = self._render_chart(
            self._build_region_sunburst_chart(df)
        )

        # Add Sankey Flow
        charts["sankey_flow"] = (
            '<iframe src="19_sankey_flow.html" style="width:100%; height:600px; border:none;"></iframe>'
        )

        # ML Charts
        ml_charts = self._build_ml_charts(metrics)
        charts.update(ml_charts)

        return charts

    def _render_chart(self, chart) -> str:
        if chart is None:
            return ""
        return chart.render_embed()

    def _build_ml_charts(self, metrics: Dict[str, object]) -> Dict[str, str]:
        charts = {}
        ml_data = metrics.get("ml", {})
        if not ml_data:
            return charts

        # Feature Importance
        features = ml_data.get("top_features", [])
        if features:
            c = (
                Bar(init_opts=opts.InitOpts(theme=ThemeType.DARK))
                .add_xaxis([f["feature"] for f in features[:10]])
                .add_yaxis("重要度", [round(f["importance"], 4) for f in features[:10]])
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="ML 特征重要度 Top 10"),
                    xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=20)),
                    datazoom_opts=[opts.DataZoomOpts()],
                )
            )
            charts["ml_feature_importance"] = c.render_embed()

        # Model Performance Gauge (Regression R2)
        reg = ml_data.get("regression", {})
        if reg:
            r2 = reg.get("r2", 0)
            c = (
                Gauge(init_opts=opts.InitOpts(theme=ThemeType.DARK))
                .add(
                    "R² Score",
                    [("R²", round(r2, 4))],
                    min_=-1,
                    max_=1,
                    axisline_opts=opts.AxisLineOpts(
                        linestyle_opts=opts.LineStyleOpts(
                            color=[(0.3, "#fd666d"), (0.7, "#e6a23c"), (1, "#67e0e3")],
                            width=30,
                        )
                    ),
                )
                .set_global_opts(title_opts=opts.TitleOpts(title="回归模型 R² 评分"))
            )
            charts["ml_r2_gauge"] = c.render_embed()

        return charts

    def _build_document(
        self,
        charts: Dict[str, str],
        metrics: Dict[str, object],
        static_charts: List[Path],
    ) -> str:
        # Collect JS dependencies (simplified: just include all common ones or rely on CDN in template)
        # For now, we'll assume the template handles the main echarts.min.js
        # But pyecharts usually needs specific maps/etc.
        # Since we are embedding, the JS is inside the embed string, but the library loading is separate.
        # We will inject a script tag to load echarts from CDN if not present.

        summary_components = self._get_summary_components(metrics)

        static_html = ""
        for chart_path in static_charts:
            filename = chart_path.name
            # URL encode filename to handle Chinese characters safely in HTML
            from urllib.parse import quote

            encoded_filename = quote(filename)
            static_html += f"""
            <div class="gallery-item" onclick="openModal('{encoded_filename}')">
                <div class="gallery-img-container">
                    <img src="{encoded_filename}" alt="{filename}" loading="lazy">
                </div>
                <div class="gallery-caption">
                    <span>{filename}</span>
                </div>
            </div>
            """

        template = self._get_template()
        # Safe substitute with all chart keys and summary components
        return template.safe_substitute(
            PAGE_TITLE=self.config.page_title,
            HERO_TITLE=self.config.hero_title,
            HERO_SUBTITLE=self.config.hero_subtitle,
            DATA_SOURCE=self.config.data_source,
            STATIC_CHARTS=static_html,
            UPDATED_AT=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **charts,  # Unpack charts dict
            **summary_components,  # Unpack summary components
        )

    def _get_template(self) -> Template:
        if self._template_cache is None:
            template_text = self.config.template_path.read_text(encoding="utf-8")
            self._template_cache = Template(template_text)
        return self._template_cache

    def _get_summary_components(self, metrics: Dict[str, object]) -> Dict[str, str]:
        platform_info = metrics["innovation"]["platform_share"]
        top_genres = metrics["innovation"]["top_genres"][:5]
        preference = metrics["innovation"]["region_preference"][:4]
        moat = metrics["innovation"]["publisher_moat"]
        ml_metrics = metrics.get("ml", {}) or {}
        time_series = metrics.get("time_series", {})
        tier_contributions = metrics.get("tier_contributions", {})
        quality_gap = metrics.get("quality_gap", {})
        cluster_insights = metrics.get("cluster_insights", [])

        components = {}
        components["METRIC_CARDS"] = "".join(
            [
                self._render_metric_card("记录数量", f"{metrics['row_count']:,}"),
                self._render_metric_card(
                    "时间跨度",
                    f"{metrics['time_span'][0]} - {metrics['time_span'][1]}",
                ),
                self._render_metric_card(
                    "销量集中度 (Gini)", f"{metrics['innovation']['gini']:.3f}"
                ),
            ]
        )
        components["TOP_GENRES"] = "".join(
            self._render_list_item(f"🎮 {genre}: {sales:.2f} 百万套")
            for genre, sales in top_genres
        )
        components["PLATFORM_SHARES"] = "".join(
            self._render_list_item(
                f"🕹️ {PLATFORM_LABELS.get(name, name)} 占比 {share:.1%}"
            )
            for name, share in platform_info[:5]
        )
        components["REGION_PREFERENCES"] = "".join(
            self._render_list_item(
                f"📍 {REGION_LABELS.get(item['Region'], item['Region'])} 对 {item['Genre']} 偏好 +{item['Preference']:.1%}"
            )
            for item in preference
        )
        components["PUBLISHER_MOAT"] = "".join(
            self._render_list_item(
                f"🏢 {entry['Publisher']} 竞争力 {entry['moat_score']:.2f}"
            )
            for entry in moat
        )
        components["ML_INSIGHTS"] = self._build_ml_insights(ml_metrics)
        components["ML_SEGMENTS"] = self._build_ml_segments(ml_metrics)
        components["ML_BEHAVIOR"] = self._build_ml_behavior_segments(ml_metrics)
        components["ML_FEATURES"] = self._build_ml_features(ml_metrics)
        components["ML_SIMILARITY"] = self._build_ml_similarity(ml_metrics)
        components["MOMENTUM"] = self._build_momentum_section(time_series)
        components["TIER_SUMMARY"] = self._build_tier_section(tier_contributions)
        components["QUALITY_GAP"] = self._build_quality_section(quality_gap)
        components["CLUSTER_INSIGHTS"] = self._build_cluster_section(cluster_insights)

        return components

    def _build_dashboard_summary(self, metrics: Dict[str, object]) -> str:
        # Deprecated but kept for compatibility if needed, or just redirect
        # Since we removed the summary template usage in _build_document, this is dead code
        # unless called externally. We can remove it or leave it as a stub.
        return ""

    def _get_summary_template(self) -> Template:
        # Deprecated
        if self._summary_template_cache is None:
            template_text = self.config.summary_template_path.read_text(
                encoding="utf-8"
            )
            self._summary_template_cache = Template(template_text)
        return self._summary_template_cache

    @staticmethod
    def _render_metric_card(title: str, value: str) -> str:
        return (
            '<div class="metric-card">' f"<h3>{title}</h3>" f"<p>{value}</p>" "</div>"
        )

    @staticmethod
    def _render_list_item(content: str) -> str:
        return f'<div class="list-item">{content}</div>'

    def _build_ml_insights(self, ml_metrics: Dict[str, object]) -> str:
        if not ml_metrics:
            return ""
        items: List[str] = []
        regression = ml_metrics.get("regression") or {}
        classification = ml_metrics.get("classification") or {}
        if regression:
            items.append(
                self._render_list_item(
                    "🤖 回归模型 MAE {:.2f} / R² {:.2f}".format(
                        regression.get("mae", 0.0), regression.get("r2", 0.0)
                    )
                )
            )
        if classification:
            items.append(
                self._render_list_item(
                    "🎯 命中预测 F1 {:.2f}，准确率 {:.2f}".format(
                        classification.get("f1", 0.0),
                        classification.get("accuracy", 0.0),
                    )
                )
            )
        return "".join(items)

    def _build_ml_segments(self, ml_metrics: Dict[str, object]) -> str:
        segments = ml_metrics.get("clustering") or []
        if not segments:
            return ""
        rendered = []
        for segment in segments:
            rendered.append(
                self._render_list_item(
                    "🧩 Segment {cluster}: {region} 偏好，样本 {size}，均值 {sales:.2f}".format(
                        cluster=segment.get("cluster"),
                        region=segment.get("dominant_region", "未知"),
                        size=segment.get("size", 0),
                        sales=segment.get("avg_global_sales", 0.0),
                    )
                )
            )
        return "".join(rendered)

    def _build_ml_features(self, ml_metrics: Dict[str, object]) -> str:
        # 优先读取 SHAP 特征重要性，否则回退到传统特征重要性
        shap_features = ml_metrics.get("shap_features") or []
        top_features = ml_metrics.get("top_features") or []

        rendered = []

        # 如果有 SHAP 特征，优先显示
        if shap_features:
            for feature in shap_features[:5]:
                rendered.append(
                    self._render_list_item(
                        "🧠 SHAP 贡献: {name} = {score:.4f}".format(
                            name=feature.get("feature"),
                            score=feature.get("shap_importance", 0.0),
                        )
                    )
                )
        elif top_features:
            # 回退显示传统特征重要性
            for feature in top_features[:5]:
                rendered.append(
                    self._render_list_item(
                        "📈 {name} 重要度 {score:.2%}".format(
                            name=feature.get("feature"),
                            score=feature.get("importance", 0.0),
                        )
                    )
                )

        return "".join(rendered)

    def _build_ml_behavior_segments(self, ml_metrics: Dict[str, object]) -> str:
        segments = ml_metrics.get("behavior_clusters") or []
        if not segments:
            return ""
        rendered = []
        for segment in segments:
            rendered.append(
                self._render_list_item(
                    "🧭 行为集群 {cluster}: 销量 {sales:.2f}，年龄 {age:.1f}, 评分 {score:.1f}".format(
                        cluster=segment.get("cluster"),
                        sales=segment.get("avg_global_sales", 0.0),
                        age=segment.get("avg_age", 0.0),
                        score=segment.get("score_median", 0.0),
                    )
                )
            )
        return "".join(rendered)

    def _build_ml_similarity(self, ml_metrics: Dict[str, object]) -> str:
        pairs = ml_metrics.get("similar_titles") or []
        if not pairs:
            return ""
        rendered = []
        for pair in pairs[:5]:
            rendered.append(
                self._render_list_item(
                    "🤝 {anchor} ↔ {candidate} (相似度 {sim:.2f})".format(
                        anchor=pair.get("anchor"),
                        candidate=pair.get("candidate"),
                        sim=pair.get("similarity", 0.0),
                    )
                )
            )
        return "".join(rendered)

    def _build_momentum_section(self, time_series: Dict[str, object]) -> str:
        if not time_series:
            return ""
        items = []
        cagr = time_series.get("cagr")
        if cagr is not None:
            items.append(self._render_list_item(f"📈 CAGR {cagr:.2%}"))
        volatility = time_series.get("volatility")
        if volatility is not None:
            items.append(self._render_list_item(f"⚡ 波动率 {volatility:.2%}"))
        density = time_series.get("recent_release_density")
        if density is not None:
            items.append(
                self._render_list_item(f"📅 近五年发布密度 {density:.1f} 款/年")
            )
        boom = time_series.get("boom_periods", [])
        if boom:
            boom_desc = ", ".join(
                f"{entry['year']}({entry['yoy']:.1%})" for entry in boom[:3]
            )
            items.append(self._render_list_item(f"🌠 高光年份：{boom_desc}"))
        bust = time_series.get("bust_periods", [])
        if bust:
            bust_desc = ", ".join(
                f"{entry['year']}({entry['yoy']:.1%})" for entry in bust[:2]
            )
            items.append(self._render_list_item(f"🌧️ 回落年份：{bust_desc}"))
        return "".join(items)

    def _build_tier_section(self, tier_contributions: Dict[str, object]) -> str:
        if not tier_contributions:
            return ""
        lift = tier_contributions.get("tier_lift") or []
        if not lift:
            return ""
        rendered = []
        for entry in lift[:5]:
            rendered.append(
                self._render_list_item(
                    f"🏷️ {entry['tier']} 平均 {entry['avg_sales']:.2f} 百万套，lift {entry['lift']:.2f}"
                )
            )
        return "".join(rendered)

    def _build_quality_section(self, quality_gap: Dict[str, object]) -> str:
        if not quality_gap:
            return ""
        items = []
        corr = quality_gap.get("correlation")
        if corr is not None:
            items.append(self._render_list_item(f"🎯 评分-销量相关性 {corr:.2f}"))
        gap = quality_gap.get("score_gap", {})
        if gap:
            items.append(
                self._render_list_item(
                    f"⚖️ 均值差 {gap.get('mean_gap', 0.0):.2f} / σ {gap.get('std_gap', 0.0):.2f}"
                )
            )
        disagreements = quality_gap.get("largest_disagreements") or []
        for row in disagreements[:3]:
            items.append(
                self._render_list_item(
                    "🛑 {name} ({platform}) 差值 {gap:.2f}".format(
                        name=row.get("name"),
                        platform=row.get("platform"),
                        gap=row.get("score_gap", 0.0),
                    )
                )
            )
        return "".join(items)

    def _build_cluster_section(self, clusters: List[Dict[str, object]]) -> str:
        if not clusters:
            return ""
        rendered = []
        for cluster in clusters[:4]:
            rendered.append(
                self._render_list_item(
                    f"🗺️ {cluster['label']}：样本 {cluster['size']}，均值 {cluster['avg_sales']:.2f}"
                )
            )
        return "".join(rendered)

    def _build_yearly_line_chart(self, df: pd.DataFrame) -> Optional[Grid]:
        yearly = df.groupby("Year")["Global_Sales"].sum().sort_index().round(2)
        if yearly.empty:
            return None
        line = (
            Line()
            .add_xaxis(yearly.index.astype(str).tolist())
            .add_yaxis(
                "全球销量",
                yearly.tolist(),
                is_smooth=True,
                areastyle_opts=opts.AreaStyleOpts(opacity=0.2),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="全球销量年度趋势"),
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                xaxis_opts=opts.AxisOpts(
                    name="年份", axislabel_opts=opts.LabelOpts(rotate=35)
                ),
                yaxis_opts=opts.AxisOpts(name="销量（百万套）"),
                toolbox_opts=opts.ToolboxOpts(),
                datazoom_opts=[opts.DataZoomOpts()],
            )
        )
        grid = Grid(
            init_opts=opts.InitOpts(theme=ThemeType.DARK, width="100%", height="100%")
        )
        grid.add(line, grid_opts=opts.GridOpts(is_contain_label=True))
        return grid

    def _build_region_stream_chart(self, df: pd.DataFrame) -> Optional[Grid]:
        region_year = df.groupby("Year")[REGION_COLS].sum().sort_index().round(2)
        if region_year.empty:
            return None
        chart = Line()
        chart.add_xaxis(region_year.index.astype(str).tolist())
        for col in REGION_COLS:
            chart.add_yaxis(
                REGION_LABELS[col],
                region_year[col].tolist(),
                stack="总量",
                areastyle_opts=opts.AreaStyleOpts(opacity=0.25),
                label_opts=opts.LabelOpts(is_show=False),
            )
        chart.set_global_opts(
            title_opts=opts.TitleOpts(title="区域销量堆叠趋势"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(
                name="年份", axislabel_opts=opts.LabelOpts(rotate=35)
            ),
            yaxis_opts=opts.AxisOpts(name="销量（百万套）"),
            legend_opts=opts.LegendOpts(pos_top="5%"),
            datazoom_opts=[opts.DataZoomOpts()],
            toolbox_opts=opts.ToolboxOpts(),
        )
        grid = Grid(
            init_opts=opts.InitOpts(theme=ThemeType.DARK, width="100%", height="100%")
        )
        grid.add(chart, grid_opts=opts.GridOpts(is_contain_label=True))
        return grid

    def _build_genre_bar_chart(self, df: pd.DataFrame) -> Optional[Grid]:
        genre_sales = (
            df.groupby("Genre")["Global_Sales"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        if genre_sales.empty:
            return None
        bar = (
            Bar()
            .add_xaxis(genre_sales.index.tolist())
            .add_yaxis("全球销量", genre_sales.round(2).tolist(), category_gap="35%")
            .set_global_opts(
                title_opts=opts.TitleOpts(title="全球热销游戏类型（前十）"),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=30)),
                yaxis_opts=opts.AxisOpts(name="销量（百万套）"),
                toolbox_opts=opts.ToolboxOpts(),
            )
        )
        grid = Grid(
            init_opts=opts.InitOpts(theme=ThemeType.DARK, width="100%", height="100%")
        )
        grid.add(bar, grid_opts=opts.GridOpts(is_contain_label=True))
        return grid

    def _build_platform_pie_chart(self, metrics: Dict[str, object]) -> Optional[Pie]:
        data = metrics["innovation"]["platform_share"]
        if not data:
            return None
        return (
            Pie(
                init_opts=opts.InitOpts(
                    theme=ThemeType.DARK, width="100%", height="100%"
                )
            )
            .add(
                "",
                [
                    opts.PieItem(
                        name=PLATFORM_LABELS.get(name, name),
                        value=round(value * 100, 2),
                    )
                    for name, value in data
                ],
                radius=["35%", "70%"],
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="平台家族销量占比"),
                legend_opts=opts.LegendOpts(
                    orient="vertical", pos_left="5%", pos_top="20%"
                ),
                toolbox_opts=opts.ToolboxOpts(),
            )
            .set_series_opts(
                tooltip_opts=opts.TooltipOpts(formatter="{b}: {c}%"),
                label_opts=opts.LabelOpts(formatter="{b}: {d}%"),
            )
        )

    def _build_platform_genre_heatmap(self, df: pd.DataFrame) -> Optional[Grid]:
        pivot = df.pivot_table(
            values="Global_Sales",
            index="Platform_Family_CN",
            columns="Genre",
            aggfunc="sum",
            fill_value=0,
        ).round(2)
        if pivot.empty:
            return None
        xaxis = pivot.columns.tolist()
        yaxis = pivot.index.tolist()
        data = [
            [i, j, pivot.iloc[j, i]]
            for i in range(len(xaxis))
            for j in range(len(yaxis))
        ]
        heatmap = (
            HeatMap()
            .add_xaxis(xaxis)
            .add_yaxis("平台-类型热力", yaxis, data)
            .set_global_opts(
                title_opts=opts.TitleOpts(title="平台家族 VS 游戏类型热力图"),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=40)),
                visualmap_opts=opts.VisualMapOpts(
                    max_=float(pivot.values.max()),
                    min_=0,
                    orient="horizontal",
                    pos_left="center",
                    pos_top="top",
                ),
                toolbox_opts=opts.ToolboxOpts(),
            )
        )
        grid = Grid(
            init_opts=opts.InitOpts(theme=ThemeType.DARK, width="100%", height="100%")
        )
        grid.add(
            heatmap, grid_opts=opts.GridOpts(is_contain_label=True, pos_bottom="15%")
        )
        return grid

    def _build_region_radar_chart(self, df: pd.DataFrame) -> Optional[Radar]:
        top_genres = df.groupby("Genre")["Global_Sales"].sum().nlargest(5).index
        dataset = df[df["Genre"].isin(top_genres)].groupby("Genre")[REGION_COLS].sum()
        if dataset.empty:
            return None
        indicators = [
            opts.RadarIndicatorItem(
                name=REGION_LABELS[col],
                max_=float(dataset[col].max()) * 1.1 + 0.01,
            )
            for col in REGION_COLS
        ]
        chart = Radar(
            init_opts=opts.InitOpts(theme=ThemeType.DARK, width="100%", height="100%")
        ).add_schema(schema=indicators, shape="circle")
        for genre in dataset.index:
            values = [float(dataset.loc[genre, col]) for col in REGION_COLS]
            chart.add(
                genre,
                [values],
                areastyle_opts=opts.AreaStyleOpts(opacity=0.1),
                linestyle_opts=opts.LineStyleOpts(width=2),
            )
        chart.set_global_opts(
            title_opts=opts.TitleOpts(title="区域偏好雷达图"),
            toolbox_opts=opts.ToolboxOpts(),
        )
        return chart

    def _build_region_sunburst_chart(self, df: pd.DataFrame) -> Optional[Sunburst]:
        region_platform = (
            df.groupby(["Top_Region_CN", "Platform_Family_CN"])["Global_Sales"]
            .sum()
            .reset_index()
        )
        if region_platform.empty:
            return None
        data = []
        for region, group in region_platform.groupby("Top_Region_CN"):
            children = [
                {
                    "name": row["Platform_Family_CN"],
                    "value": round(row["Global_Sales"], 2),
                }
                for _, row in group.iterrows()
            ]
            data.append(
                {
                    "name": region or "未知",
                    "value": round(group["Global_Sales"].sum(), 2),
                    "children": children,
                }
            )
        return (
            Sunburst(
                init_opts=opts.InitOpts(
                    theme=ThemeType.DARK, width="100%", height="100%"
                )
            )
            .add("", data, radius=[0, "85%"])
            .set_global_opts(
                title_opts=opts.TitleOpts(title="区域与平台层级结构"),
                toolbox_opts=opts.ToolboxOpts(),
            )
        )
