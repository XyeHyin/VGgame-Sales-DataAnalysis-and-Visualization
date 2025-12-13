from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from pyecharts import options as opts
from pyecharts.charts import (
    Bar,
    Grid,
    HeatMap,
    Line,
    Page,
    Pie,
    Radar,
    Sunburst,
    Gauge,
    Sankey,
)
from pyecharts.commons.utils import JsCode
from pyecharts.globals import CurrentConfig, ThemeType

from src.settings import (
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
        self.env = Environment(loader=FileSystemLoader(config.template_path.parent))
        self.template = self.env.get_template(config.template_path.name)

    def build(
        self,
        df: pd.DataFrame,
        metrics: Dict[str, object],
        static_charts: Optional[List[Path]] = None,
    ) -> None:
        LOGGER.info("正在生成高精度交互式仪表盘")
        charts = self._build_charts_dict(df, metrics)

        # 构建静态图库 HTML (Data Vault)
        static_gallery_html = self._build_static_gallery(static_charts or [])

        html = self._build_document(charts, metrics, static_gallery_html)
        self.output_path.write_text(html, encoding="utf-8")
        LOGGER.info("仪表盘构建完成：%s", self.output_path)

    def _build_static_gallery(self, static_charts: List[Path]) -> str:
        """构建高科技感的静态图库数据 (JSON)"""
        gallery_data = []

        for chart_path in static_charts:
            filename = chart_path.name
            # 假设 HTML 在 outputs/，图片在 outputs/gallery/
            # 相对路径应该是 gallery/filename
            relative_path = f"gallery/{filename}"

            gallery_data.append(
                {
                    "src": relative_path,
                    "name": filename.replace(".png", "").replace("_", " ").upper(),
                    "id": filename,
                }
            )

        return json.dumps(gallery_data)

    def _build_charts_dict(
        self, df: pd.DataFrame, metrics: Dict[str, object]
    ) -> Dict[str, str]:
        charts = {}

        # --- 通用配置优化：消除留白，自适应宽度 ---
        # GridOpts: pos_left/right="0%" 消除左右留白
        full_width_grid = opts.GridOpts(
            pos_left="2%", pos_right="2%", pos_bottom="10%", is_contain_label=True
        )

        # 1. 核心预测图表 (放宽布局)
        charts["yearly_line"] = self._render_chart(
            self._build_yearly_line_chart(df, full_width_grid)
        )
        charts["region_stream"] = self._render_chart(
            self._build_region_stream_chart(df, full_width_grid)
        )

        # 2. 结构分析
        charts["genre_bar"] = self._render_chart(
            self._build_genre_bar_chart(df, full_width_grid)
        )
        charts["platform_pie"] = self._render_chart(
            self._build_platform_pie_chart(metrics)
        )

        # 3. 复杂关系 (热力图通常很宽，需要特殊处理)
        charts["platform_heatmap"] = self._render_chart(
            self._build_platform_genre_heatmap(df)
        )

        # 4. 区域详情
        charts["region_radar"] = self._render_chart(self._build_region_radar_chart(df))
        charts["region_sunburst"] = self._render_chart(
            self._build_region_sunburst_chart(df)
        )

        # 5. 桑基图 (重新设计配色)
        charts["sankey_flow"] = self._render_chart(self._build_sankey_chart(df))

        # 6. ML 仪表
        ml_charts = self._build_ml_charts(metrics)
        charts.update(ml_charts)

        return charts

    def _render_chart(self, chart) -> str:
        if chart is None:
            return '<div class="no-data">DATA FRAGMENTED</div>'
        return chart.render_embed()

    # --- 图表构建函数 (优化版) ---

    def _build_yearly_line_chart(
        self, df: pd.DataFrame, grid_opt: opts.GridOpts
    ) -> Optional[Grid]:
        yearly = df.groupby("Year")["Global_Sales"].sum().sort_index().round(2)
        if yearly.empty:
            return None

        c = (
            Line(
                init_opts=opts.InitOpts(
                    theme=ThemeType.DARK, width="100%", height="100%"
                )
            )
            .add_xaxis(yearly.index.astype(str).tolist())
            .add_yaxis(
                "全球销量",
                yearly.tolist(),
                is_smooth=True,
                areastyle_opts=opts.AreaStyleOpts(opacity=0.3, color="#00f3ff"),
                itemstyle_opts=opts.ItemStyleOpts(color="#00f3ff"),
                symbol_size=6,
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    axislabel_opts=opts.LabelOpts(rotate=0)
                ),  # 0度旋转，减少空间占用
                legend_opts=opts.LegendOpts(
                    is_show=False
                ),  # 标题已有说明，隐藏图例省空间
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                datazoom_opts=[
                    opts.DataZoomOpts(type_="inside")
                ],  # 隐藏滑块，只允许滚轮
            )
        )
        grid = Grid(
            init_opts=opts.InitOpts(theme=ThemeType.DARK, width="100%", height="100%")
        )
        grid.add(c, grid_opts=grid_opt)
        return grid

    def _build_region_stream_chart(
        self, df: pd.DataFrame, grid_opt: opts.GridOpts
    ) -> Optional[Grid]:
        region_year = df.groupby("Year")[REGION_COLS].sum().sort_index().round(2)
        if region_year.empty:
            return None

        c = Line(
            init_opts=opts.InitOpts(theme=ThemeType.DARK, width="100%", height="100%")
        )
        c.add_xaxis(region_year.index.astype(str).tolist())

        colors = ["#00f3ff", "#bc13fe", "#0aff60", "#ff0055"]
        for idx, col in enumerate(REGION_COLS):
            c.add_yaxis(
                REGION_LABELS[col],
                region_year[col].tolist(),
                stack="total",
                is_smooth=True,
                areastyle_opts=opts.AreaStyleOpts(opacity=0.6),
                itemstyle_opts=opts.ItemStyleOpts(color=colors[idx % len(colors)]),
                label_opts=opts.LabelOpts(is_show=False),
                symbol="none",  # 移除点，纯流图
            )

        c.set_global_opts(
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            legend_opts=opts.LegendOpts(pos_top="0%", pos_right="0%"),  # 图例放右上角
            datazoom_opts=[opts.DataZoomOpts(type_="inside")],
            xaxis_opts=opts.AxisOpts(boundary_gap=False),  # 消除X轴两侧留白
        )
        grid = Grid(
            init_opts=opts.InitOpts(theme=ThemeType.DARK, width="100%", height="100%")
        )
        grid.add(c, grid_opts=grid_opt)
        return grid

    def _build_genre_bar_chart(
        self, df: pd.DataFrame, grid_opt: opts.GridOpts
    ) -> Optional[Grid]:
        genre_sales = (
            df.groupby("Genre")["Global_Sales"]
            .sum()
            .sort_values(ascending=True)
            .tail(10)
        )
        if genre_sales.empty:
            return None

        c = (
            Bar(
                init_opts=opts.InitOpts(
                    theme=ThemeType.DARK, width="100%", height="100%"
                )
            )
            .add_xaxis(genre_sales.index.tolist())
            .add_yaxis(
                "销量",
                genre_sales.round(2).tolist(),
                itemstyle_opts=opts.ItemStyleOpts(
                    color=JsCode(
                        """
                        new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                            {offset: 0, color: '#bc13fe'},
                            {offset: 1, color: '#00f3ff'}
                        ])
                       """
                    )
                ),
            )
            .reversal_axis()  # 横向柱状图更适合长标签，不挤
            .set_global_opts(
                legend_opts=opts.LegendOpts(is_show=False),
                xaxis_opts=opts.AxisOpts(
                    splitline_opts=opts.SplitLineOpts(
                        is_show=True, linestyle_opts=opts.LineStyleOpts(opacity=0.1)
                    )
                ),
                yaxis_opts=opts.AxisOpts(
                    axisline_opts=opts.AxisLineOpts(is_show=False)
                ),
            )
        )
        # 横向图表留白调整
        grid = Grid(
            init_opts=opts.InitOpts(theme=ThemeType.DARK, width="100%", height="100%")
        )
        grid.add(
            c,
            grid_opts=opts.GridOpts(
                pos_left="15%", pos_right="5%", pos_bottom="10%", pos_top="5%"
            ),
        )
        return grid

    def _build_platform_genre_heatmap(self, df: pd.DataFrame) -> Optional[Grid]:
        pivot = df.pivot_table(
            index="Platform_Family_CN",
            columns="Genre",
            values="Global_Sales",
            aggfunc="sum",
            fill_value=0,
        ).round(1)
        if pivot.empty:
            return None

        c = (
            HeatMap(
                init_opts=opts.InitOpts(
                    theme=ThemeType.DARK, width="100%", height="100%"
                )
            )
            .add_xaxis(pivot.columns.tolist())
            .add_yaxis(
                "",
                pivot.index.tolist(),
                [
                    [i, j, pivot.iloc[j, i]]
                    for i in range(len(pivot.columns))
                    for j in range(len(pivot.index))
                ],
            )
            .set_global_opts(
                visualmap_opts=opts.VisualMapOpts(
                    pos_left="center",
                    pos_bottom="0%",
                    orient="horizontal",
                    is_calculable=True,
                    dimension=2,
                    range_color=[
                        "#050505",
                        "#300f5c",
                        "#bc13fe",
                        "#00f3ff",
                    ],  # 赛博朋克配色
                ),
                xaxis_opts=opts.AxisOpts(
                    axislabel_opts=opts.LabelOpts(rotate=45, font_size=10)
                ),
                yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=10)),
                tooltip_opts=opts.TooltipOpts(formatter="{b}: {c}"),
            )
        )
        grid = Grid(
            init_opts=opts.InitOpts(theme=ThemeType.DARK, width="100%", height="100%")
        )
        # 底部留白给 visualmap
        grid.add(
            c,
            grid_opts=opts.GridOpts(
                pos_bottom="15%", pos_top="5%", pos_left="10%", pos_right="5%"
            ),
        )
        return grid

    def _build_sankey_chart(self, df: pd.DataFrame) -> Optional[Sankey]:
        # 重写桑基图，不再使用 iframe，直接嵌入以控制样式
        data = (
            df.groupby(["Platform_Family_CN", "Genre", "Top_Region_CN"])["Global_Sales"]
            .sum()
            .reset_index()
        )
        if data.empty:
            return None

        # 节点和链接构建
        nodes_set = (
            set(data["Platform_Family_CN"])
            | set(data["Genre"])
            | set(data["Top_Region_CN"])
        )
        nodes = [{"name": n} for n in nodes_set]
        links = []

        # 第一层：平台 -> 类型
        l1 = (
            data.groupby(["Platform_Family_CN", "Genre"])["Global_Sales"]
            .sum()
            .reset_index()
        )
        for _, r in l1.iterrows():
            links.append({"source": r.iloc[0], "target": r.iloc[1], "value": r.iloc[2]})

        # 第二层：类型 -> 区域
        l2 = (
            data.groupby(["Genre", "Top_Region_CN"])["Global_Sales"].sum().reset_index()
        )
        for _, r in l2.iterrows():
            links.append({"source": r.iloc[0], "target": r.iloc[1], "value": r.iloc[2]})

        c = (
            Sankey(
                init_opts=opts.InitOpts(
                    theme=ThemeType.DARK, width="100%", height="100%"
                )
            )
            .add(
                "流向",
                nodes,
                links,
                pos_left="2%",
                pos_right="15%",
                pos_top="5%",
                pos_bottom="5%",
                linestyle_opt=opts.LineStyleOpts(
                    opacity=0.3, curve=0.5, color="source"
                ),
                label_opts=opts.LabelOpts(
                    position="right", color="#e0e0e0", font_size=12
                ),
                node_gap=10,
                node_width=25,
                layout_iterations=64,
                itemstyle_opts=opts.ItemStyleOpts(border_width=1, border_color="#aaa"),
            )
            .set_global_opts(
                tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}: {c}")
            )
        )
        return c

    def _build_ml_charts(self, metrics: Dict[str, object]) -> Dict[str, str]:
        charts = {}
        ml_data = metrics.get("ml", {})
        if not ml_data:
            return charts

        # 特征重要性
        features = ml_data.get("shap_features", []) or ml_data.get("top_features", [])
        if features:
            c = (
                Bar(
                    init_opts=opts.InitOpts(
                        theme=ThemeType.DARK, width="100%", height="100%"
                    )
                )
                .add_xaxis([f["feature"] for f in features[:8]])  # 只展示前8个，防拥挤
                .add_yaxis(
                    "SHAP Value",
                    [
                        round(f.get("shap_importance", f.get("importance", 0)), 4)
                        for f in features[:8]
                    ],
                    itemstyle_opts=opts.ItemStyleOpts(color="#0aff60"),
                )
                .reversal_axis()
                .set_global_opts(
                    xaxis_opts=opts.AxisOpts(
                        splitline_opts=opts.SplitLineOpts(
                            is_show=True, linestyle_opts=opts.LineStyleOpts(opacity=0.2)
                        )
                    ),
                    yaxis_opts=opts.AxisOpts(
                        axisline_opts=opts.AxisLineOpts(is_show=False)
                    ),
                )
            )
            grid = Grid(
                init_opts=opts.InitOpts(
                    theme=ThemeType.DARK, width="100%", height="100%"
                )
            )
            grid.add(
                c,
                grid_opts=opts.GridOpts(
                    pos_left="30%", pos_right="5%", pos_top="5%", pos_bottom="10%"
                ),
            )  # 左侧留大点给文字
            charts["ml_feature_importance"] = grid.render_embed()

        # R2 仪表盘
        reg = ml_data.get("regression", {})
        if reg:
            r2 = reg.get("r2", 0)
            c = (
                Gauge(
                    init_opts=opts.InitOpts(
                        theme=ThemeType.DARK, width="100%", height="100%"
                    )
                )
                .add(
                    "",
                    [("R² Score", round(r2, 3))],
                    min_=-1,
                    max_=1,
                    detail_label_opts=opts.GaugeDetailOpts(
                        offset_center=[0, "60%"], color="#fff", font_size=20
                    ),
                    axisline_opts=opts.AxisLineOpts(
                        linestyle_opts=opts.LineStyleOpts(
                            color=[(0.3, "#ff0055"), (0.7, "#bc13fe"), (1, "#00f3ff")],
                            width=20,
                        )
                    ),
                )
                .set_global_opts(legend_opts=opts.LegendOpts(is_show=False))
            )
            charts["ml_r2_gauge"] = c.render_embed()

        return charts

    # --- 其他辅助图表 (保持原样但应用 full width 逻辑) ---
    def _build_platform_pie_chart(self, metrics: Dict[str, object]) -> Optional[Pie]:
        data = metrics["innovation"]["platform_share"]
        if not data:
            return None

        # 计算总销量用于标签显示
        total_share = sum(v for _, v in data)

        return (
            Pie(
                init_opts=opts.InitOpts(
                    theme=ThemeType.DARK, width="100%", height="100%"
                )
            )
            .add(
                "平台份额",
                [
                    opts.PieItem(
                        name=PLATFORM_LABELS.get(n, n), value=round(v * 100, 2)
                    )
                    for n, v in data
                ],
                radius=["35%", "65%"],
                center=["55%", "50%"],
                label_opts=opts.LabelOpts(
                    formatter="{b}: {d}%",
                    color="#e0e0e0",
                    font_size=11,
                ),
            )
            .set_global_opts(
                legend_opts=opts.LegendOpts(
                    type_="scroll",
                    orient="vertical",
                    pos_left="2%",
                    pos_top="middle",
                    textstyle_opts=opts.TextStyleOpts(color="#e0e0e0", font_size=10),
                ),
                tooltip_opts=opts.TooltipOpts(
                    trigger="item", formatter="{a} <br/>{b}: {c}% ({d}%)"
                ),
            )
            .set_series_opts(
                label_opts=opts.LabelOpts(
                    formatter="{b}\n{d}%",
                    color="#e0e0e0",
                    font_size=10,
                )
            )
        )

    def _build_region_radar_chart(self, df: pd.DataFrame) -> Optional[Radar]:
        top_genres = df.groupby("Genre")["Global_Sales"].sum().nlargest(5).index
        dataset = df[df["Genre"].isin(top_genres)].groupby("Genre")[REGION_COLS].sum()
        if dataset.empty:
            return None

        indicators = [
            opts.RadarIndicatorItem(
                name=REGION_LABELS[c], max_=float(dataset[c].max() * 1.1)
            )
            for c in REGION_COLS
        ]
        c = Radar(
            init_opts=opts.InitOpts(theme=ThemeType.DARK, width="100%", height="100%")
        )
        c.add_schema(
            schema=indicators,
            shape="polygon",
            splitarea_opt=opts.SplitAreaOpts(
                is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=0.1)
            ),
        )

        colors = ["#00f3ff", "#ff0055", "#0aff60", "#bc13fe", "#ffff00"]
        for idx, genre in enumerate(dataset.index):
            c.add(
                genre,
                [dataset.loc[genre].tolist()],
                color=colors[idx % 5],
                areastyle_opts=opts.AreaStyleOpts(opacity=0.1),
                linestyle_opts=opts.LineStyleOpts(width=2),
            )
        c.set_global_opts(legend_opts=opts.LegendOpts(pos_bottom="0%"))
        return c

    def _build_region_sunburst_chart(self, df: pd.DataFrame) -> Optional[Sunburst]:
        # 简化版旭日图
        data = (
            df.groupby(["Top_Region_CN", "Platform_Family_CN"])["Global_Sales"]
            .sum()
            .reset_index()
        )
        if data.empty:
            return None
        tree = []
        for reg, g in data.groupby("Top_Region_CN"):
            children = [
                {"name": r["Platform_Family_CN"], "value": r["Global_Sales"]}
                for _, r in g.iterrows()
            ]
            tree.append({"name": reg, "children": children})

        c = (
            Sunburst(
                init_opts=opts.InitOpts(
                    theme=ThemeType.DARK, width="100%", height="100%"
                )
            )
            .add("", tree, radius=[0, "90%"])
            .set_global_opts(title_opts=opts.TitleOpts(title=""))
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}"))
        )
        return c

    def _build_document(
        self, charts: Dict[str, str], metrics: Dict[str, object], static_gallery: str
    ) -> str:
        summary_context = self._get_summary_context(metrics)

        context = {
            "PAGE_TITLE": self.config.page_title,
            "HERO_TITLE": self.config.hero_title,
            "HERO_SUBTITLE": self.config.hero_subtitle,
            "UPDATED_AT": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "STATIC_GALLERY_DATA": static_gallery,
            **charts,
            **summary_context,
        }

        return self.template.render(context)

    def _get_summary_context(self, metrics: Dict[str, object]) -> Dict[str, object]:
        platform_info = metrics["innovation"]["platform_share"]
        top_genres = metrics["innovation"]["top_genres"][:8]
        preference = metrics["innovation"]["region_preference"][:8]
        moat = metrics["innovation"]["publisher_moat"]
        ml_metrics = metrics.get("ml", {}) or {}
        time_series = metrics.get("time_series", {})
        tier_contributions = metrics.get("tier_contributions", {})
        quality_gap = metrics.get("quality_gap", {})
        cluster_insights = metrics.get("cluster_insights", [])

        context = {}
        context["metric_cards"] = [
            {"title": "记录数量", "value": f"{metrics['row_count']:,}"},
            {
                "title": "时间跨度",
                "value": f"{metrics['time_span'][0]} - {metrics['time_span'][1]}",
            },
            {
                "title": "销量集中度 (Gini)",
                "value": f"{metrics['innovation']['gini']:.3f}",
            },
        ]

        context["top_genres"] = [
            f"🎮 {genre}: {sales:.2f} 百万套" for genre, sales in top_genres
        ]
        context["platform_shares"] = [
            f"🕹️ {PLATFORM_LABELS.get(name, name)} 占比 {share:.1%}"
            for name, share in platform_info[:8]
        ]
        context["region_preferences"] = [
            f"📍 {REGION_LABELS.get(item['Region'], item['Region'])} 对 {item['Genre']} 偏好 +{item['Preference']:.1%}"
            for item in preference
        ]
        context["publisher_moat"] = [
            f"🏢 {entry['Publisher']} 竞争力 {entry['moat_score']:.2f}"
            for entry in moat
        ]

        context["ml_insights"] = self._build_ml_insights(ml_metrics)
        context["ml_segments"] = self._build_ml_segments(ml_metrics)
        context["ml_behavior"] = self._build_ml_behavior_segments(ml_metrics)
        context["ml_features"] = self._build_ml_features(ml_metrics)
        context["ml_similarity"] = self._build_ml_similarity(ml_metrics)
        context["momentum"] = self._build_momentum_section(time_series)
        context["tier_summary"] = self._build_tier_section(tier_contributions)
        context["quality_gap"] = self._build_quality_section(quality_gap)
        context["cluster_insights"] = self._build_cluster_section(cluster_insights)

        return context

    def _build_ml_insights(self, ml_metrics: Dict[str, object]) -> List[str]:
        if not ml_metrics:
            return []
        items: List[str] = []
        regression = ml_metrics.get("regression") or {}
        classification = ml_metrics.get("classification") or {}
        if regression:
            items.append(
                "🤖 回归模型 MAE {:.2f} / R² {:.2f}".format(
                    regression.get("mae", 0.0), regression.get("r2", 0.0)
                )
            )
        if classification:
            items.append(
                "🎯 命中预测 F1 {:.2f}，准确率 {:.2f}".format(
                    classification.get("f1", 0.0),
                    classification.get("accuracy", 0.0),
                )
            )
        return items

    def _build_ml_segments(self, ml_metrics: Dict[str, object]) -> List[str]:
        segments = ml_metrics.get("clustering") or []
        if not segments:
            return []
        rendered = []
        for segment in segments:
            rendered.append(
                "🧩 Segment {cluster}: {region} 偏好，样本 {size}，均值 {sales:.2f}".format(
                    cluster=segment.get("cluster"),
                    region=segment.get("dominant_region", "未知"),
                    size=segment.get("size", 0),
                    sales=segment.get("avg_global_sales", 0.0),
                )
            )
        return rendered

    def _build_ml_features(self, ml_metrics: Dict[str, object]) -> List[str]:
        # 优先读取 SHAP 特征重要性，否则回退到传统特征重要性
        shap_features = ml_metrics.get("shap_features") or []
        top_features = ml_metrics.get("top_features") or []

        rendered = []

        # 如果有 SHAP 特征，优先显示
        if shap_features:
            for feature in shap_features[:5]:
                rendered.append(
                    "🧠 SHAP 贡献: {name} = {score:.4f}".format(
                        name=feature.get("feature"),
                        score=feature.get("shap_importance", 0.0),
                    )
                )
        elif top_features:
            # 回退显示传统特征重要性
            for feature in top_features[:5]:
                rendered.append(
                    "📈 {name} 重要度 {score:.2%}".format(
                        name=feature.get("feature"),
                        score=feature.get("importance", 0.0),
                    )
                )

        return rendered

    def _build_ml_behavior_segments(self, ml_metrics: Dict[str, object]) -> List[str]:
        segments = ml_metrics.get("behavior_clusters") or []
        if not segments:
            return []
        rendered = []
        for segment in segments:
            rendered.append(
                "🧭 行为集群 {cluster}: 销量 {sales:.2f}，年龄 {age:.1f}, 评分 {score:.1f}".format(
                    cluster=segment.get("cluster"),
                    sales=segment.get("avg_global_sales", 0.0),
                    age=segment.get("avg_age", 0.0),
                    score=segment.get("score_median", 0.0),
                )
            )
        return rendered

    def _build_ml_similarity(self, ml_metrics: Dict[str, object]) -> List[str]:
        pairs = ml_metrics.get("similar_titles") or []
        if not pairs:
            return []
        rendered = []
        for pair in pairs[:5]:
            rendered.append(
                "🤝 {anchor} ↔ {candidate} (相似度 {sim:.2f})".format(
                    anchor=pair.get("anchor"),
                    candidate=pair.get("candidate"),
                    sim=pair.get("similarity", 0.0),
                )
            )
        return rendered

    def _build_momentum_section(self, time_series: Dict[str, object]) -> List[str]:
        if not time_series:
            return []
        items = []
        cagr = time_series.get("cagr")
        if cagr is not None:
            items.append(f"📈 CAGR {cagr:.2%}")
        volatility = time_series.get("volatility")
        if volatility is not None:
            items.append(f"⚡ 波动率 {volatility:.2%}")
        density = time_series.get("recent_release_density")
        if density is not None:
            items.append(f"📅 近五年发布密度 {density:.1f} 款/年")
        boom = time_series.get("boom_periods", [])
        if boom:
            boom_desc = ", ".join(
                f"{entry['year']}({entry['yoy']:.1%})" for entry in boom[:3]
            )
            items.append(f"🌠 高光年份：{boom_desc}")
        bust = time_series.get("bust_periods", [])
        if bust:
            bust_desc = ", ".join(
                f"{entry['year']}({entry['yoy']:.1%})" for entry in bust[:2]
            )
            items.append(f"🌧️ 回落年份：{bust_desc}")
        return items

    def _build_tier_section(self, tier_contributions: Dict[str, object]) -> List[str]:
        if not tier_contributions:
            return []
        lift = tier_contributions.get("tier_lift") or []
        if not lift:
            return []
        rendered = []
        for entry in lift[:5]:
            rendered.append(
                f"🏷️ {entry['tier']} 平均 {entry['avg_sales']:.2f} 百万套，lift {entry['lift']:.2f}"
            )
        return rendered

    def _build_quality_section(self, quality_gap: Dict[str, object]) -> List[str]:
        if not quality_gap:
            return []
        items = []
        corr = quality_gap.get("correlation")
        if corr is not None:
            items.append(f"🎯 评分-销量相关性 {corr:.2f}")
        gap = quality_gap.get("score_gap", {})
        if gap:
            items.append(
                f"⚖️ 均值差 {gap.get('mean_gap', 0.0):.2f} / σ {gap.get('std_gap', 0.0):.2f}"
            )
        disagreements = quality_gap.get("largest_disagreements") or []
        for row in disagreements[:3]:
            items.append(
                "🛑 {name} ({platform}) 差值 {gap:.2f}".format(
                    name=row.get("name"),
                    platform=row.get("platform"),
                    gap=row.get("score_gap", 0.0),
                )
            )
        return items

    def _build_cluster_section(self, clusters: List[Dict[str, object]]) -> List[str]:
        if not clusters:
            return []
        rendered = []
        for cluster in clusters[:4]:
            rendered.append(
                f"🗺️ {cluster['label']}：样本 {cluster['size']}，均值 {cluster['avg_sales']:.2f}"
            )
        return rendered
