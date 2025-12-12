<<<<<<< HEAD
# 电子游戏销量分析管道

该仓库已实现围绕 `vgsales.csv` 数据集的完整工作流：从采集、ML 加持的独立清洗、入库、再到数据库驱动的分析、可视化与机器学习建模。分析阶段默认直接从 PostgreSQL 表 `vgchartz_clean` 读取整洁数据，仅在数据库不可用时回退至 CSV。输出更新为 20+ 张静态图表（含季度日历、六边形密度、平台生命周期、区域扩散、瀑布图等高级图），1 份交互式 HTML 仪表盘以及多份 ML/分析 JSON，且所有描述均为中文。

## 1. 环境准备
1. 安装 Python 3.10 及以上版本。
2. 在仓库根目录执行：
   ```powershell
   pip install -r requirements.txt
   ```
3. 确保本地 PostgreSQL 可通过 `postgresql://postgres:123400@localhost:5432/postgres` 访问。分析脚本会直接查询该库中的 `vgchartz_clean` 表；如连接信息不同，请在 `config.json` 或 `analysis.py` 中调整 `PostgresConfig`。
4. 若需要中文字体在图表中正确显示，请确保系统已安装微软雅黑/苹方/思源黑体等常见 CJK 字体。

## 2. 运行步骤
```powershell
python analysis.py
```
脚本将依次完成：
- 生成缺失值、异常年份、去重情况等数据质量报告。
- 数据清洗与特征工程（年代、平台家族、区域占比、长尾标记等）。
- 将标准化数据写入 Parquet，并同步一份到 PostgreSQL 表 `vgchartz_clean`（供分析端加载）。
- 计算多层指标：区域份额/HHI、平台生命周期动能、销量/评分分层、评分-销量相关性、集群偏好、Gini、CAGR/波动率等。
- 训练多模型组合：随机森林回归 + 分位回归带 + HistGB 备选模型、梯度提升命中分类 + 概率校准、区域 & 行为聚类，以及相似游戏推荐。
- 批量导出 20+ 张 Matplotlib/Plotly 图表（含季度日历、六边形密度、生命周期动能、区域扩散、生命周期瀑布、桑基流等）、中文 Markdown 摘要与交互式 HTML 仪表盘（Pyecharts 增强卡片）。

如需直接从 VGChartz 爬取最新榜单，可执行：

```powershell
python scrape_vgchartz.py --pages 10 --page-size 100 --platform All --sleep 1.2 \
   --dump-html-dir outputs/html_debug
```

该脚本会自动请求新版 `games.php` 列表并启用北美/PAL/日本/“Other” 分栏及总销量字段，输出到 `outputs/vgchartz_scrape.csv`。若站点前端再次变动，可使用 `--base-url` 覆盖目标地址，`--dump-html-dir` 则会保留原始 HTML 以便调试。

## 3. 输出物
| 文件                                 | 说明                                                                                            |
| ------------------------------------ | ----------------------------------------------------------------------------------------------- |
| `outputs/vgsales_clean.parquet`      | 清洗后的权威数据，可直接供 BI/建模使用。                                                        |
| `outputs/data_quality_report.json`   | 中文数据质量指标（缺失、去重、异常年份）。                                                      |
| `outputs/analysis_summary.md`        | 中文叙述摘要，列出核心结论、市场动能、分层表现等。                                              |
| `outputs/interactive_dashboard.html` | 交互式网页仪表盘，集成折线/面积/饼/热力/雷达/旭日/卡片。                                        |
| `outputs/ml_metrics.json`            | 多模型指标（随机森林 + 分位区间 + HistGB，对应 MAE/R²/带宽；分类含 Brier/校准；推荐与相似度）。 |
| `outputs/ml_clusters.json`           | 区域 + 行为聚类片段（含样本量、均值、年龄、评分、区域画像）。                                   |
| `outputs/ml_predictions_sample.json` | 模型预测示例 & 命中概率 Top 列表 & 相似游戏推荐对。                                             |
| `outputs/ml_feature_importance.png`  | 随机森林特征重要度 Top10 可视化。                                                               |
| `outputs/vgchartz_scrape.csv`        | VGChartz 实时爬取结果，含区域销量、开发商与基础评分字段。                                       |
| `outputs/01...21_*.png`              | 21 张静态图/可视化，覆盖分布、季度密度、六边形密度、生命周期、扩散、瀑布等。                    |

## 4. 额外提示
- 全部图表以 300 DPI 输出，适合直接嵌入报告或幻灯片。
- 若 PostgreSQL 未开启，脚本会记录中文告警，但不会阻断其余分析流程。
- 可在 `_generate_figures`、`_generate_interactive_dashboard` 或 `_compute_innovation_metrics` 中新增函数，扩展图表与指标而无需改动主流程。
- 交互式 HTML 页面位于 `outputs/interactive_dashboard.html`，可直接在浏览器中浏览或嵌入其他站点，适合线上展示。
- VGChartz 爬虫默认按页面间隔 1.5 秒访问，若手动调低 `--sleep`，请自行确保符合目标站点的访问策略。
=======
# VGgame-Sales-DataAnalysis-and-Visualization
>>>>>>> 8b0eb9337d6e6eba65f2efecc0f4a9de6f814017
