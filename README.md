# 电子游戏销量分析管道

该仓库已实现围绕 `vgchartz_scrape.csv` 数据集的完整工作流：从采集、ML 加持的独立清洗、入库、再到数据库驱动的分析、可视化与机器学习建模。分析阶段默认直接从 PostgreSQL 表 `vgchartz_clean` 读取整洁数据，仅在数据库不可用时回退至 CSV。输出更新为 19 张静态图表（含季度日历、六边形密度、平台生命周期、区域扩散、瀑布图等高级图），1 份交互式 HTML 仪表盘以及多份 ML/分析 JSON，且所有描述均为中文。

## 项目结构

```
dabian/
├── run.py                # 主入口脚本
├── requirements.txt      # Python 依赖
├── config/               # 配置文件目录
│   ├── config.json       # 路径与技术配置
│   ├── user_config.json  # 用户可编辑配置（数据库、日志等）
│   └── mapper.json       # 数据映射表（平台、类型、区域标签）
├── src/                  # 源代码模块
│   ├── __init__.py
│   ├── analysis.py       # 数据分析主模块
│   ├── clean_and_store.py # 清洗入库脚本
│   ├── dashboard.py      # 交互式仪表盘生成
│   ├── data_cleaning.py  # 数据清洗逻辑
│   ├── database.py       # PostgreSQL 数据库操作
│   ├── ml.py             # 机器学习模型
│   ├── pipeline.py       # 数据处理管道
│   ├── plots.py          # 静态图表生成
│   ├── scrape_vgchartz.py # VGChartz 爬虫
│   └── settings.py       # 配置加载与常量
├── data/                 # 原始数据文件
│   ├── vgchartz_scrape.csv
│   └── vgsales.csv
├── outputs/              # 输出结果
│   ├── gallery/          # 交互式图表
│   └── html_debug/       # 爬虫调试 HTML
├── templates/            # HTML 模板
│   ├── dashboard.html
│   └── dashboard_summary.html
└── docs/                 # 文档
```

## 1. 环境准备
1. 安装 Python 3.10 及以上版本。
2. 在仓库根目录执行：
   ```powershell
   pip install -r requirements.txt
   ```
3. 编辑 `config/user_config.json` 配置数据库连接信息：
   ```json
   {
     "database": {
       "host": "localhost",
       "port": 5432,
       "user": "postgres",
       "password": "your_password",
       "dbname": "postgres",
       "table": "vgchartz_clean"
     }
   }
   ```
4. 若需要中文字体在图表中正确显示，请确保系统已安装微软雅黑/苹方/思源黑体等常见 CJK 字体。

## 2. 运行步骤
1.从 VGChartz 爬取最新榜单，可执行：

```powershell
python -m src.scrape_vgchartz --pages 1 --page-size 10000 --proxy-ignore
```
该脚本会自动请求新版 `games.php` 列表并启用北美/PAL/日本/"Other" 分栏及总销量字段，输出到 `outputs/vgchartz_scrape.csv`。若站点前端再次变动，可使用 `--base-url` 覆盖目标地址，`--dump-html-dir` 则会保留原始 HTML 以便调试。

2.执行数据清洗和入库：
```powershell
python -m src.clean_and_store
```
3.运行完整分析管线：
```powershell
python run.py
```
脚本将依次完成：
- 生成缺失值、异常年份、去重情况等数据质量报告。
- 数据清洗与特征工程（年代、平台家族、区域占比、长尾标记等）。
- 将标准化数据写入 Parquet，并同步一份到 PostgreSQL 表 `vgchartz_clean`（供分析端加载）。
- 计算多层指标：区域份额/HHI、平台生命周期动能、销量/评分分层、评分-销量相关性、集群偏好、Gini、CAGR/波动率等。
- 训练多模型组合：随机森林回归 + 分位回归带 + HistGB 备选模型、梯度提升命中分类 + 概率校准、区域 & 行为聚类，以及相似游戏推荐。
- 批量导出 20 张 Matplotlib/Plotly 图表（含季度日历、六边形密度、生命周期动能、区域扩散、生命周期瀑布、桑基流等）、中文 Markdown 摘要与交互式 HTML 仪表盘（Pyecharts 增强卡片）。
## 3. 配置说明

| 配置文件                  | 说明                                                                     |
| ------------------------- | ------------------------------------------------------------------------ |
| `config/config.json`      | 路径配置（数据文件、输出目录）                                           |
| `config/user_config.json` | 用户配置（数据库连接、日志级别、仪表盘文本），**克隆仓库后需编辑此文件** |
| `config/mapper.json`      | 数据映射表（平台家族、平台/类型/区域的中文标签）                         |

## 4. 输出物
| 文件                                         | 说明                                                                                            |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `outputs/data/vgchartz_clean.parquet`        | 清洗后的权威数据，可直接供 BI/建模使用。                                                        |
| `outputs/metrics/data_quality.json`          | 中文数据质量指标（缺失、去重、异常年份）。                                                      |
| `outputs/reports/analysis_summary.md`        | 中文叙述摘要，列出核心结论、市场动能、分层表现等。                                              |
| `outputs/dashboard.html`                     | 交互式网页仪表盘，集成折线/面积/饼/热力/雷达/旭日/卡片。                                        |
| `outputs/metrics/ml_metrics.json`            | 多模型指标（随机森林 + 分位区间 + HistGB，对应 MAE/R²/带宽；分类含 Brier/校准；推荐与相似度）。 |
| `outputs/metrics/ml_clusters.json`           | 区域 + 行为聚类片段（含样本量、均值、年龄、评分、区域画像）。                                   |
| `outputs/metrics/ml_predictions_sample.json` | 模型预测示例 & 命中概率 Top 列表 & 相似游戏推荐对。                                             |
| `outputs/plots/ml_feature_importance.png`    | 随机森林特征重要度 Top10 可视化。                                                               |
| `outputs/data/vgchartz_scrape.csv`           | VGChartz 实时爬取结果，含区域销量、开发商与基础评分字段。                                       |
| `outputs/gallery/01...20_*.png`              | 20 张静态图/可视化，覆盖分布、季度密度、六边形密度、生命周期、扩散、瀑布等。                    |

## 5. 额外提示
- 全部图表以 300 DPI 输出，适合直接嵌入报告或幻灯片。
- 若 PostgreSQL 未开启，脚本会记录中文告警，但不会阻断其余分析流程。
- 可在 `_generate_figures`、`_generate_interactive_dashboard` 或 `_compute_innovation_metrics` 中新增函数，扩展图表与指标而无需改动主流程。
- 交互式 HTML 页面位于 `outputs/dashboard.html`，可直接在浏览器中浏览或嵌入其他站点，适合线上展示。
- VGChartz 爬虫默认按页面间隔 1.5 秒访问，若手动调低 `--sleep`，请自行确保符合目标站点的访问策略。
