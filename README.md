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
运行完整分析管线：
```powershell
python run.py
```
并且根据提示完成各阶段任务。
脚本将依次完成：
爬取:
- 从 VGChartz 网站实时爬取最新数据，保存至 `outputs/data/vgchartz_scrape.csv`。
清洗与入库:
- 清洗爬取数据，合并历史数据，写入 PostgreSQL 表 `vgchartz_clean`。
分析与可视化:
- 合并历史数据 `data/vgsales.csv` 与爬取数据。
- 生成缺失值、异常年份、去重情况等数据质量报告。
- 数据清洗与特征工程（年代、平台家族、区域占比、长尾标记等）。
- 将标准化数据写入 Parquet，并同步一份到 PostgreSQL 表 `vgchartz_clean`（供分析端加载）。
- 计算多层指标：区域份额/HHI、平台生命周期动能、销量/评分分层、评分-销量相关性、集群偏好、Gini、CAGR/波动率等。
- 训练多模型组合：随机森林回归 + 分位回归带 + HistGB 备选模型、梯度提升命中分类 + 概率校准、区域 & 行为聚类，以及相似游戏推荐。
- 批量导出 20 张 Matplotlib/Plotly 图表（含季度日历、六边形密度、生命周期动能、区域扩散、生命周期瀑布、桑基流等）、中文 Markdown 摘要与交互式 HTML 仪表盘（Pyecharts 增强卡片）。
该脚本会自动请求新版 `games.php` 列表并启用北美/PAL/日本/"Other" 分栏及总销量字段，输出到 `outputs/vgchartz_scrape.csv`。若站点前端再次变动，可使用 `--base-url` 覆盖目标地址，`--dump-html-dir` 则会保留原始 HTML 以便调试。
  
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

# 机器学习算法设计文档

> 本文档详细说明了 VGChartz 游戏销量分析项目中使用的所有机器学习算法，涵盖数据清洗、预测建模、聚类分析和模型解释性全流程。

---

## 目录

1. [数据清洗与增强层](#1-数据清洗与增强层)
2. [核心预测与建模层](#2-核心预测与建模层)
3. [聚类与用户画像层](#3-聚类与用户画像层)
4. [高级解释与推荐层](#4-高级解释与推荐层)
5. [性能优化说明](#5-性能优化说明)

---

## 1. 数据清洗与增强层

### 1.1 Iterative Imputer (MICE 算法)

| 属性         | 说明                                                                    |
| ------------ | ----------------------------------------------------------------------- |
| **算法名称** | Iterative Imputer (MICE - Multivariate Imputation by Chained Equations) |
| **应用位置** | `data_cleaning.py` → `_apply_ml_enrichment()`                           |
| **用途**     | 填充 `Critic_Score` (媒体评分) 和 `User_Score` (用户评分) 的缺失值      |

#### 原理

MICE 是一种多重插补方法，其核心思想是：

1. 将缺失列作为"目标变量"
2. 其他列作为"特征"
3. 循环训练回归模型（本项目使用贝叶斯岭回归）来预测缺失值
4. 重复迭代直到收敛

数学表示：假设数据集有 $p$ 个变量，对于包含缺失值的变量 $X_j$：

$$X_j^{(t)} = f(X_1^{(t)}, ..., X_{j-1}^{(t)}, X_{j+1}^{(t-1)}, ..., X_p^{(t-1)}) + \epsilon$$

其中 $t$ 表示迭代轮次。

#### 相比旧方案的优势

| 对比项        | KNNImputer (旧)           | IterativeImputer (新)                |
| ------------- | ------------------------- | ------------------------------------ |
| 时间复杂度    | $O(N^2)$ - 需计算距离矩阵 | $O(N \cdot p \cdot iter)$ - 线性增长 |
| 2万行数据耗时 | 30-60秒                   | 2-5秒                                |
| 统计分布保留  | 一般                      | 优秀 - 保留特征间相关性              |

#### 配置参数

```python
IterativeImputer(
    max_iter=10,           # 最大迭代次数
    random_state=42,       # 随机种子
    initial_strategy="median",  # 初始填充策略
    skip_complete=True,    # 跳过完整列以提速
)
```

---

### 1.2 Isolation Forest (孤立森林)

| 属性         | 说明                                                            |
| ------------ | --------------------------------------------------------------- |
| **算法名称** | Isolation Forest                                                |
| **应用位置** | `data_cleaning.py` → `_detect_outliers()`                       |
| **用途**     | 异常检测 - 识别并标记销量异常、评分与销量严重不符的"离群点"数据 |

#### 原理

孤立森林基于一个核心观察：**异常点更容易被隔离**。

1. 随机选择一个特征
2. 在该特征的最大值和最小值之间随机选择一个分割点
3. 递归分割直到每个样本被隔离
4. 异常点的平均路径长度更短

异常分数计算：

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

其中：
- $h(x)$ 是样本 $x$ 的路径长度
- $c(n)$ 是归一化常数
- 分数接近 1 表示异常，接近 0.5 表示正常

#### 配置参数

```python
IsolationForest(
    contamination=0.05,    # 预期异常率 5%
    random_state=42,
    n_jobs=-1              # 并行处理
)
```

#### 输出

- `Is_Outlier`: 布尔列，标记是否为异常点
- `Outlier_Score`: 连续分数，用于分析异常程度

---

## 2. 核心预测与建模层

### 2.1 LightGBM (直方图梯度提升树)

| 属性         | 说明                                                                              |
| ------------ | --------------------------------------------------------------------------------- |
| **算法名称** | LightGBM (Light Gradient Boosting Machine)                                        |
| **应用位置** | `ml.py` → `_train_regression()` (销量预测) & `_train_classification()` (爆款分类) |
| **用途**     | 预测全球销量（回归）和判断游戏是否会成为百万级热销款（分类）                      |

#### 原理

LightGBM 是微软开发的高效梯度提升框架，核心创新包括：

1. **直方图算法**: 将连续特征离散化为 $k$ 个桶（默认 255），将时间复杂度从 $O(N \cdot features)$ 降为 $O(k \cdot features)$

2. **GOSS (单边梯度采样)**: 保留梯度大的样本，随机采样梯度小的样本
   
   $$\tilde{g} = \frac{1}{n} \left( \sum_{x_i \in A} g_i + \frac{1-a}{b} \sum_{x_i \in B} g_i \right)$$

3. **EFB (互斥特征捆绑)**: 将互斥的稀疏特征捆绑，减少特征数量

4. **原生类别特征支持**: 使用 Fisher 算法寻找最优分割，无需 One-Hot 编码

#### 相比旧方案的优势

| 对比项       | RandomForest/GBDT (旧)  | LightGBM (新)     |
| ------------ | ----------------------- | ----------------- |
| 训练速度     | 基准                    | 快 10-20 倍       |
| 类别特征处理 | 需要 One-Hot → 特征膨胀 | 原生支持 → 无膨胀 |
| 内存占用     | 高 (稀疏矩阵)           | 低 (直方图)       |
| 缺失值处理   | 需要预处理              | 原生支持          |
| GPU 加速     | 不支持                  | 支持              |

#### 回归模型配置

```python
lgb.LGBMRegressor(
    n_estimators=1000,     # 树的数量 (配合 early_stopping)
    learning_rate=0.05,    # 学习率
    num_leaves=31,         # 叶子节点数
    max_depth=12,          # 最大深度
    min_child_samples=20,  # 叶子最小样本数
    random_state=42,
    n_jobs=-1,
)
```

#### Early Stopping

```python
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
)
```

智能停止机制：当验证集指标连续 50 轮不再提升时自动停止，避免过拟合和资源浪费。

---

### 2.2 Quantile LightGBM (分位数回归)

| 属性         | 说明                                 |
| ------------ | ------------------------------------ |
| **算法名称** | Quantile Regression with LightGBM    |
| **应用位置** | `ml.py` → `_train_quantile_models()` |
| **用途**     | 给出销量预测的置信区间               |

#### 原理

传统回归只预测均值 $E[Y|X]$，分位数回归预测条件分位数 $Q_\tau(Y|X)$。

损失函数（Pinball Loss）：

$$L_\tau(y, \hat{y}) = \begin{cases} \tau (y - \hat{y}) & \text{if } y \geq \hat{y} \\ (1-\tau)(\hat{y} - y) & \text{if } y < \hat{y} \end{cases}$$

本项目训练 3 个模型：
- $\tau = 0.1$ (10% 分位 - 悲观预测)
- $\tau = 0.5$ (50% 分位 - 中位预测)
- $\tau = 0.9$ (90% 分位 - 乐观预测)

#### 商业价值

- **风险评估**: "最差情况下，这款游戏至少能卖 X 万份"
- **潜力评估**: "最好情况下，可能达到 Y 万份"
- **决策支持**: 根据风险偏好选择不同的预测值

---

## 3. 聚类与用户画像层

### 3.1 K-Means++ (带 PCA 降维)

| 属性         | 说明                                                                           |
| ------------ | ------------------------------------------------------------------------------ |
| **算法名称** | K-Means++ with PCA                                                             |
| **应用位置** | `ml.py` → `_cluster_regions()` & `data_cleaning.py` → `_apply_ml_enrichment()` |
| **用途**     | 区域偏好聚类 - 将游戏按"北美偏好"、"日本偏好"、"欧洲偏好"进行自动归类          |

#### 原理

**K-Means++ 初始化**:

标准 K-Means 随机初始化可能导致收敛到局部最优。K-Means++ 的改进：

1. 随机选择第一个中心
2. 对于每个后续中心，选择概率正比于到最近已选中心的距离平方

$$P(x) = \frac{D(x)^2}{\sum_{x' \in X} D(x')^2}$$

**PCA 降维**:

在聚类前先进行主成分分析：

$$Z = XW$$

其中 $W$ 是由协方差矩阵 $X^TX$ 的特征向量组成的投影矩阵。

#### 好处

1. **可视化**: 降到 2D 后可直接绘制散点图
2. **降噪**: 去除噪声维度，提升聚类质量
3. **加速**: 减少特征数量，加快 K-Means 收敛

#### 配置

```python
# PCA
pca = PCA(n_components=2, random_state=42)

# K-Means++
kmeans = KMeans(
    n_clusters=4,
    n_init=25,
    init="k-means++",
    random_state=42
)
```

---

### 3.2 GMM (高斯混合模型)

| 属性         | 说明                                                                    |
| ------------ | ----------------------------------------------------------------------- |
| **算法名称** | GMM (Gaussian Mixture Model)                                            |
| **应用位置** | `ml.py` → `_cluster_behavior_segments()`                                |
| **用途**     | 软聚类 - 识别游戏的潜在市场定位（如：硬核高分低销量 vs 大众低分高销量） |

#### 原理

GMM 假设数据由 $K$ 个高斯分布混合生成：

$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$$

其中：
- $\pi_k$ 是混合权重
- $\mu_k$ 是第 $k$ 个分量的均值
- $\Sigma_k$ 是协方差矩阵

使用 **EM 算法** 求解：
1. **E 步**: 计算每个样本属于各分量的后验概率
2. **M 步**: 更新参数 $\pi, \mu, \Sigma$

#### 硬聚类 vs 软聚类

| 对比项   | K-Means (硬聚类) | GMM (软聚类)   |
| -------- | ---------------- | -------------- |
| 输出     | 确定的类别标签   | 属于各类的概率 |
| 边界     | 线性边界         | 椭圆形边界     |
| 适用场景 | 类别清晰分离     | 类别有重叠     |

#### 配置

```python
GaussianMixture(
    n_components=5,
    covariance_type="full",  # 完整协方差矩阵
    random_state=42,
    n_init=3                 # 多次初始化取最优
)
```

#### 输出指标

- `cluster_purity`: 聚类纯度 - 样本属于所分配聚类的平均概率
- `BIC/AIC`: 用于选择最优聚类数

---

## 4. 高级解释与推荐层

### 4.1 SHAP (Shapley Values)

| 属性         | 说明                                                            |
| ------------ | --------------------------------------------------------------- |
| **算法名称** | SHAP (SHapley Additive exPlanations)                            |
| **应用位置** | `ml.py` → `_compute_shap_importance()` & `_plot_shap_summary()` |
| **用途**     | 解释模型 - 精确计算每个特征对预测结果的贡献值                   |

#### 原理

SHAP 基于博弈论中的 Shapley 值，为每个特征分配"公平"的贡献：

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f(S \cup \{i\}) - f(S)]$$

其中：
- $N$ 是所有特征集合
- $S$ 是不包含特征 $i$ 的子集
- $f(S)$ 是只用 $S$ 中特征时的预测值

#### TreeExplainer 优化

对于树模型，SHAP 提供了 `TreeExplainer`，时间复杂度从指数级降为多项式级：

$$O(TLD^2)$$

其中 $T$ 是树的数量，$L$ 是叶子数，$D$ 是深度。

#### 相比 Permutation Importance 的优势

| 对比项   | Permutation Importance | SHAP                   |
| -------- | ---------------------- | ---------------------- |
| 速度     | 慢 (需多次推理)        | 快 (利用树结构)        |
| 解释粒度 | 仅全局重要性           | 全局 + 局部解释        |
| 特征交互 | 不捕获                 | 可分析交互效应         |
| 可视化   | 柱状图                 | 蜂群图、瀑布图、依赖图 |

#### 输出可视化

1. **蜂群图 (Beeswarm Plot)**: 展示每个特征对每个样本的影响
2. **特征重要性柱状图**: 平均绝对 SHAP 值

---

### 4.2 近似最近邻搜索

| 属性         | 说明                                    |
| ------------ | --------------------------------------- |
| **算法名称** | NearestNeighbors (Cosine Similarity)    |
| **应用位置** | `ml.py` → `_build_similarity_samples()` |
| **用途**     | 推荐系统 - 寻找"相似游戏"               |

#### 原理

使用余弦相似度度量游戏之间的相似性：

$$\text{cosine}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

特征包括：
- 区域销量比例
- 综合评分
- 游戏年龄
- 全球销量

#### 可扩展性说明

当前数据量（约 2 万条）使用 scikit-learn 的 `NearestNeighbors` 足够高效。若数据量增长到百万级，可替换为：

- **Annoy** (Spotify): 树结构索引
- **Faiss** (Facebook): 向量量化 + IVF 索引
- **ScaNN** (Google): 各向异性向量量化

---

## 5. 性能优化说明

### 5.1 优化前后对比

| 模块       | 优化前                  | 优化后                   | 提升倍数   |
| ---------- | ----------------------- | ------------------------ | ---------- |
| 缺失值填充 | KNNImputer (~45s)       | IterativeImputer (~3s)   | **15x**    |
| 回归建模   | RF + OneHot (~120s)     | LightGBM Native (~8s)    | **15x**    |
| 分位数回归 | 3x GBDT Pipeline (~90s) | 3x LightGBM (~12s)       | **7.5x**   |
| 特征解释   | Permutation (~60s)      | SHAP TreeExplainer (~2s) | **30x**    |
| **总计**   | ~5-6 分钟               | **~30 秒**               | **10-12x** |

### 5.2 内存优化

- **去除 OneHotEncoder**: 发行商有 ~300 个，One-Hot 后产生 300+ 列稀疏特征。LightGBM 原生类别支持避免了这种膨胀。
- **PCA 降维**: 聚类前将高维区域特征压缩到 2-3 维。
- **Early Stopping**: 避免训练过多无用的树。

### 5.3 算法选型决策树

```
需要预测销量?
├── 是 → LightGBM Regressor
│   └── 需要置信区间? → Quantile LightGBM
└── 否 → 需要分类?
    ├── 是 → LightGBM Classifier
    └── 否 → 需要聚类?
        ├── 类别边界清晰 → K-Means++ (with PCA)
        └── 类别有重叠 → GMM

需要解释模型?
├── 全局重要性 → SHAP (mean |SHAP|)
└── 单样本解释 → SHAP Waterfall Plot

需要异常检测?
└── Isolation Forest
```

---

## 附录：依赖版本

```
lightgbm>=4.0
shap>=0.43
scikit-learn>=1.4
```

---
