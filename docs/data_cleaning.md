# VGChartz 数据清理说明

本文档详细说明 `clean_and_store.py` / `GameDataCleaner` 在处理 `vgchartz_scrape.csv` 时执行的全部步骤、规则、算法与控制台报告格式，方便后续审计或扩展。

## 1. 输入与输出
- **输入**：`vgchartz_scrape.csv`（VGChartz 最新抓取的 66,771 条原始记录）。
- **核心输出**：
  - `outputs/vgchartz_clean.csv`（UTF-8 with BOM，Excel 直开不乱码）。
  - `outputs/vgchartz_clean.parquet`。
  - `outputs/data_quality.json`（清洗摘要 + 缺失值/字段类型）。
- **可选输出**：若执行 `clean_and_store.py` 时不带 `--skip-db`，会将清洗后的 DataFrame 直接写入 `postgres.vgchartz_clean` 表。

## 2. 清洗流水线
1. **列名与空值归一**  
   - `strip()` 去除列名首尾空格。  
   - 全表替换 `"" / "N/A" / "na" / "None"` 为 `NaN`。

2. **字符串标准化**  
   - 针对 `Name / Platform / ... / Last_Update` 等文本列统一为 `pandas.StringDtype()`，并再次剔除空白字符串。

3. **时间字段解析**  
   - `Release_Date` 与 `Last_Update` 依次尝试 `%d %b %Y`, `%d %B %Y`, `%Y-%m-%d`, `%d-%b-%Y`，并兼容 “01st Jan 88” 这样的后缀。  
   - `Year` 优先使用原始数值；若缺失则回填 `Release_Date/Last_Update` 的年份；最后裁剪在 `[1970, 当前年份]`。

4. **数值列类型转换**  
   - `Rank`, `NA_Sales`, `VGChartz_Score`, `User_Score` 等列统一 `pd.to_numeric(errors="coerce")`，保证后续算法使用浮点数。

5. **销量列合并（Global vs Total_Shipped）**  
   - `Global_Sales` 与 `Total_Shipped` 视为同一业务指标。  
   - 先数值化两列；随后若 `Global_Sales` 为 `NaN` 或 `0` 且 `Total_Shipped` 有正值，则使用 `Total_Shipped`。  
   - 该合并在 *任何过滤或特征工程之前* 完成，避免因 0 值被提前丢弃。  
   - 两列仍会同时保留，方便追踪数据来源。

6. **分类字段补全**  
   - `Genre` 缺失填充 “其他”。  
   - `Platform_Family` 若缺失，依据 `config.json -> taxonomy.platform_family_map` 的最新映射推导；映射覆盖了 82 个 `Platform` 值（Nintendo / Sega / Atari / Mobile / Home Computer 等）。

7. **地区销量整理**  
   - `NA/EU/JP/Other_Sales` 统一裁剪为非负，并汇总 `Regional_Sales_Sum`（若列缺失则记 0）。

8. **记录过滤**  
   - 丢弃 `Platform="Series"` 这类聚合行。  
   - 要求 `Name`, `Platform`, `Global_Sales` 非空。  
   - 仅保留 `Global_Sales > 0` 且 `1970 <= Year <= 当前年份` 的记录。  
   - 按 `(Name, Platform, Year, Publisher)` 去重，保留第一条。

9. **特征工程**  
   - `Genre` 映射为中文标签（已覆盖 Action-Adventure、Visual Novel、MMO 等 20 个类别，如未命中则保留英文）。  
   - 计算 `Top_Region`/`Top_Region_CN`、各区域份额、`Platform_Family_CN`、`Decade`、`Age_Years`、`Release_Quarter`。  
   - 综合评分：`Composite_Score = mean(Critic_Score, User_Score)`，并给出 `Score_Gap`、`Has_Critic_Score`、`Has_User_Score` 标志。  
   - 离散化：
     - `Sales_Tier`：`<1M / 1-5M / 5-10M / 10-20M / 20M+`。
     - `Score_Tier`：为避免 Excel 将 "6-8" 识别成日期，改用 `"6_to_8"`，其余为 `"<=6"`、`">=8"`。
   - 生成 `Canonical_Name`（去除 ®/™/©，压缩空格）。

10. **输出编码**  
    - `to_csv(..., encoding="utf-8-sig")`，以 BOM 形式保证在 Excel 中正常显示中文与特殊字符。

## 3. 机器学习步骤
1. **KNNImputer（缺失评分插补）**  
   - 触发条件：`VGChartz_Score`、`Critic_Score`、`User_Score` 中存在缺失且样本量 ≥ 5。  
   - 特征集合：`[VGChartz_Score, Critic_Score, User_Score, Global_Sales, Year]`。  
   - `n_neighbors = min(8, max(2, row_count-1))`，`weights="distance"`，插补值统一截断为 `>=0` 并四舍五入到两位小数。  
   - 控制台报告会打印是否执行、使用特征与邻居数量。

2. **KMeans（区域销量聚类）**  
   - 触发条件：具备全量区域列且样本量 ≥ 20。  
   - 将区域销量标准化为份额后再 `StandardScaler`，`n_clusters` 会根据样本量自适应（2/3/4）。  
   - 结果包含 `Regional_Cluster` 与中文描述 `Regional_Cluster_Label`（形如 “北美主导”）。

两类算法的配置由 `GameDataCleaner` 的 `random_state=42` 保证重现性。

## 4. 控制台与质量报告
- `clean_and_store.py` 运行时会输出：
  - 原始行数 / 最终行数 / 去重条数 / 关键字段缺失数 / 无销量记录数。
  - KNN 与聚类是否执行及其参数。
  - 数据保留率（%）。
- `outputs/data_quality.json` 记录：字段类型、缺失值分布、重复条数、年份范围、完整的 `CleaningResult.summary`。

## 5. 运行示例
```powershell
# 仅生成文件
T:/study/anaconda/python.exe clean_and_store.py --skip-db

# 生成文件并写入 PostgreSQL（需保证 config.json 中的连接可用）
T:/study/anaconda/python.exe clean_and_store.py
```

执行完成后，再运行 `analysis.py` / `pipeline.VGSalesPipeline` 即可对新的 `outputs/vgchartz_clean.csv` 继续生成指标、图表与仪表盘。
