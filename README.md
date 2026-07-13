# 中小微企业信贷决策模型 2.0

本仓库对应 2020 年全国大学生数学建模竞赛 C 题。模型以监督违约概率 PD 为风险主线，以固定尺度 TOPSIS 为经营质量解释支线，并通过单调客户流失率和混合整数规划联合决定贷款利率与额度。

## 核心结果

| 指标 | 结果 |
|---|---:|
| 样本外 ROC-AUC | 0.8557 |
| 样本外 PR-AUC | 0.7339 |
| Brier 分数 | 0.1066 |
| 问题 1 | 80 家，8000 万元，期望净收益 70.54 万元 |
| 问题 2 | 100 家，10000 万元，期望净收益 133.56 万元 |
| 问题 3 | 100 家，10000 万元，期望净收益 132.19 万元 |
| 原始 D 级获贷企业数 | 0 |

## 建模流程

```text
原始企业与发票数据
  -> 保留红冲符号并补齐连续月份
  -> 构造规模、盈利、稳定性和经营连续性特征
  -> 正则逻辑回归输出样本外 PD
  -> 固定 PD 阈值划分 A、B、C、D 档
  -> 保序回归拟合利率与客户流失率
  -> 计算留存利息、违约损失和资金成本
  -> MILP 在预算与准入约束下配置额度
  -> 疫情赔率冲击后重新评级、定价和分配
```

TOPSIS 不负责预测违约，只用于解释企业经营质量。模型验证采用重复嵌套交叉验证，避免使用训练内成绩证明模型效果。

## 仓库结构

```text
.
├── data/raw/                         # 原始题目与三个附件，Excel 由 Git LFS 管理
├── 信贷决策建模/
│   ├── main.py                       # 完整流水线入口
│   ├── config.py                     # 路径、随机种子与业务假设
│   ├── data_loader.py                # Excel 清洗与 SQLite 入库
│   ├── feature_engineer.py           # 发票特征工程
│   ├── risk_model.py                 # 监督 PD 与 TOPSIS 解释
│   ├── credit_strategy.py            # 定价与 MILP 额度优化
│   ├── sensitivity.py                # 真实权重扰动
│   ├── dashboard_renderer.py         # 离线论文伴读网页
│   ├── tests/                        # 单元和产物测试
│   ├── output/                       # CSV、JSON、Excel 和正式图表
│   └── paper/                        # LaTeX 源文件与 19 页 PDF
├── docs/                             # 设计规格与实施记录
└── archive/legacy-v1/                # 未参与当前主流程的旧版代码和图表
```

## 快速开始

### 1. 获取原始数据

```powershell
git lfs install
git lfs pull
```

原始文件清单和 SHA256 见 [`data/raw/README.md`](data/raw/README.md)。程序固定从 `data/raw/` 读取附件。

### 2. 安装依赖并运行

```powershell
python -m pip install -r "信贷决策建模\requirements.txt"
python "信贷决策建模\main.py"
```

完整运行会重建本地 SQLite 数据库，并同步生成 Excel、JSON、CSV、图表、网页和论文指标宏。`credit_data.db` 是可重建中间文件，不纳入 Git。

### 3. 编译论文

```powershell
Set-Location "信贷决策建模\paper"
latexmk -xelatex -interaction=nonstopmode -halt-on-error paper.tex
```

### 4. 运行验证

```powershell
python -m unittest discover -s "信贷决策建模\tests" -v
python "信贷决策建模\verify_results.py"
```

## 正式产物

- [`信贷决策建模/paper/paper.pdf`](信贷决策建模/paper/paper.pdf)：19 页教学型建模论文。
- [`信贷决策建模/dashboard.html`](信贷决策建模/dashboard.html)：完全离线的论文伴读与结果展示网页。
- [`信贷决策建模/output/信贷决策建模结果.xlsx`](信贷决策建模/output/信贷决策建模结果.xlsx)：企业评分、三问策略、收益拆分、系数和灵敏度结果。
- [`信贷决策建模/output/report_metrics.json`](信贷决策建模/output/report_metrics.json)：论文和网页共用的汇总指标源。

## 关键假设与限制

- 原始信誉 D 级或预测 PD 为 D 档的企业均禁贷。
- 单户额度为 0 或 10 至 100 万元。
- 问题 1 预算暂设 8000 万元；问题 2、3 预算为 1 亿元。
- LGD 暂设 60%，资金成本暂设 2.5%。
- 疫情行业分类来自脱敏企业名称，只作为压力测试，不是疫情影响的因果估计。
- 当前样本仅有 123 家带违约标签企业，正式应用仍需跨时间和跨地区外部验证。

所有随机过程使用固定种子，论文、网页和 Excel 指标由同一次流水线生成，避免手工抄录漂移。
