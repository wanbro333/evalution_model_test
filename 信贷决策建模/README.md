# 建模程序开发说明

当前目录保存模型 2.0 的源码、测试和正式输出。仓库总览、结果与完整运行方法见上级 [`README.md`](../README.md)。

## 数据与路径

- 原始题目和附件：`../data/raw/`
- 可重建 SQLite：`credit_data.db`
- 正式结果：`output/`
- 论文：`paper/`
- 离线网页：`dashboard.html`

路径、随机种子、预算、LGD、资金成本和 PD 档位集中定义在 `config.py`。不要在其他模块重复写死这些参数。

## 入口

从仓库根目录执行：

```powershell
python "信贷决策建模\main.py"
python -m unittest discover -s "信贷决策建模\tests" -v
python "信贷决策建模\verify_results.py"
```

论文编译：

```powershell
Set-Location "信贷决策建模\paper"
latexmk -xelatex -interaction=nonstopmode -halt-on-error paper.tex
```

## 模块边界

- `data_loader.py`：原始 Excel 清洗与 SQLite 入库。
- `feature_engineer.py`：连续月历和企业级发票特征。
- `risk_model.py`：正则逻辑回归、嵌套样本外预测、固定 PD 档位与 TOPSIS 解释。
- `credit_strategy.py`：保序流失率、期望收益定价和 MILP 额度优化。
- `sensitivity.py`：真实 TOPSIS 权重扰动与指标移除检验。
- `reporting.py`：统一指标、Excel、JSON 与 LaTeX 宏。
- `dashboard_renderer.py`：完全离线的教学型网页。
- `gen_paper_charts.py`：论文和网页共用图表。

旧版 `utils.py`、`visualize.py` 和旧图表已移至 `../archive/legacy-v1/`，不参与当前主流程。
