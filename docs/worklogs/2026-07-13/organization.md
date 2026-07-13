# 仓库整理记录

## 整理目标

将原始资料、当前模型、正式产物、设计文档和旧版文件分区保存，并形成可在 GitHub 完整复现的归档。

## 文件移动

- 四个原始题目文件从仓库根目录移至 `data/raw/`，文件内容和原文件名不变。
- `task_plan.md`、`findings.md`、`progress.md` 移至 `docs/worklogs/2026-07-13/`。
- 未被模型 2.0 引用的 `utils.py`、`visualize.py` 和 7 张旧图移至 `archive/legacy-v1/`。
- 当前主流程、论文、网页、Excel、JSON、CSV 和 11 张正式 `fig_*.png` 保持在 `信贷决策建模/` 内。

## 配套调整

- 新增 `config.DATA_DIR`，模型固定从 `data/raw/` 读取附件。
- 三个 Excel 通过 Git LFS 管理；140 MB SQLite 数据库继续作为本地可重建文件忽略。
- 重写根目录 README，修正旧版 AHP/熵权/TOPSIS 主模型描述。
- 测试改用 `pypdf` 读取已提交 PDF 页数，LaTeX 日志存在时再检查版式警告。
- `.gitignore` 补充 Python、LaTeX、QA 截图、编辑器和本地 agent 配置规则。

## 验证结果

- 从新数据目录完整重跑成功。
- 样本外 ROC-AUC 0.8557，PR-AUC 0.7339，Brier 0.1066。
- 三问预算为 8000、10000、10000 万元，期望净收益为 70.54、133.56、132.19 万元。
- Excel 含 739 个公式，结构错误为 0。
- 论文 19 页且无 LaTeX 警告。
- 8 项自动测试在有、无 `paper.log` 两种状态下均通过。
