"""中小微企业信贷决策模型的端到端运行入口。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from config import OUTPUT_DIR, ensure_directories
from credit_strategy import credit_strategy_pipeline
from dashboard_renderer import write_dashboard
from data_loader import load_all_data
from feature_engineer import build_all_features
from gen_paper_charts import create_all_charts
from reporting import (
    build_report_metrics,
    export_excel,
    save_report_metrics,
    validate_workbook,
    write_paper_macros,
)
from risk_model import risk_model_pipeline
from sensitivity import run_sensitivity


def _save_json(data, path: Path) -> None:
    with path.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _print_strategy(name, strategy, amount_col, net_col) -> None:
    selected = strategy[strategy[amount_col] > 0]
    print(
        f"  {name}: 放贷{len(selected)}家，额度{selected[amount_col].sum():.0f}万元，"
        f"期望净收益{strategy[net_col].sum():.2f}万元"
    )


def main() -> dict:
    """运行数据、风险、优化、报告与可视化全流程并返回核心指标。"""
    ensure_directories()
    print('=' * 72)
    print('中小微企业信贷决策模型 2.0：监督PD + 收益优化')
    print('=' * 72)

    conn = load_all_data()
    try:
        features1, features2 = build_all_features(conn)
    finally:
        conn.close()

    features1.to_csv(OUTPUT_DIR / 'features_附件1.csv', encoding='utf-8-sig')
    features2.to_csv(OUTPUT_DIR / 'features_附件2.csv', encoding='utf-8-sig')

    scores1, scores2, model_package, validation, coefficients = risk_model_pipeline(
        features1, features2
    )
    scores1.to_csv(OUTPUT_DIR / 'scores_附件1.csv', index=False, encoding='utf-8-sig')
    scores2.to_csv(OUTPUT_DIR / 'scores_附件2.csv', index=False, encoding='utf-8-sig')
    coefficients.to_csv(OUTPUT_DIR / 'model_coefficients.csv', index=False, encoding='utf-8-sig')
    _save_json(model_package, OUTPUT_DIR / 'model_package.json')

    strategy1, strategy2, strategy3, churn_model, churn_diagnostics = credit_strategy_pipeline(
        scores1, scores2
    )
    strategy1.to_csv(OUTPUT_DIR / 'strategy_附件1.csv', index=False, encoding='utf-8-sig')
    strategy2.to_csv(OUTPUT_DIR / 'strategy_附件2.csv', index=False, encoding='utf-8-sig')
    strategy3.to_csv(OUTPUT_DIR / 'strategy_附件2_疫情调整.csv', index=False, encoding='utf-8-sig')
    _save_json(churn_model, OUTPUT_DIR / 'churn_model.json')

    sensitivity_trials, indicator_removal, sensitivity_summary = run_sensitivity(
        features1, model_package
    )
    sensitivity_trials.to_csv(
        OUTPUT_DIR / 'sensitivity_trials.csv', index=False, encoding='utf-8-sig'
    )
    indicator_removal.to_csv(
        OUTPUT_DIR / 'indicator_removal.csv', index=False, encoding='utf-8-sig'
    )

    metrics = build_report_metrics(
        scores1,
        scores2,
        strategy1,
        strategy2,
        strategy3,
        validation,
        sensitivity_summary,
        churn_diagnostics,
        model_package,
    )
    create_all_charts(
        scores1,
        scores2,
        features1,
        coefficients,
        churn_model,
        sensitivity_trials,
        strategy1,
        strategy2,
        strategy3,
        metrics['data_summary'],
    )
    metrics_path = save_report_metrics(metrics)
    macros_path = write_paper_macros(metrics)
    dashboard_path = write_dashboard(metrics)
    excel_path = export_excel(
        scores1,
        scores2,
        strategy1,
        strategy2,
        strategy3,
        coefficients,
        validation,
        sensitivity_trials,
        indicator_removal,
        OUTPUT_DIR / '信贷决策建模结果.xlsx',
    )
    workbook_check = validate_workbook(excel_path)
    if workbook_check['errors'] or workbook_check['formula_count'] == 0:
        raise RuntimeError(f'Excel产物校验失败: {workbook_check}')

    print('\n结果摘要')
    _print_strategy('问题1', strategy1, '贷款额度_万元', '期望净收益')
    _print_strategy('问题2', strategy2, '贷款额度_万元', '期望净收益')
    _print_strategy('问题3', strategy3, '调整后额度', '调整后期望净收益')
    print(
        f"  样本外ROC-AUC={validation['roc_auc']:.4f}，"
        f"PR-AUC={validation['pr_auc']:.4f}，Brier={validation['brier']:.4f}"
    )
    print(f"  Excel公式数={workbook_check['formula_count']}，结构错误={len(workbook_check['errors'])}")
    print(f'  指标: {metrics_path}')
    print(f'  Excel: {excel_path}')
    print(f'  Dashboard: {dashboard_path}')
    print(f'  LaTeX宏: {macros_path}')
    return metrics


if __name__ == '__main__':
    main()
