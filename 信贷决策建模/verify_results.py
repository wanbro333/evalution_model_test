"""对 main.py 生成的正式产物执行快速业务约束核验。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import BUDGET_PROBLEM_1, BUDGET_PROBLEM_2, OUTPUT_DIR
from reporting import validate_workbook


def _check_allocation(frame, amount_col, budget, grade_col):
    amount = frame[amount_col].to_numpy(float)
    if not np.isclose(amount.sum(), budget, atol=1e-4):
        raise AssertionError(f'{amount_col}合计{amount.sum()}不等于预算{budget}')
    selected = amount[amount > 1e-7]
    if np.any((selected < 10 - 1e-6) | (selected > 100 + 1e-6)):
        raise AssertionError(f'{amount_col}违反单户10至100万元约束')
    if (frame.loc[frame[grade_col].eq('D'), amount_col] > 0).any():
        raise AssertionError(f'{grade_col}为D的企业获得贷款')


def main():
    strategy1 = pd.read_csv(OUTPUT_DIR / 'strategy_附件1.csv')
    strategy2 = pd.read_csv(OUTPUT_DIR / 'strategy_附件2.csv')
    strategy3 = pd.read_csv(OUTPUT_DIR / 'strategy_附件2_疫情调整.csv')
    _check_allocation(strategy1, '贷款额度_万元', BUDGET_PROBLEM_1, '风险等级')
    _check_allocation(strategy2, '贷款额度_万元', BUDGET_PROBLEM_2, '风险等级')
    _check_allocation(strategy3, '调整后额度', BUDGET_PROBLEM_2, '调整后等级')
    if (strategy1.loc[strategy1['信誉评级'].eq('D'), '贷款额度_万元'] > 0).any():
        raise AssertionError('银行原始D级企业获得贷款')

    workbook_check = validate_workbook(OUTPUT_DIR / '信贷决策建模结果.xlsx')
    if workbook_check['errors'] or workbook_check['formula_count'] == 0:
        raise AssertionError(f'Excel校验失败: {workbook_check}')
    with (OUTPUT_DIR / 'report_metrics.json').open(encoding='utf-8') as handle:
        metrics = json.load(handle)
    if metrics['validation']['roc_auc'] <= 0.5:
        raise AssertionError('样本外ROC-AUC未优于随机排序')

    print('核验通过')
    print(f"  ROC-AUC: {metrics['validation']['roc_auc']:.4f}")
    print(f"  三问预算: {strategy1['贷款额度_万元'].sum():.0f} / {strategy2['贷款额度_万元'].sum():.0f} / {strategy3['调整后额度'].sum():.0f} 万元")
    print(f"  原始D级获贷: {metrics['original_d_loan_count']} 家")
    print(f"  Excel公式: {workbook_check['formula_count']} 个，结构错误: {len(workbook_check['errors'])}")


if __name__ == '__main__':
    main()
