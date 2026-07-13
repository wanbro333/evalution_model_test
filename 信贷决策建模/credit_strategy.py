"""单调流失率、期望收益定价、MILP额度分配与疫情赔率调整。"""
import os
import re
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    BUDGET_PROBLEM_1,
    BUDGET_PROBLEM_2,
    DB_PATH,
    FUNDING_COST,
    LGD,
    LOAN_MAX,
    LOAN_MIN,
    SHOCK_FACTORS,
)
from risk_model import grade_from_pd


def pava(values):
    """最小二乘单调递增回归（pool-adjacent-violators）。"""
    levels, weights, counts = [], [], []
    for value in np.asarray(values, dtype=float):
        levels.append(float(value))
        weights.append(1.0)
        counts.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            weight = weights[-2] + weights[-1]
            level = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / weight
            count = counts[-2] + counts[-1]
            levels[-2:] = [level]
            weights[-2:] = [weight]
            counts[-2:] = [count]
    return np.concatenate([np.full(count, level) for level, count in zip(levels, counts)])


def fit_rate_churn_model(db_path=DB_PATH):
    connection = sqlite3.connect(db_path)
    df = pd.read_sql('SELECT * FROM rate_churn ORDER BY 贷款年利率', connection)
    connection.close()
    rates = df['贷款年利率'].to_numpy(float)
    order = np.argsort(rates)
    rates = rates[order]
    model = {'rates': rates.tolist(), 'curves': {}}
    diagnostics = {}
    for grade in ['A', 'B', 'C']:
        raw = df[f'客户流失率_{grade}'].to_numpy(float)[order]
        fitted = np.clip(pava(raw), 0, 1)
        model['curves'][grade] = fitted.tolist()
        diagnostics[grade] = {
            'raw_monotonic_violations': int(np.sum(np.diff(raw) < 0)),
            'rmse': float(np.sqrt(np.mean((raw - fitted) ** 2))),
            'churn_at_4pct': float(np.interp(0.04, rates, fitted)),
            'churn_at_15pct': float(np.interp(0.15, rates, fitted)),
        }
    print('\n阶段4.1：单调利率—流失率模型')
    for grade, info in diagnostics.items():
        print(f"  {grade}级: 原始下降点{info['raw_monotonic_violations']}个, 单调拟合RMSE={info['rmse']:.4f}")
    return model, diagnostics


def churn_rate(rate, grade, model):
    if grade not in model['curves']:
        return 1.0
    value = np.interp(float(rate), np.asarray(model['rates']), np.asarray(model['curves'][grade]))
    return float(np.clip(value, 0, 1))


def choose_rate(pd_value, grade, model, lgd=LGD, funding_cost=FUNDING_COST):
    rates = np.asarray(model['rates'], dtype=float)
    churn = np.array([churn_rate(rate, grade, model) for rate in rates])
    retention = 1 - churn
    unit_profit = retention * ((1 - pd_value) * rates - pd_value * lgd - funding_cost)
    best = int(np.argmax(unit_profit))
    return float(rates[best]), float(churn[best]), float(unit_profit[best])


def _solve_allocation(unit_profit, budget, loan_min=LOAN_MIN, loan_max=LOAN_MAX):
    unit_profit = np.asarray(unit_profit, dtype=float)
    n = len(unit_profit)
    if n == 0:
        raise ValueError('没有符合条件的放贷企业')
    if budget < loan_min - 1e-9 or budget > n * loan_max + 1e-9:
        raise ValueError(f'预算{budget:.2f}万元不可行；当前候选企业可行上限为{n * loan_max:.2f}万元')

    objective = np.r_[-unit_profit, np.zeros(n)]
    integrality = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    lower = np.zeros(2 * n)
    upper = np.r_[np.full(n, loan_max), np.ones(n)]

    matrix = np.zeros((2 * n + 1, 2 * n), dtype=float)
    constraint_lower = np.full(2 * n + 1, -np.inf)
    constraint_upper = np.zeros(2 * n + 1)
    for i in range(n):
        matrix[2 * i, i] = 1
        matrix[2 * i, n + i] = -loan_max
        matrix[2 * i + 1, i] = -1
        matrix[2 * i + 1, n + i] = loan_min
    matrix[-1, :n] = 1
    constraint_lower[-1] = budget
    constraint_upper[-1] = budget

    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, constraint_lower, constraint_upper),
        options={'time_limit': 60},
    )
    if not result.success:
        raise RuntimeError(f'信贷额度MILP求解失败: {result.message}')
    loans = result.x[:n]
    loans[np.abs(loans) < 1e-7] = 0
    return loans


def allocate_credit(df_scores, total_budget, churn_model, lgd=LGD, funding_cost=FUNDING_COST):
    df = df_scores.copy().reset_index(drop=True)
    predicted_d = df['风险等级'].eq('D')
    original_d = df['信誉评级'].eq('D') if '信誉评级' in df.columns else pd.Series(False, index=df.index)
    eligible_mask = ~(predicted_d | original_d)
    eligible = df.loc[eligible_mask].copy()

    rates, churns, units = [], [], []
    for row in eligible.itertuples(index=False):
        rate, churn, unit = choose_rate(float(getattr(row, '违约概率')), getattr(row, '风险等级'), churn_model, lgd, funding_cost)
        rates.append(rate)
        churns.append(churn)
        units.append(unit)
    eligible['贷款年利率'] = rates
    eligible['预计流失率'] = churns
    eligible['单位额度期望收益'] = units
    eligible['贷款额度_万元'] = _solve_allocation(units, float(total_budget))

    df['贷款额度_万元'] = 0.0
    df['贷款年利率'] = 0.0
    df['预计流失率'] = 0.0
    df['单位额度期望收益'] = 0.0
    for col in ['贷款额度_万元', '贷款年利率', '预计流失率', '单位额度期望收益']:
        df.loc[eligible.index, col] = eligible[col]

    df['禁贷原因'] = ''
    df.loc[predicted_d, '禁贷原因'] = '预测D级'
    df.loc[original_d, '禁贷原因'] = '银行原始D级'
    df.loc[predicted_d & original_d, '禁贷原因'] = '银行原始D级且预测D级'

    amount = df['贷款额度_万元'].to_numpy(float)
    rate = df['贷款年利率'].to_numpy(float)
    retention = 1 - df['预计流失率'].to_numpy(float)
    pd_value = df['违约概率'].to_numpy(float)
    df['客户留存率'] = np.where(amount > 0, retention, 0.0)
    df['预计实际放款额'] = amount * retention
    df['名义利息'] = amount * rate
    df['预计留存后利息'] = amount * retention * (1 - pd_value) * rate
    df['预计违约损失'] = amount * retention * pd_value * lgd
    df['预计资金成本'] = amount * retention * funding_cost
    df['期望净收益'] = df['预计留存后利息'] - df['预计违约损失'] - df['预计资金成本']
    df['预算强制负收益'] = (amount > 0) & (df['单位额度期望收益'] < 0)

    total = df['贷款额度_万元'].sum()
    if abs(total - total_budget) > 1e-4:
        raise AssertionError(f'额度合计{total}与预算{total_budget}不一致')
    if '信誉评级' in df.columns and (df.loc[df['信誉评级'].eq('D'), '贷款额度_万元'] > 0).any():
        raise AssertionError('存在银行原始D级企业获得贷款')
    print(f"  预算{total_budget:.0f}万元: 放贷{(amount > 0).sum()}家, 期望净收益{df['期望净收益'].sum():.2f}万元")
    return df


def classify_industry(name):
    if not isinstance(name, str) or re.fullmatch(r'\s*个体经营E\d+\s*', name):
        return '未知', 0.0
    cleaned = name.replace('***', '').replace('*', '').strip()
    keyword_groups = [
        ('严重负面', ['旅游', '旅行', '酒店', '宾馆', '住宿', '餐饮', '餐厅', '饭店', '影院', '电影', '娱乐', 'KTV', '演出', '会展', '航空', '客运']),
        ('中度负面', ['建筑', '建设', '建材', '装修', '装饰', '房地产', '房产', '批发', '零售', '商贸', '商店', '超市', '商场', '制造', '加工', '生产', '工厂', '纺织', '服装', '运输', '物流', '货运', '快递', '教育', '培训', '学校', '汽车', '汽配', '车行']),
        ('正面/轻影响', ['医疗', '医药', '药品', '生物', '卫生', '健康', '科技', '技术', '信息', '软件', '网络', '数据', '电子', '电商', '在线', '互联网', '数字', '通信', '通讯', '计算机']),
    ]
    for category, keywords in keyword_groups:
        if any(keyword in cleaned for keyword in keywords):
            return category, 0.9
    return '其他', 0.3


def adjust_probability_by_odds(probability, factor):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    odds = probability / (1 - probability)
    adjusted = odds * np.asarray(factor, dtype=float)
    return adjusted / (1 + adjusted)


def apply_emergency_adjustment(df_scores, base_strategy, churn_model, total_budget=BUDGET_PROBLEM_2):
    scores = df_scores.copy()
    classified = scores['企业名称'].apply(classify_industry)
    scores['行业类别'] = classified.map(lambda x: x[0])
    scores['行业识别置信度'] = classified.map(lambda x: x[1])
    scores['冲击系数'] = scores['行业类别'].map(SHOCK_FACTORS).fillna(1.0)
    scores['原违约概率'] = scores['违约概率']
    scores['原风险等级'] = scores['风险等级']
    scores['违约概率'] = adjust_probability_by_odds(scores['违约概率'], scores['冲击系数'])
    scores['风险等级'] = grade_from_pd(scores['违约概率'])

    adjusted = allocate_credit(scores, total_budget, churn_model)
    adjusted = adjusted.rename(columns={
        '违约概率': '调整后违约概率',
        '风险等级': '调整后等级',
        '贷款额度_万元': '调整后额度',
        '贷款年利率': '调整后利率',
        '预计流失率': '调整后预计流失率',
        '期望净收益': '调整后期望净收益',
    })
    original = base_strategy[['企业代号', '违约概率', '风险等级', '贷款额度_万元', '贷款年利率', '预计流失率', '期望净收益']].rename(columns={
        '违约概率': '原违约概率_策略',
        '风险等级': '原风险等级_策略',
        '贷款额度_万元': '原贷款额度',
        '贷款年利率': '原贷款利率',
        '预计流失率': '原预计流失率',
        '期望净收益': '原期望净收益',
    })
    adjusted = adjusted.merge(original, on='企业代号', how='left')
    if abs(adjusted['调整后额度'].sum() - total_budget) > 1e-4:
        raise AssertionError('疫情调整后预算不等于目标预算')
    print(f"  疫情调整: {(adjusted['原风险等级'] != adjusted['调整后等级']).sum()}家等级变化")
    return adjusted


def credit_strategy_pipeline(df_scores_1, df_scores_2):
    churn_model, churn_diagnostics = fit_rate_churn_model()
    print('\n阶段4.2：期望收益信贷优化')
    strategy1 = allocate_credit(df_scores_1, BUDGET_PROBLEM_1, churn_model)
    strategy2 = allocate_credit(df_scores_2, BUDGET_PROBLEM_2, churn_model)
    print('\n阶段5：疫情冲击赔率调整')
    strategy3 = apply_emergency_adjustment(df_scores_2, strategy2, churn_model)
    return strategy1, strategy2, strategy3, churn_model, churn_diagnostics


if __name__ == '__main__':
    from config import OUTPUT_DIR

    score1 = pd.read_csv(OUTPUT_DIR / 'scores_附件1.csv')
    score2 = pd.read_csv(OUTPUT_DIR / 'scores_附件2.csv')
    st1, st2, st3, _, _ = credit_strategy_pipeline(score1, score2)
    st1.to_csv(OUTPUT_DIR / 'strategy_附件1.csv', index=False, encoding='utf-8-sig')
    st2.to_csv(OUTPUT_DIR / 'strategy_附件2.csv', index=False, encoding='utf-8-sig')
    st3.to_csv(OUTPUT_DIR / 'strategy_附件2_疫情调整.csv', index=False, encoding='utf-8-sig')
