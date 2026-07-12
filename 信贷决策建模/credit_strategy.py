"""
信贷策略与突发因素调整模块
内容：
  1. 利率-流失率关系建模（附件3）
  2. 信贷额度与利率分配策略
  3. 突发因素（新冠疫情）影响调整
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), 'credit_data.db')


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = ' '.join(str(a) for a in args)
        safe_text = text.encode('ascii', errors='replace').decode('ascii')
        print(safe_text, **kwargs)


# ========== 1. 利率-流失率建模 ==========
def fit_rate_churn_model():
    """
    从附件3拟合利率与客户流失率的关系
    对A/B/C三个信誉等级分别拟合多项式
    返回: 拟合函数字典
    """
    safe_print("\n" + "=" * 60)
    safe_print("阶段4.1: 利率-流失率关系建模")
    safe_print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM rate_churn", conn)
    conn.close()

    rates = df['贷款年利率'].values
    churn_A = df['客户流失率_A'].values
    churn_B = df['客户流失率_B'].values
    churn_C = df['客户流失率_C'].values

    safe_print(f"  数据点: {len(rates)}个, 利率范围 [{rates.min():.2%}, {rates.max():.2%}]")

    # 多项式拟合 (degree=2)
    coeffs = {}
    for grade, churn in [('A', churn_A), ('B', churn_B), ('C', churn_C)]:
        # 二次多项式: y = a*x^2 + b*x + c
        coeff = np.polyfit(rates, churn, 2)
        coeffs[grade] = coeff

        # 计算R^2
        y_pred = np.polyval(coeff, rates)
        ss_res = np.sum((churn - y_pred) ** 2)
        ss_tot = np.sum((churn - np.mean(churn)) ** 2)
        r2 = 1 - ss_res / ss_tot

        safe_print(f"  等级{grade}: loss = {coeff[0]:.2f}*r^2 + {coeff[1]:.2f}*r + {coeff[2]:.4f}, R^2={r2:.4f}")

        # 检查是否满足R^2>0.9
        if r2 < 0.9:
            safe_print(f"  [WARNING] 等级{grade} R^2={r2:.4f} < 0.9, 考虑使用更高阶拟合")

    def get_churn_rate(rate, grade):
        """根据利率和信誉等级预测流失率"""
        if grade not in coeffs:
            return 0.0
        churn = np.polyval(coeffs[grade], rate)
        return max(0.0, min(1.0, churn))  # 限制在[0,1]

    safe_print("\n  示例: 利率=8%时各等级流失率:")
    for g in ['A', 'B', 'C']:
        safe_print(f"    等级{g}: {get_churn_rate(0.08, g):.4f}")

    return get_churn_rate, coeffs


# ========== 2. 信贷策略制定 ==========
def allocate_credit(df_scores, total_budget=10000, get_churn_rate=None):
    """
    根据风险评分分配信贷额度和利率
    原则:
      - D级不放贷
      - 额度按TOPSIS得分比例分配，限制在10-100万
      - 利率按风险等级阶梯: A=4-6%, B=6-9%, C=9-12%, D=不贷
      - 考虑客户流失率

    参数:
      df_scores: 含TOPSIS得分和风险等级的DataFrame
      total_budget: 年度信贷总额(万元), 默认1亿=10000万
      get_churn_rate: 流失率计算函数
    """
    safe_print("\n" + "=" * 60)
    safe_print("阶段4.2: 信贷策略制定")
    safe_print(f"  年度信贷总额: {total_budget}万元")
    safe_print("=" * 60)

    df = df_scores.copy()

    # 1. 排除D级或得分极低的企业
    df_eligible = df[df['风险等级'] != 'D'].copy()
    n_eligible = len(df_eligible)
    safe_print(f"  放贷企业数: {n_eligible}/{len(df)} ({n_eligible/len(df)*100:.1f}%)")

    # 2. 按TOPSIS得分分配额度权重
    scores = df_eligible['TOPSIS得分'].values
    total_score = scores.sum()
    if total_score == 0:
        total_score = 1e-12

    # 3. 基于得分比例计算初始额度
    df_eligible['初始额度'] = (scores / total_score) * total_budget

    # 4. 限制在10-100万范围
    loan_min, loan_max = 10, 100
    df_eligible['贷款额度_万元'] = df_eligible['初始额度'].clip(loan_min, loan_max)

    # 5. 调整总额到约束内
    actual_total = df_eligible['贷款额度_万元'].sum()
    if actual_total > total_budget:
        # 超出预算，按比例缩减(不低于10万)
        scale = total_budget / actual_total
        df_eligible['贷款额度_万元'] = np.maximum(
            loan_min,
            df_eligible['贷款额度_万元'] * scale
        )
        # 迭代调整
        for _ in range(10):
            actual_total = df_eligible['贷款额度_万元'].sum()
            if abs(actual_total - total_budget) / total_budget < 0.01:
                break
            if actual_total > total_budget:
                excess = actual_total - total_budget
                # 从额度最高的企业中削减
                adjustable = df_eligible['贷款额度_万元'] > loan_min
                if adjustable.any():
                    total_adjustable = df_eligible.loc[adjustable, '贷款额度_万元'].sum() - \
                                       adjustable.sum() * loan_min
                    if total_adjustable > 0:
                        reduction_factor = 1 - min(1, excess / total_adjustable)
                        df_eligible.loc[adjustable, '贷款额度_万元'] = np.maximum(
                            loan_min,
                            df_eligible.loc[adjustable, '贷款额度_万元'] * reduction_factor
                        )

    # 6. 利率分配
    base_rates = {'A': 0.04, 'B': 0.065, 'C': 0.10}
    df_eligible['贷款年利率'] = df_eligible['风险等级'].map(base_rates).fillna(0.10)

    # 7. 根据得分微调利率(得分高 = 利率优惠)
    # 在基准利率上下浮动±1%
    score_min, score_max = scores.min(), scores.max()
    if score_max > score_min:
        # 标准化得分到[0, 1]
        score_norm = (scores - score_min) / (score_max - score_min)
        # 利率调整: 得分高的减利率, 得分低的加利率 (基准利率 ± 0.01)
        rate_adjust = (0.5 - score_norm) * 0.02  # 范围[-1%, +1%]
        df_eligible['贷款年利率'] = df_eligible['贷款年利率'] + rate_adjust
        df_eligible['贷款年利率'] = df_eligible['贷款年利率'].clip(0.04, 0.15)

    # 8. 客户流失率估计
    if get_churn_rate:
        churn_rates = []
        for _, row in df_eligible.iterrows():
            churn = get_churn_rate(row['贷款年利率'], row['风险等级'])
            churn_rates.append(churn)
        df_eligible['预计流失率'] = churn_rates

    # 9. D级企业不放贷
    df_d = df[df['风险等级'] == 'D'].copy()
    df_d['贷款额度_万元'] = 0
    df_d['贷款年利率'] = 0
    if get_churn_rate:
        df_d['预计流失率'] = 0

    # 合并结果
    df_result = pd.concat([df_eligible, df_d], ignore_index=True)

    # 统计
    safe_print(f"\n  额度统计:")
    safe_print(f"    总额度: {df_result['贷款额度_万元'].sum():.0f}万元 (预算{total_budget}万)")
    safe_print(f"    平均额度: {df_eligible['贷款额度_万元'].mean():.1f}万元")
    safe_print(f"    最小/最大额度: {loan_min}/{loan_max}万元")

    safe_print(f"\n  利率统计:")
    for grade in ['A', 'B', 'C']:
        gmask = df_result['风险等级'] == grade
        if gmask.any():
            rates_g = df_result.loc[gmask, '贷款年利率']
            safe_print(f"    等级{grade}: 利率{rates_g.min():.2%}~{rates_g.max():.2%}, 均值{rates_g.mean():.2%}")

    # 预期收益
    expected_revenue = (df_result['贷款额度_万元'] * df_result['贷款年利率']).sum()
    safe_print(f"\n  预期利息收入: {expected_revenue:.0f}万元/年")

    if '预计流失率' in df_result.columns:
        avg_churn = df_eligible['预计流失率'].mean()
        safe_print(f"  平均预计流失率: {avg_churn:.2%}")

    return df_result


# ========== 3. 突发因素调整 ==========
def classify_industry(name):
    """
    从企业名称推断行业类别
    基于关键词匹配
    """
    if not isinstance(name, str):
        return '其他'

    name = name.replace('***', '').replace('*', '').strip()

    # 严重受疫情影响(负面)
    severe_negative = ['旅游', '旅行', '酒店', '宾馆', '住宿', '餐饮', '餐厅',
                       '饭店', '影院', '电影', '娱乐城', 'KTV', '演出', '会展',
                       '航空', '客运', '旅行社']
    for kw in severe_negative:
        if kw in name:
            return '严重负面'

    # 中度受疫情影响
    moderate_negative = ['建筑', '建材', '装修', '装饰', '房地产', '房产',
                         '批发', '零售', '商贸', '商店', '超市', '商场',
                         '制造', '加工', '生产', '工厂', '纺织', '服装',
                         '运输', '物流', '货运', '快递',
                         '教育', '培训', '学校',
                         '汽车', '汽配', '车行']
    for kw in moderate_negative:
        if kw in name:
            return '中度负面'

    # 轻度影响/正面
    positive_kw = ['医疗', '医药', '药品', '生物', '卫生', '健康',
                   '科技', '技术', '信息', '软件', '网络', '数据',
                   '电子', '电商', '在线', '互联网', '数字',
                   '通信', '通讯', '计算机']
    for kw in positive_kw:
        if kw in name:
            return '正面/轻影响'

    # 默认：中性
    return '其他'


def apply_emergency_adjustment(df_scores, df_strategy, get_churn_rate=None):
    """
    问题3: 考虑突发因素（新冠疫情）调整信贷策略

    调整原则:
      - 严重负面行业: 风险得分下调20%, 利率上浮2%, 额度削减
      - 中度负面行业: 风险得分下调10%, 利率上浮1%
      - 正面行业: 风险得分上调5%, 利率优惠1%
    """
    safe_print("\n" + "=" * 60)
    safe_print("阶段5: 突发因素（新冠疫情）信贷调整")
    safe_print("=" * 60)

    df = df_strategy.copy()

    # 1. 行业分类
    if '企业名称' not in df.columns:
        safe_print("  [WARNING] 无企业名称信息, 无法进行行业分类")
        safe_print("  随机分配行业以演示调整效果")
        industries = ['严重负面', '中度负面', '其他', '正面/轻影响']
        df['行业类别'] = np.random.choice(industries, size=len(df),
                                         p=[0.15, 0.30, 0.40, 0.15])
    else:
        df['行业类别'] = df['企业名称'].apply(classify_industry)

    safe_print("\n  行业分布:")
    industry_counts = df['行业类别'].value_counts()
    for ind, cnt in industry_counts.items():
        safe_print(f"    {ind}: {cnt}家 ({cnt/len(df)*100:.1f}%)")

    # 2. 行业冲击系数
    shock_factors = {
        '严重负面': 1.30,      # 风险增加30%
        '中度负面': 1.15,      # 风险增加15%
        '其他': 1.00,          # 无变化
        '正面/轻影响': 0.90,   # 风险降低10%
    }

    # 3. 调整风险得分
    df['冲击系数'] = df['行业类别'].map(shock_factors).fillna(1.0)
    df['调整后得分'] = df['TOPSIS得分'] / df['冲击系数']  # 得分除以系数 = 风险调整

    # 确保得分在有效范围内
    df['调整后得分'] = df['调整后得分'].clip(0, 1)

    # 4. 重新排序和分级
    df = df.sort_values('调整后得分', ascending=False).reset_index(drop=True)
    n = len(df)
    df['调整后等级'] = 'D'
    df.loc[df.index < n * 0.80, '调整后等级'] = 'C'
    df.loc[df.index < n * 0.50, '调整后等级'] = 'B'
    df.loc[df.index < n * 0.25, '调整后等级'] = 'A'

    # 5. 重新计算利率
    base_rates = {'A': 0.04, 'B': 0.065, 'C': 0.10}
    df['调整后利率'] = df['调整后等级'].map(base_rates).fillna(0.10)
    # 加上行业影响的利率调整
    rate_adj_map = {'严重负面': 0.02, '中度负面': 0.01, '其他': 0.0, '正面/轻影响': -0.01}
    df['行业利率调整'] = df['行业类别'].map(rate_adj_map).fillna(0.0)
    df['调整后利率'] = df['调整后利率'] + df['行业利率调整']
    df['调整后利率'] = df['调整后利率'].clip(0.04, 0.15)

    # 6. 调整后的额度分配
    eligible = df[df['调整后等级'] != 'D']
    total_budget = 10000  # 问题3也是1亿元

    adj_scores = eligible['调整后得分'].values
    total_adj_score = adj_scores.sum()
    if total_adj_score > 0:
        eligible_copy = eligible.copy()
        eligible_copy['调整后额度'] = (adj_scores / total_adj_score) * total_budget
        eligible_copy['调整后额度'] = eligible_copy['调整后额度'].clip(10, 100)

        df['调整后额度'] = 0.0
        df.loc[eligible.index, '调整后额度'] = eligible_copy['调整后额度'].values
    else:
        df['调整后额度'] = df['贷款额度_万元']

    # 7. 对比分析
    safe_print("\n  调整前后对比:")
    safe_print(f"    放贷企业数: {df['贷款额度_万元'].gt(0).sum()} -> {df['调整后额度'].gt(0).sum()}")
    safe_print(f"    总额度: {df['贷款额度_万元'].sum():.0f}万 -> {df['调整后额度'].sum():.0f}万")
    safe_print(f"    平均利率: {df[df['贷款额度_万元']>0]['贷款年利率'].mean():.2%} -> {df[df['调整后额度']>0]['调整后利率'].mean():.2%}")

    # 行业维度分析
    safe_print("\n  各行业调整后平均额度:")
    for ind in ['严重负面', '中度负面', '其他', '正面/轻影响']:
        mask = df['行业类别'] == ind
        if mask.any():
            avg_loan = df.loc[mask, '调整后额度'].mean()
            avg_rate = df.loc[mask, '调整后利率'].mean()
            safe_print(f"    {ind}: 均额{avg_loan:.1f}万, 均利率{avg_rate:.2%}")

    # 等级变化
    changed = (df['风险等级'] != df['调整后等级']).sum()
    safe_print(f"\n  等级变化: {changed}家企业等级发生调整")

    return df


def credit_strategy_pipeline(df_scores_1, df_scores_2):
    """信贷策略完整流程"""
    # 拟合利率-流失率模型
    get_churn, coeffs = fit_rate_churn_model()

    # 问题1: 123家企业 (总额=123*平均额度假设)
    # 题目说"年度信贷总额固定", 假设为123家企业 * 60万 = 7380万 ≈ 8000万
    total_1 = 8000  # 万元
    safe_print(f"\n>>> 问题1信贷策略: 总额{total_1}万元 <<<")
    strategy_1 = allocate_credit(df_scores_1, total_1, get_churn)

    # 问题2: 302家企业, 总额1亿元
    total_2 = 10000  # 万元 (1亿)
    safe_print(f"\n>>> 问题2信贷策略: 总额{total_2}万元(1亿) <<<")
    strategy_2 = allocate_credit(df_scores_2, total_2, get_churn)

    # 问题3: 考虑突发因素调整
    safe_print(f"\n>>> 问题3: 考虑疫情影响的信贷调整 <<<")
    strategy_2_adjusted = apply_emergency_adjustment(df_scores_2, strategy_2, get_churn)

    return strategy_1, strategy_2, strategy_2_adjusted, get_churn, coeffs


if __name__ == '__main__':
    base = os.path.dirname(__file__)

    # 加载评分数据
    s1 = pd.read_csv(os.path.join(base, 'scores_附件1.csv'))
    s2 = pd.read_csv(os.path.join(base, 'scores_附件2.csv'))

    st1, st2, st2_adj, gc, cf = credit_strategy_pipeline(s1, s2)

    # 保存结果
    st1.to_csv(os.path.join(base, 'strategy_附件1.csv'), encoding='utf-8-sig', index=False)
    st2.to_csv(os.path.join(base, 'strategy_附件2.csv'), encoding='utf-8-sig', index=False)
    st2_adj.to_csv(os.path.join(base, 'strategy_附件2_疫情调整.csv'), encoding='utf-8-sig', index=False)
    safe_print("\n[OK] 信贷策略结果已保存")
