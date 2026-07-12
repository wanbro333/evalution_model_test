"""
结果数据合理性全面核查
检查：预算利用率、流失率合理性、评分异常企业、行业分类准确性、疫情调整合理性
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import sqlite3

BASE = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE, 'credit_data.db')

def safe_print(*args, **kwargs):
    try: print(*args, **kwargs)
    except: print(' '.join(str(a) for a in args))


def check1_budget_utilization(strategy_1, strategy_2):
    """核查1: 预算利用率异常"""
    safe_print("=" * 60)
    safe_print("[核查1] 预算利用率分析")
    safe_print("=" * 60)

    for name, df, budget in [("问题1", strategy_1, 8000), ("问题2", strategy_2, 10000)]:
        df_loan = df[df['贷款额度_万元'] > 0]
        total = df_loan['贷款额度_万元'].sum()
        n_firms = len(df_loan)
        utilization = total / budget * 100

        safe_print(f"\n{name} (预算{budget}万):")
        safe_print(f"  放贷{n_firms}家, 总额{total:.0f}万, 利用率{utilization:.1f}%")
        safe_print(f"  额度范围: {df_loan['贷款额度_万元'].min():.0f}-{df_loan['贷款额度_万元'].max():.0f}万")
        safe_print(f"  中位数: {df_loan['贷款额度_万元'].median():.0f}万")

        # 检查有多少企业被限制在100万上限
        capped = (df_loan['贷款额度_万元'] == 100).sum()
        floored = (df_loan['贷款额度_万元'] == 10).sum()
        safe_print(f"  上限100万: {capped}家, 下限10万: {floored}家")

        # 检查浪费：额度上限导致预算用不完
        if capped > 0:
            overshoot = (df_loan[df_loan['贷款额度_万元'] == 100]['初始额度'] if '初始额度' in df_loan.columns else None)
            safe_print(f"  说明: {capped}家企业被限制在100万上限, 导致部分预算无法分配")

        if utilization < 95:
            safe_print(f"  [ISSUE] 预算利用率偏低({utilization:.1f}%), 原因: 10-100万约束+均分机制")


def check2_churn_rate_reasonableness():
    """核查2: 流失率数据是否合理"""
    safe_print("\n" + "=" * 60)
    safe_print("[核查2] 流失率数据合理性")
    safe_print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM rate_churn", conn)
    conn.close()

    safe_print(f"  数据点: {len(df)}行")
    safe_print(f"  利率范围: {df['贷款年利率'].min()*100:.1f}% ~ {df['贷款年利率'].max()*100:.1f}%")

    for col in ['客户流失率_A', '客户流失率_B', '客户流失率_C']:
        vals = df[col].values
        safe_print(f"\n  {col}:")
        safe_print(f"    范围: {vals.min()*100:.1f}% ~ {vals.max()*100:.1f}%")

        # 检查异常：4%利率的流失率应为0
        rate_4_mask = df['贷款年利率'] == 0.04
        if rate_4_mask.any():
            churn_at_4 = df.loc[rate_4_mask, col].values[0]
            safe_print(f"    利率4%时流失率: {churn_at_4*100:.1f}% {'(应为0!)' if churn_at_4 > 0.01 else '(正确)'}")

        # 检查单调性：利率越高流失率应该越高
        sorted_vals = df.sort_values('贷款年利率')[col].values
        is_monotonic = all(sorted_vals[i] <= sorted_vals[i+1] for i in range(len(sorted_vals)-1))
        safe_print(f"    单调递增: {'是' if is_monotonic else '[ISSUE] 不是!'}")

        # 检查15%利率时流失率应接近1
        rate_15_mask = df['贷款年利率'] == 0.15
        if rate_15_mask.any():
            churn_at_15 = df.loc[rate_15_mask, col].values[0]
            safe_print(f"    利率15%时流失率: {churn_at_15*100:.1f}%")
            if churn_at_15 < 0.8:
                safe_print(f"    [ISSUE] 15%利率流失率偏低, 应在80%以上")


def check3_score_anomalies(df_scores_1, df_features_1):
    """核查3: 查找评分异常的企业"""
    safe_print("\n" + "=" * 60)
    safe_print("[核查3] 评分异常企业排查")
    safe_print("=" * 60)

    # 3a. 原评级D但模型给A的 (假阴性风险)
    safe_print("\n  原评级D但模型给高分(A/B)的企业 (潜在风险误判):")
    d_mask = df_scores_1['信誉评级'] == 'D'
    high_mask = df_scores_1['风险等级'].isin(['A', 'B'])
    anomalies = df_scores_1[d_mask & high_mask]
    if len(anomalies) > 0:
        for _, row in anomalies.iterrows():
            safe_print(f"    {row['企业代号']}: 原D级->模型{row['风险等级']}级, TOPSIS={row['TOPSIS得分']:.4f}")
            # 检查该企业的特征
            if df_features_1 is not None and row['企业代号'] in df_features_1.index:
                feats = df_features_1.loc[row['企业代号']]
                safe_print(f"      收入={feats.get('销项总金额',0)/1e4:.0f}万, "
                           f"利润={feats.get('利润',0)/1e4:.0f}万, "
                           f"利润率={feats.get('利润率',0)*100:.1f}%, "
                           f"销项有效率={feats.get('销项有效率',0)*100:.1f}%")
    else:
        safe_print("    无")

    # 3b. 原评级A但模型给D的 (假阳性)
    safe_print("\n  原评级A但模型给低分(C/D)的企业 (可能过度保守):")
    a_mask = df_scores_1['信誉评级'] == 'A'
    low_mask = df_scores_1['风险等级'].isin(['C', 'D'])
    anomalies2 = df_scores_1[a_mask & low_mask]
    if len(anomalies2) > 0:
        for _, row in anomalies2.iterrows():
            safe_print(f"    {row['企业代号']}: 原A级->模型{row['风险等级']}级, TOPSIS={row['TOPSIS得分']:.4f}")
    else:
        safe_print("    无")

    # 3c. 得分分布检查
    safe_print(f"\n  TOPSIS得分分布:")
    safe_print(f"    Mean={df_scores_1['TOPSIS得分'].mean():.4f}, "
               f"Std={df_scores_1['TOPSIS得分'].std():.4f}")
    safe_print(f"    Skewness={df_scores_1['TOPSIS得分'].skew():.4f} "
               f"(<0=左偏/高分多, >0=右偏/低分多)")


def check4_industry_classification(strategy_2_adj):
    """核查4: 行业分类准确性"""
    safe_print("\n" + "=" * 60)
    safe_print("[核查4] 行业分类准确性抽查")
    safe_print("=" * 60)

    for ind in ['严重负面', '中度负面', '正面/轻影响', '其他']:
        mask = strategy_2_adj['行业类别'] == ind
        if mask.any():
            samples = strategy_2_adj[mask].head(5)
            names = samples['企业名称'].tolist() if '企业名称' in samples.columns else []
            safe_print(f"\n  {ind} ({mask.sum()}家), 抽样:")
            for n in names[:5]:
                safe_print(f"    {str(n)[:50]}")


def check5_emergency_reasonableness(strategy_2, strategy_2_adj):
    """核查5: 疫情调整合理性"""
    safe_print("\n" + "=" * 60)
    safe_print("[核查5] 疫情调整合理性")
    safe_print("=" * 60)

    # 检查严重负面行业是否真的得到了更严格的对待
    for ind in ['严重负面', '中度负面', '正面/轻影响']:
        mask = strategy_2_adj['行业类别'] == ind
        if mask.any():
            orig_loan = strategy_2_adj.loc[mask, '贷款额度_万元'].mean()
            adj_loan = strategy_2_adj.loc[mask, '调整后额度'].mean()
            orig_rate = strategy_2_adj.loc[mask, '贷款年利率'].mean()
            adj_rate = strategy_2_adj.loc[mask, '调整后利率'].mean()
            grade_changes = (strategy_2_adj.loc[mask, '风险等级'] !=
                             strategy_2_adj.loc[mask, '调整后等级']).sum()

            safe_print(f"\n  {ind}:")
            safe_print(f"    额度变化: {orig_loan:.1f} -> {adj_loan:.1f}万 ({'+'if adj_loan>orig_loan else ''}{adj_loan-orig_loan:.1f})")
            safe_print(f"    利率变化: {orig_rate*100:.2f}% -> {adj_rate*100:.2f}%")
            safe_print(f"    等级调整: {grade_changes}/{mask.sum()}家")

            # 合理性判断
            if ind == '严重负面' and adj_loan >= orig_loan:
                safe_print(f"    [ISSUE] 严重负面行业额度应下降, 但{'上升' if adj_loan>orig_loan else '不变'}了")
            if ind == '正面/轻影响' and adj_rate >= orig_rate:
                safe_print(f"    [ISSUE] 正面行业利率应下降, 但{'上升' if adj_rate>orig_rate else '不变'}了")


def check6_feature_outliers(df_features_1):
    """核查6: 特征极端值分析"""
    safe_print("\n" + "=" * 60)
    safe_print("[核查6] 特征极端值分析")
    safe_print("=" * 60)

    # 利润率极端值
    if '利润率' in df_features_1.columns:
        extreme_profit = df_features_1[df_features_1['利润率'].abs() > 0.8]
        safe_print(f"\n  利润率极端(|>80%|): {len(extreme_profit)}家")
        if len(extreme_profit) > 0:
            safe_print(f"    这可能是因为: 新企业(成本极低/收入正常) 或 数据时间窗口不一致")

    # 负利润
    if '利润' in df_features_1.columns:
        neg_profit = (df_features_1['利润'] < 0).sum()
        safe_print(f"  负利润企业: {neg_profit}家 ({neg_profit/len(df_features_1)*100:.1f}%)")
        if neg_profit > len(df_features_1) * 0.3:
            safe_print(f"    [NOTE] 超过30%企业亏损, 符合中小微企业实际情况")

    # 销项有效率异常低
    if '销项有效率' in df_features_1.columns:
        low_eff = (df_features_1['销项有效率'] < 0.5).sum()
        safe_print(f"  销项有效率<50%: {low_eff}家")
        if low_eff > 0:
            worst = df_features_1.nsmallest(3, '销项有效率')[['销项有效率', '销项作废率']]
            safe_print(f"    最差3家: \n{worst.to_string()}")


def check7_credit_rating_direction(df_scores_1):
    """核查7: 模型评级vs原评级的一致性"""
    safe_print("\n" + "=" * 60)
    safe_print("[核查7] 模型评级与银行评级一致性")
    safe_print("=" * 60)

    ct = pd.crosstab(df_scores_1['信誉评级'], df_scores_1['风险等级'])
    safe_print(f"\n{ct.to_string()}")

    # 计算一致率 (A->A, B->B, C->C, D->D)
    correct = 0
    for grade in ['A', 'B', 'C', 'D']:
        mask = df_scores_1['信誉评级'] == grade
        if mask.any():
            correct += (df_scores_1.loc[mask, '风险等级'] == grade).sum()
    accuracy = correct / len(df_scores_1)
    safe_print(f"\n  完全一致率: {accuracy:.1%}")
    safe_print(f"  [NOTE] 不一致是正常的: 模型基于发票数据(客观), 银行评级基于人工判断(主观)")

    # 计算相邻一致率 (差1级以内)
    rank_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    orig_rank = df_scores_1['信誉评级'].map(rank_map)
    model_rank = df_scores_1['风险等级'].map(rank_map)
    adjacent = (abs(orig_rank - model_rank) <= 1).sum()
    safe_print(f"  相邻一致率(差≤1级): {adjacent/len(df_scores_1):.1%}")


def main():
    safe_print("=" * 70)
    safe_print("  结果数据合理性全面核查")
    safe_print("=" * 70)

    # 加载结果
    OUT = os.path.join(BASE, 'output')
    s1 = pd.read_csv(os.path.join(OUT, 'scores_附件1.csv'))
    s2 = pd.read_csv(os.path.join(OUT, 'scores_附件2.csv'))
    st1 = pd.read_csv(os.path.join(OUT, 'strategy_附件1.csv'))
    st2 = pd.read_csv(os.path.join(OUT, 'strategy_附件2.csv'))
    st2_adj = pd.read_csv(os.path.join(OUT, 'strategy_附件2_疫情调整.csv'))

    # 加载特征
    f1 = pd.read_csv(os.path.join(OUT, 'features_附件1.csv'), index_col='企业代号')

    # 执行7项核查
    check1_budget_utilization(st1, st2)
    check2_churn_rate_reasonableness()
    check3_score_anomalies(s1, f1)
    check4_industry_classification(st2_adj)
    check5_emergency_reasonableness(st2, st2_adj)
    check6_feature_outliers(f1)
    check7_credit_rating_direction(s1)

    safe_print("\n" + "=" * 70)
    safe_print("  核查完成")
    safe_print("=" * 70)


if __name__ == '__main__':
    main()
