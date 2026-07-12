"""
主控脚本 - 中小微企业信贷决策建模
2020年数学建模国赛C题完整解答

流程:
  Phase 1: 数据加载与SQLite入库
  Phase 2: 特征工程
  Phase 3: 风险评估模型 (AHP+熵权+TOPSIS)
  Phase 4: 信贷策略制定
  Phase 5: 突发因素(新冠疫情)调整
  Phase 6: 灵敏度分析与可视化
"""
import os
import sys
import numpy as np
import pandas as pd

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_all_data
from feature_engineer import build_all_features
from risk_model import risk_model_pipeline
from credit_strategy import credit_strategy_pipeline
from visualize import create_all_visualizations
from sensitivity import run_sensitivity


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = ' '.join(str(a) for a in args)
        safe_text = text.encode('ascii', errors='replace').decode('ascii')
        print(safe_text, **kwargs)


def export_to_excel(df_scores_1, df_scores_2, strategy_1, strategy_2, strategy_2_adj, output_path):
    """将结果导出到Excel"""
    safe_print(f"\n[导出] 保存结果到 Excel: {os.path.basename(output_path)}")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: 附件1评分
        if df_scores_1 is not None:
            cols1 = [c for c in ['企业代号', '企业名称', 'TOPSIS得分', '排名', '风险等级',
                                 '信誉评级', '是否违约'] if c in df_scores_1.columns]
            df_scores_1[cols1].to_excel(writer, sheet_name='附件1_风险评分', index=False)

        # Sheet 2: 附件2评分
        if df_scores_2 is not None:
            cols2 = [c for c in ['企业代号', '企业名称', 'TOPSIS得分', '排名', '风险等级']
                    if c in df_scores_2.columns]
            df_scores_2[cols2].to_excel(writer, sheet_name='附件2_风险评分', index=False)

        # Sheet 3: 问题1信贷策略
        if strategy_1 is not None:
            sc1 = [c for c in ['企业代号', '企业名称', '风险等级', 'TOPSIS得分',
                               '贷款额度_万元', '贷款年利率', '预计流失率']
                  if c in strategy_1.columns]
            strategy_1[sc1].to_excel(writer, sheet_name='问题1_信贷策略', index=False)

        # Sheet 4: 问题2信贷策略
        if strategy_2 is not None:
            sc2 = [c for c in ['企业代号', '企业名称', '风险等级', 'TOPSIS得分',
                               '贷款额度_万元', '贷款年利率', '预计流失率']
                  if c in strategy_2.columns]
            strategy_2[sc2].to_excel(writer, sheet_name='问题2_信贷策略', index=False)

        # Sheet 5: 问题3调整后策略
        if strategy_2_adj is not None:
            sc3 = [c for c in ['企业代号', '企业名称', '行业类别', '风险等级', '调整后等级',
                               'TOPSIS得分', '调整后得分', '贷款额度_万元', '调整后额度',
                               '贷款年利率', '调整后利率', '冲击系数']
                  if c in strategy_2_adj.columns]
            strategy_2_adj[sc3].to_excel(writer, sheet_name='问题3_疫情调整策略', index=False)

    safe_print(f"  [OK] 已导出到: {output_path}")


def main():
    """完整建模流程"""
    print("=" * 70)
    print("  中小微企业信贷决策建模")
    print("  2020年数学建模国赛C题")
    print("=" * 70)

    # ========== Phase 1: 数据加载 ==========
    conn = load_all_data()

    # ========== Phase 2: 特征工程 ==========
    df_features_1, df_features_2 = build_all_features(conn)

    # 保存特征
    base_dir = os.path.dirname(__file__)
    df_features_1.to_csv(os.path.join(base_dir, 'features_附件1.csv'), encoding='utf-8-sig')
    df_features_2.to_csv(os.path.join(base_dir, 'features_附件2.csv'), encoding='utf-8-sig')

    # ========== Phase 3: 风险评估 ==========
    df_scores_1, df_scores_2, w_dict = risk_model_pipeline(df_features_1, df_features_2)

    # 保存评分
    df_scores_1.to_csv(os.path.join(base_dir, 'scores_附件1.csv'), encoding='utf-8-sig', index=False)
    df_scores_2.to_csv(os.path.join(base_dir, 'scores_附件2.csv'), encoding='utf-8-sig', index=False)

    # ========== Phase 4: 信贷策略 ==========
    strategy_1, strategy_2, strategy_2_adj, get_churn, coeffs = credit_strategy_pipeline(
        df_scores_1, df_scores_2
    )

    # 保存策略
    strategy_1.to_csv(os.path.join(base_dir, 'strategy_附件1.csv'), encoding='utf-8-sig', index=False)
    strategy_2.to_csv(os.path.join(base_dir, 'strategy_附件2.csv'), encoding='utf-8-sig', index=False)
    strategy_2_adj.to_csv(os.path.join(base_dir, 'strategy_附件2_疫情调整.csv'),
                          encoding='utf-8-sig', index=False)

    # ========== Phase 5: 突发因素（已在credit_strategy_pipeline中完成）==========

    # ========== Phase 6: 灵敏度分析 ==========
    # 从w_dict提取权重（对齐特征列）
    sensitivity_results = run_sensitivity(df_features_1, df_scores_1, w_dict)

    # ========== 可视化 ==========
    create_all_visualizations(
        df_scores_1, df_scores_2, strategy_2_adj, coeffs, w_dict
    )

    # ========== 导出Excel ==========
    output_excel = os.path.join(base_dir, '信贷决策建模结果.xlsx')
    export_to_excel(df_scores_1, df_scores_2, strategy_1, strategy_2, strategy_2_adj, output_excel)

    # ========== 汇总报告 ==========
    print("\n" + "=" * 70)
    print("  建模完成 - 结果汇总")
    print("=" * 70)

    print(f"\n问题1 (123家, 总额8000万):")
    if strategy_1 is not None:
        loan_count = (strategy_1['贷款额度_万元'] > 0).sum()
        total_loan = strategy_1['贷款额度_万元'].sum()
        print(f"  放贷: {loan_count}家, 总额: {total_loan:.0f}万元")
        print(f"  利率范围: {strategy_1[strategy_1['贷款额度_万元']>0]['贷款年利率'].min():.2%} - "
              f"{strategy_1[strategy_1['贷款额度_万元']>0]['贷款年利率'].max():.2%}")

    print(f"\n问题2 (302家, 总额1亿):")
    if strategy_2 is not None:
        loan_count2 = (strategy_2['贷款额度_万元'] > 0).sum()
        total_loan2 = strategy_2['贷款额度_万元'].sum()
        print(f"  放贷: {loan_count2}家, 总额: {total_loan2:.0f}万元")

    print(f"\n问题3 (疫情调整):")
    if strategy_2_adj is not None:
        changed = (strategy_2_adj['风险等级'] != strategy_2_adj['调整后等级']).sum()
        print(f"  等级调整: {changed}家")
        for ind in ['严重负面', '中度负面', '正面/轻影响']:
            mask = strategy_2_adj['行业类别'] == ind
            if mask.any():
                adj_loan = strategy_2_adj.loc[mask, '调整后额度'].mean()
                print(f"  {ind}: 均额{adj_loan:.1f}万元")

    conn.close()
    print("\n" + "=" * 70)
    print("  所有结果已保存到 信贷决策建模/ 目录")
    print("=" * 70)


if __name__ == '__main__':
    main()
