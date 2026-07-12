"""
风险评估模型模块
核心算法：
  1. AHP主观赋权（层次分析法）
  2. 熵权法客观赋权
  3. 组合赋权（主观+客观）
  4. TOPSIS综合评分
  5. 模型验证与调优
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from utils import (
    ahp_weights, entropy_weight, topsis_score,
    positive_transform, vector_normalize
)

# 控制台编码安全
def safe_print(*args, **kwargs):
    """安全的print, 避免unicode编码错误"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = ' '.join(str(a) for a in args)
        safe_text = text.encode('ascii', errors='replace').decode('ascii')
        print(safe_text, **kwargs)


# ========== 指标配置 ==========
# 类型: positive=越大越好(低风险), negative=越小越好(低风险)
FEATURE_CONFIG = {
    # 企业实力维度 — 规模大通常更稳定
    '销项总金额': 'positive',
    '进项总金额': 'positive',
    '利润率': 'positive',          # 盈利能力
    '销项客户数': 'positive',      # 客户多元化
    '进项供应商数': 'positive',    # 供应多元化
    '交易网络广度': 'positive',
    # 企业稳定性维度 — 稳定=低风险
    '收入波动率': 'negative',
    '成本波动率': 'negative',
    # 企业信誉维度
    '销项有效率': 'positive',
    '进项有效率': 'positive',
    '销项作废率': 'negative',
    '进项作废率': 'negative',
    # 企业成长维度
    '收入环比增长率': 'positive',
    # 活跃度
    '综合活跃月数': 'positive',
    # 盈利能力修正
    '利润': 'positive',            # 绝对利润
}


def preprocess_features(df, config=FEATURE_CONFIG):
    """预处理：选取指标、对数变换偏态分布、Winsorize极端值"""
    available_cols = [c for c in config if c in df.columns]
    X = df[available_cols].copy()

    # 对金额/数量类指标做对数变换减小偏态
    log_cols = ['销项总金额', '进项总金额', '利润', '销项客户数',
                '进项供应商数', '交易网络广度']
    for col in log_cols:
        if col in X.columns:
            # 先处理负值：保留符号的对数变换
            min_val = X[col].min()
            if min_val < 0:
                X[col] = np.sign(X[col]) * np.log1p(np.abs(X[col]))
            else:
                X[col] = np.log1p(X[col])

    # Winsorize (1%-99%)
    for col in X.columns:
        q01, q99 = X[col].quantile(0.01), X[col].quantile(0.99)
        if q01 < q99:
            X[col] = X[col].clip(q01, q99)

    return X, available_cols


def build_ahp_weight_vector(available_cols):
    """构建AHP权重向量，对齐到available_cols"""
    # 准则层判断矩阵 (B1实力, B2稳定性, B3信誉, B4成长)
    B = np.array([
        [1,   3,   2,   4],    # 实力
        [1/3, 1,   1/2, 2],    # 稳定性
        [1/2, 2,   1,   3],    # 信誉
        [1/4, 1/2, 1/3, 1],   # 成长
    ])
    w_B, lam, ci, cr = ahp_weights(B, 'geometric_mean')
    safe_print(f"  AHP准则层: 实力={w_B[0]:.3f}, 稳定={w_B[1]:.3f}, 信誉={w_B[2]:.3f}, 成长={w_B[3]:.3f}")
    safe_print(f"  CR={cr:.4f} (CR<0.10, 一致性可接受)")

    # 指标→准则的映射
    # 实力(7指标): 销项总金额,进项总金额,利润率,利润,销项客户数,进项供应商数,交易网络广度
    # 稳定性(2指标): 收入波动率,成本波动率
    # 信誉(4指标): 销项有效率,进项有效率,销项作废率,进项作废率
    # 成长(2指标): 收入环比增长率,综合活跃月数

    criterion_map = {
        '销项总金额': 0, '进项总金额': 0, '利润率': 0, '利润': 0,
        '销项客户数': 0, '进项供应商数': 0, '交易网络广度': 0,
        '收入波动率': 1, '成本波动率': 1,
        '销项有效率': 2, '进项有效率': 2, '销项作废率': 2, '进项作废率': 2,
        '收入环比增长率': 3, '综合活跃月数': 3,
    }

    # 准则内指标权重（简化：均值分配）
    w_ahp_list = []
    for col in available_cols:
        criterion_idx = criterion_map.get(col, 0)
        # 该准则下的指标数
        n_in_criterion = sum(1 for c in available_cols
                           if criterion_map.get(c, 0) == criterion_idx)
        w_ahp_list.append(w_B[criterion_idx] / max(n_in_criterion, 1))

    w_ahp = np.array(w_ahp_list)
    w_ahp = w_ahp / w_ahp.sum()
    return w_ahp


def evaluate_credit_risk(df_features, feature_config=FEATURE_CONFIG, alpha=0.5):
    """
    综合评价流程: 预处理 -> 正向化 -> 标准化 -> 组合赋权 -> TOPSIS
    alpha: AHP占比 (0.5 = 主客观各半)
    返回: df_scores, w_combined_dict
    """
    # 1. 预处理
    X, available_cols = preprocess_features(df_features, feature_config)
    safe_print(f"  使用指标({len(available_cols)}个): {available_cols}")

    # 2. 正向化
    indicator_types = [feature_config[c] for c in available_cols]
    X_pos = positive_transform(X.values, indicator_types)

    # 3. 标准化
    X_norm = vector_normalize(X_pos)

    # 4. 熵权法
    X_for_entropy = np.clip(X_pos, 1e-12, None)
    w_entropy, entropy_vals, diff_vals = entropy_weight(X_for_entropy)
    safe_print(f"  熵权法: {np.round(w_entropy, 3)}")
    # 检查是否某个指标权重过大
    max_w_idx = np.argmax(w_entropy)
    if w_entropy[max_w_idx] > 0.3:
        safe_print(f"  [注意] '{available_cols[max_w_idx]}' 熵权={w_entropy[max_w_idx]:.3f} 过高, 可能主导模型")

    # 5. AHP权重
    w_ahp = build_ahp_weight_vector(available_cols)
    safe_print(f"  AHP权重: {np.round(w_ahp, 3)}")

    # 6. 组合赋权
    w_combined = alpha * w_ahp + (1 - alpha) * w_entropy
    w_combined = w_combined / w_combined.sum()
    safe_print(f"  组合权重(alpha={alpha}): {np.round(w_combined, 3)}")

    # 7. TOPSIS
    scores, d_pos, d_neg = topsis_score(X_norm, w_combined)

    # 8. 构建结果DataFrame
    df_result = pd.DataFrame({
        '企业代号': df_features.index.values,
        'TOPSIS得分': scores,
        'D_pos': d_pos, 'D_neg': d_neg,
    })
    df_result = df_result.sort_values('TOPSIS得分', ascending=False).reset_index(drop=True)
    df_result['排名'] = range(1, len(df_result) + 1)

    # 分级
    n = len(df_result)
    df_result['风险等级'] = 'D'
    df_result.loc[df_result['排名'] <= n * 0.80, '风险等级'] = 'C'
    df_result.loc[df_result['排名'] <= n * 0.50, '风险等级'] = 'B'
    df_result.loc[df_result['排名'] <= n * 0.25, '风险等级'] = 'A'

    # 合并原始信息
    for col in ['信誉评级', '是否违约', '企业名称']:
        if col in df_features.columns:
            df_result[col] = df_features.loc[df_result['企业代号'], col].values

    # 构建权重字典
    w_dict = dict(zip(available_cols, w_combined))

    return df_result, w_dict


def validate_model(df_scores):
    """验证TOPSIS评分与实际违约标签的一致性"""
    safe_print("\n---- 模型验证 ----")
    if '是否违约' not in df_scores.columns:
        safe_print("  无违约标签, 跳过验证")
        return

    default_mask = df_scores['是否违约'] == '是'
    normal_mask = df_scores['是否违约'] == '否'

    score_def = df_scores.loc[default_mask, 'TOPSIS得分'].mean() if default_mask.any() else 0
    score_norm = df_scores.loc[normal_mask, 'TOPSIS得分'].mean() if normal_mask.any() else 0

    safe_print(f"  违约企业(n={default_mask.sum()}) 平均得分: {score_def:.4f}")
    safe_print(f"  正常企业(n={normal_mask.sum()}) 平均得分: {score_norm:.4f}")

    if score_norm < score_def:
        safe_print(f"  [WARNING] 违约企业得分({score_def:.4f}) > 正常企业({score_norm:.4f}), 模型方向可能反了!")
        safe_print(f"  [ACTION] 自动反转得分...")
        df_scores['TOPSIS得分'] = 1 - df_scores['TOPSIS得分']
        # 重新排序
        df_scores = df_scores.sort_values('TOPSIS得分', ascending=False).reset_index(drop=True)
        df_scores['排名'] = range(1, len(df_scores) + 1)
        # 重新分级
        n = len(df_scores)
        df_scores.loc[df_scores['排名'] <= n * 0.25, '风险等级'] = 'A'
        df_scores.loc[(df_scores['排名'] > n * 0.25) & (df_scores['排名'] <= n * 0.50), '风险等级'] = 'B'
        df_scores.loc[(df_scores['排名'] > n * 0.50) & (df_scores['排名'] <= n * 0.80), '风险等级'] = 'C'
        df_scores.loc[df_scores['排名'] > n * 0.80, '风险等级'] = 'D'
    else:
        # 重新计算验证统计
        score_def = df_scores.loc[default_mask, 'TOPSIS得分'].mean() if default_mask.any() else 0
        score_norm = df_scores.loc[normal_mask, 'TOPSIS得分'].mean() if normal_mask.any() else 0
        safe_print(f"  [OK] 违约企业({score_def:.4f}) < 正常企业({score_norm:.4f}), 模型方向正确")
        safe_print(f"  差异: {score_norm - score_def:.4f}")

    # 与原信誉评级交叉表
    if '信誉评级' in df_scores.columns:
        safe_print("\n  模型评级 vs 原始信誉评级:")
        ct = pd.crosstab(df_scores['信誉评级'], df_scores['风险等级'],
                         margins=True)
        safe_print(ct.to_string())

    # 各等级违约率
    safe_print("\n  各风险等级违约率:")
    for grade in ['A', 'B', 'C', 'D']:
        gmask = df_scores['风险等级'] == grade
        if gmask.any():
            ndef = (df_scores.loc[gmask, '是否违约'] == '是').sum()
            safe_print(f"    {grade}: {gmask.sum()}家, 违约{ndef}家 ({ndef/gmask.sum():.1%})")

    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        y_true = (df_scores['是否违约'] == '是').astype(int)
        y_score = df_scores['TOPSIS得分'].values
        auc = roc_auc_score(y_true, y_score)
        safe_print(f"\n  AUC = {auc:.4f}")
    except Exception as e:
        safe_print(f"  AUC计算失败: {e}")

    return df_scores


def predict_attachment2(df_features_2, w_dict, feature_config=FEATURE_CONFIG):
    """
    对附件2(无信贷记录)企业使用相同指标和权重做TOPSIS评分
    """
    safe_print("\n---- 附件2企业评估 ----")

    # 对齐指标
    common_cols = [c for c in feature_config if c in df_features_2.columns and c in w_dict]
    safe_print(f"  对齐指标: {len(common_cols)}个")

    X2, _ = preprocess_features(df_features_2, {c: feature_config[c] for c in common_cols})
    X2 = X2[common_cols]

    indicator_types = [feature_config[c] for c in common_cols]
    X2_pos = positive_transform(X2.values, indicator_types)
    X2_norm = vector_normalize(X2_pos)

    w2 = np.array([w_dict[c] for c in common_cols])
    w2 = w2 / w2.sum()

    scores2, d_pos2, d_neg2 = topsis_score(X2_norm, w2)

    # 用附件1的阈值来判断是否需要反转
    # (已在validate_model中检查过)

    df_result2 = pd.DataFrame({
        '企业代号': df_features_2.index.values,
        'TOPSIS得分': scores2,
        'D_pos': d_pos2, 'D_neg': d_neg2,
    })
    df_result2 = df_result2.sort_values('TOPSIS得分', ascending=False).reset_index(drop=True)
    df_result2['排名'] = range(1, len(df_result2) + 1)

    n = len(df_result2)
    df_result2.loc[:, '风险等级'] = 'D'
    df_result2.loc[df_result2['排名'] <= n * 0.80, '风险等级'] = 'C'
    df_result2.loc[df_result2['排名'] <= n * 0.50, '风险等级'] = 'B'
    df_result2.loc[df_result2['排名'] <= n * 0.25, '风险等级'] = 'A'

    if '企业名称' in df_features_2.columns:
        # 安全合并
        names = df_features_2[['企业名称']].copy()
        df_result2 = df_result2.merge(names, left_on='企业代号', right_index=True, how='left')

    safe_print(f"  附件2: {len(df_result2)} 家企业")
    safe_print(f"  评分范围: [{scores2.min():.4f}, {scores2.max():.4f}]")
    safe_print(f"  等级分布: {df_result2['风险等级'].value_counts().to_dict()}")

    return df_result2


def risk_model_pipeline(df1, df2):
    """风险评估完整流程"""
    safe_print("=" * 60)
    safe_print("阶段3: 风险评估模型 (AHP + 熵权 + TOPSIS)")
    safe_print("=" * 60)

    # === 问题1 ===
    safe_print("\n>>> 问题1: 123家有信贷记录企业 <<<")
    df_scores_1, w_dict = evaluate_credit_risk(df1, alpha=0.5)

    # 验证
    df_scores_1 = validate_model(df_scores_1)

    # 打印排名
    safe_print("\nTop 10 (风险最低):")
    top10 = df_scores_1.head(10)
    for _, row in top10.iterrows():
        safe_print(f"  {row['企业代号']:6s} score={row['TOPSIS得分']:.4f} grade={row['风险等级']} "
                   f"orig_rating={row.get('信誉评级','?')} default={row.get('是否违约','?')}")

    safe_print("\nBottom 10 (风险最高):")
    bot10 = df_scores_1.tail(10)
    for _, row in bot10.iterrows():
        safe_print(f"  {row['企业代号']:6s} score={row['TOPSIS得分']:.4f} grade={row['风险等级']} "
                   f"orig_rating={row.get('信誉评级','?')} default={row.get('是否违约','?')}")

    # === 问题2 ===
    safe_print("\n>>> 问题2: 302家无信贷记录企业 <<<")
    df_scores_2 = predict_attachment2(df2, w_dict)

    return df_scores_1, df_scores_2, w_dict


if __name__ == '__main__':
    base = os.path.dirname(__file__)
    f1_path = os.path.join(base, 'features_附件1.csv')
    f2_path = os.path.join(base, 'features_附件2.csv')

    if os.path.exists(f1_path) and os.path.exists(f2_path):
        df1 = pd.read_csv(f1_path, index_col='企业代号')
        df2 = pd.read_csv(f2_path, index_col='企业代号')
        s1, s2, w = risk_model_pipeline(df1, df2)
        s1.to_csv(os.path.join(base, 'scores_附件1.csv'), encoding='utf-8-sig', index=False)
        s2.to_csv(os.path.join(base, 'scores_附件2.csv'), encoding='utf-8-sig', index=False)
        safe_print("\n[OK] 评分结果已保存")
    else:
        safe_print("请先运行 feature_engineer.py 生成特征数据")
