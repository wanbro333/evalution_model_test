"""真实重算的TOPSIS权重扰动与指标移除分析。"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from config import RANDOM_SEED
from risk_model import topsis_scores


def _ranks_descending(values):
    return pd.Series(values).rank(method='average', ascending=False).to_numpy()


def topsis_weight_perturbation(df_features, topsis_model, n_trials=100, seed=RANDOM_SEED):
    base_weights = np.asarray(topsis_model['weights'], dtype=float)
    base_scores, _, _ = topsis_scores(df_features, topsis_model, base_weights)
    base_ranks = _ranks_descending(base_scores)
    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(n_trials):
        weights = base_weights * rng.uniform(0.9, 1.1, len(base_weights))
        weights /= weights.sum()
        scores, _, _ = topsis_scores(df_features, topsis_model, weights)
        ranks = _ranks_descending(scores)
        rows.append({
            '试验编号': trial + 1,
            'Spearman相关系数': float(spearmanr(base_ranks, ranks).statistic),
            'Kendall相关系数': float(kendalltau(base_ranks, ranks).statistic),
            '最大权重变化': float(np.max(np.abs(weights - base_weights))),
        })
    return pd.DataFrame(rows)


def topsis_indicator_removal(df_features, topsis_model):
    base_weights = np.asarray(topsis_model['weights'], dtype=float)
    base_scores, _, _ = topsis_scores(df_features, topsis_model, base_weights)
    base_ranks = _ranks_descending(base_scores)
    rows = []
    for index, feature in enumerate(topsis_model['features']):
        weights = base_weights.copy()
        weights[index] = 0
        weights /= weights.sum()
        scores, _, _ = topsis_scores(df_features, topsis_model, weights)
        ranks = _ranks_descending(scores)
        rows.append({
            '移除指标': feature,
            '原权重': float(base_weights[index]),
            'Spearman相关系数': float(spearmanr(base_ranks, ranks).statistic),
            'Kendall相关系数': float(kendalltau(base_ranks, ranks).statistic),
        })
    return pd.DataFrame(rows).sort_values('Spearman相关系数')


def run_sensitivity(df1_features, model_package, n_trials=100):
    print('\n阶段6：真实权重扰动灵敏度分析')
    trials = topsis_weight_perturbation(
        df1_features, model_package['topsis'], n_trials=n_trials, seed=model_package['random_seed']
    )
    removal = topsis_indicator_removal(df1_features, model_package['topsis'])
    summary = {
        'trial_count': int(len(trials)),
        'spearman_mean': float(trials['Spearman相关系数'].mean()),
        'spearman_min': float(trials['Spearman相关系数'].min()),
        'kendall_mean': float(trials['Kendall相关系数'].mean()),
        'kendall_min': float(trials['Kendall相关系数'].min()),
        'most_influential_indicator': str(removal.iloc[0]['移除指标']),
        'most_influential_removal_spearman': float(removal.iloc[0]['Spearman相关系数']),
    }
    print(f"  Spearman均值={summary['spearman_mean']:.4f}, 最小值={summary['spearman_min']:.4f}")
    print(f"  最有影响指标={summary['most_influential_indicator']}")
    return trials, removal, summary


if __name__ == '__main__':
    import json
    from config import OUTPUT_DIR

    features = pd.read_csv(OUTPUT_DIR / 'features_附件1.csv', index_col='企业代号')
    with open(OUTPUT_DIR / 'model_package.json', encoding='utf-8') as handle:
        package = json.load(handle)
    trial_df, removal_df, _ = run_sensitivity(features, package)
    trial_df.to_csv(OUTPUT_DIR / 'sensitivity_trials.csv', index=False, encoding='utf-8-sig')
    removal_df.to_csv(OUTPUT_DIR / 'sensitivity_indicator_removal.csv', index=False, encoding='utf-8-sig')
