"""监督违约概率模型与固定尺度TOPSIS解释模型。"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(__file__))
from config import GRADE_THRESHOLDS, RANDOM_SEED


PD_FEATURES = [
    '年化销项净额', '年化利润', '利润率', '年化客户数', '年化供应商数',
    '收入波动率', '成本波动率', '收入趋势率', '活跃月份比例', '最近断档月数',
    '销项作废率', '进项作废率', '销项负数发票率', '进项负数发票率',
    '销项负数金额率', '进项负数金额率',
]

LOG_FEATURES = {'年化销项净额', '年化利润', '年化客户数', '年化供应商数'}

TOPSIS_CONFIG = {
    '年化销项净额': ('positive', 0.10),
    '年化利润': ('positive', 0.12),
    '利润率': ('positive', 0.10),
    '年化客户数': ('positive', 0.07),
    '年化供应商数': ('positive', 0.07),
    '活跃月份比例': ('positive', 0.08),
    '收入趋势率': ('positive', 0.08),
    '收入波动率': ('negative', 0.08),
    '成本波动率': ('negative', 0.06),
    '最近断档月数': ('negative', 0.05),
    '销项作废率': ('negative', 0.05),
    '进项作废率': ('negative', 0.04),
    '销项负数金额率': ('negative', 0.05),
    '进项负数金额率': ('negative', 0.05),
}


def signed_log1p(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def fit_preprocessor(df, feature_names=PD_FEATURES):
    x = df[feature_names].to_numpy(float).copy()
    for j, name in enumerate(feature_names):
        if name in LOG_FEATURES:
            x[:, j] = signed_log1p(x[:, j])
    lower = np.quantile(x, 0.01, axis=0)
    upper = np.quantile(x, 0.99, axis=0)
    x = np.clip(x, lower, upper)
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    return {
        'features': list(feature_names),
        'lower': lower.tolist(),
        'upper': upper.tolist(),
        'mean': mean.tolist(),
        'scale': scale.tolist(),
        'log_features': sorted(LOG_FEATURES.intersection(feature_names)),
    }


def apply_preprocessor(df, preprocessor):
    names = preprocessor['features']
    x = df[names].to_numpy(float).copy()
    log_names = set(preprocessor['log_features'])
    for j, name in enumerate(names):
        if name in log_names:
            x[:, j] = signed_log1p(x[:, j])
    x = np.clip(x, np.asarray(preprocessor['lower']), np.asarray(preprocessor['upper']))
    return (x - np.asarray(preprocessor['mean'])) / np.asarray(preprocessor['scale'])


def fit_logistic(x, y, l2=1.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = x.shape
    positives = max(float(y.sum()), 1.0)
    negatives = max(float(n - y.sum()), 1.0)
    sample_weight = np.where(y == 1, n / (2 * positives), n / (2 * negatives))
    weight_sum = sample_weight.sum()

    def objective(params):
        intercept, coef = params[0], params[1:]
        z = intercept + x @ coef
        losses = np.logaddexp(0.0, z) - y * z
        value = float(np.dot(sample_weight, losses) / weight_sum + 0.5 * l2 * np.dot(coef, coef))
        residual = sample_weight * (expit(z) - y) / weight_sum
        gradient = np.r_[residual.sum(), x.T @ residual + l2 * coef]
        return value, gradient

    result = minimize(objective, np.zeros(p + 1), jac=True, method='L-BFGS-B', options={'maxiter': 1000})
    if not result.success:
        raise RuntimeError(f'逻辑回归优化失败: {result.message}')
    return result.x


def _prior_corrected_probability(x, params, prior):
    prior = float(np.clip(prior, 1e-6, 1 - 1e-6))
    prior_offset = np.log(prior / (1 - prior))
    return expit(params[0] + x @ params[1:] + prior_offset)


def stratified_folds(y, n_splits=5, seed=RANDOM_SEED):
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    groups = []
    for label in [0, 1]:
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        groups.append(np.array_split(indices, n_splits))
    all_indices = np.arange(len(y))
    for fold in range(n_splits):
        test = np.sort(np.concatenate([groups[0][fold], groups[1][fold]]))
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        yield train, test


def binary_log_loss(y, probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability)))


def select_lambda(df, y, candidates=(0.01, 0.1, 1.0, 10.0), n_splits=4, seed=RANDOM_SEED):
    losses = {}
    for value in candidates:
        fold_losses = []
        for train, valid in stratified_folds(y, n_splits=n_splits, seed=seed):
            prep = fit_preprocessor(df.iloc[train])
            params = fit_logistic(apply_preprocessor(df.iloc[train], prep), y[train], value)
            pred = _prior_corrected_probability(
                apply_preprocessor(df.iloc[valid], prep), params, np.mean(y[train])
            )
            fold_losses.append(binary_log_loss(y[valid], pred))
        losses[float(value)] = float(np.mean(fold_losses))
    best = min(losses, key=losses.get)
    return best, losses


def nested_oof_predictions(df, y, repeats=5, n_splits=5, seed=RANDOM_SEED):
    y = np.asarray(y, dtype=int)
    prediction_sum = np.zeros(len(y), dtype=float)
    prediction_count = np.zeros(len(y), dtype=int)
    selected = []
    for repeat in range(repeats):
        outer_seed = seed + repeat * 1009
        for fold, (train, test) in enumerate(stratified_folds(y, n_splits, outer_seed)):
            best, _ = select_lambda(
                df.iloc[train], y[train], n_splits=4, seed=outer_seed + fold + 1
            )
            prep = fit_preprocessor(df.iloc[train])
            params = fit_logistic(apply_preprocessor(df.iloc[train], prep), y[train], best)
            pred = _prior_corrected_probability(
                apply_preprocessor(df.iloc[test], prep), params, np.mean(y[train])
            )
            prediction_sum[test] += pred
            prediction_count[test] += 1
            selected.append(best)
    if np.any(prediction_count == 0):
        raise RuntimeError('存在未获得样本外预测的企业')
    return prediction_sum / prediction_count, selected


def roc_auc(y, score):
    y = np.asarray(y, dtype=int)
    ranks = rankdata(np.asarray(score, dtype=float), method='average')
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y, score):
    y = np.asarray(y, dtype=int)
    order = np.argsort(-np.asarray(score, dtype=float), kind='mergesort')
    sorted_y = y[order]
    precision = np.cumsum(sorted_y) / np.arange(1, len(y) + 1)
    return float(precision[sorted_y == 1].mean())


def grade_from_pd(probability):
    p = np.asarray(probability, dtype=float)
    result = np.full(len(p), 'D', dtype=object)
    result[p < GRADE_THRESHOLDS['D'][0]] = 'C'
    result[p < GRADE_THRESHOLDS['C'][0]] = 'B'
    result[p < GRADE_THRESHOLDS['B'][0]] = 'A'
    return result


def evaluate_predictions(y, probability):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    metrics = {
        'roc_auc': roc_auc(y, probability),
        'pr_auc': average_precision(y, probability),
        'brier': float(np.mean((probability - y) ** 2)),
        'log_loss': binary_log_loss(y, probability),
        'base_default_rate': float(y.mean()),
    }
    grades = grade_from_pd(probability)
    grade_rows = []
    for grade in ['A', 'B', 'C', 'D']:
        mask = grades == grade
        if mask.any():
            grade_rows.append({
                '风险等级': grade,
                '企业数': int(mask.sum()),
                '违约数': int(y[mask].sum()),
                '实际违约率': float(y[mask].mean()),
                '平均预测PD': float(probability[mask].mean()),
            })
    metrics['grade_table'] = grade_rows
    return metrics


def fit_topsis(df):
    names = list(TOPSIS_CONFIG)
    x = df[names].to_numpy(float).copy()
    for j, name in enumerate(names):
        if name in LOG_FEATURES:
            x[:, j] = signed_log1p(x[:, j])
    lower = np.quantile(x, 0.01, axis=0)
    upper = np.quantile(x, 0.99, axis=0)
    x = np.clip(x, lower, upper)
    minimum = x.min(axis=0)
    maximum = x.max(axis=0)
    weights = np.array([TOPSIS_CONFIG[name][1] for name in names], dtype=float)
    weights /= weights.sum()
    return {
        'features': names,
        'lower': lower.tolist(),
        'upper': upper.tolist(),
        'minimum': minimum.tolist(),
        'maximum': maximum.tolist(),
        'directions': [TOPSIS_CONFIG[name][0] for name in names],
        'weights': weights.tolist(),
        'log_features': sorted(LOG_FEATURES.intersection(names)),
    }


def topsis_matrix(df, model):
    names = model['features']
    x = df[names].to_numpy(float).copy()
    log_names = set(model['log_features'])
    for j, name in enumerate(names):
        if name in log_names:
            x[:, j] = signed_log1p(x[:, j])
    x = np.clip(x, np.asarray(model['lower']), np.asarray(model['upper']))
    minimum = np.asarray(model['minimum'])
    span = np.asarray(model['maximum']) - minimum
    span[span < 1e-12] = 1.0
    normalized = (x - minimum) / span
    for j, direction in enumerate(model['directions']):
        if direction == 'negative':
            normalized[:, j] = 1 - normalized[:, j]
    return np.clip(normalized, 0, 1)


def topsis_scores(df, model, weights=None):
    normalized = topsis_matrix(df, model)
    weights = np.asarray(model['weights'] if weights is None else weights, dtype=float)
    weights /= weights.sum()
    weighted = normalized * weights
    positive = weights
    negative = np.zeros_like(weights)
    d_positive = np.sqrt(np.sum((weighted - positive) ** 2, axis=1))
    d_negative = np.sqrt(np.sum((weighted - negative) ** 2, axis=1))
    score = d_negative / np.maximum(d_positive + d_negative, 1e-12)
    return score, d_positive, d_negative


def _score_frame(df, probability, topsis_score, d_pos, d_neg):
    result = pd.DataFrame({
        '企业代号': df.index.astype(str),
        '违约概率': probability,
        '风险等级': grade_from_pd(probability),
        'TOPSIS得分': topsis_score,
        'D_pos': d_pos,
        'D_neg': d_neg,
    })
    result['PD排名'] = result['违约概率'].rank(method='min', ascending=True).astype(int)
    result['TOPSIS排名'] = result['TOPSIS得分'].rank(method='min', ascending=False).astype(int)
    for col in ['企业名称', '信誉评级', '是否违约']:
        if col in df.columns:
            result[col] = df.loc[result['企业代号'], col].to_numpy()
    if '信誉评级' in result.columns:
        result['硬性禁贷'] = result['信誉评级'].eq('D') | result['风险等级'].eq('D')
    else:
        result['硬性禁贷'] = result['风险等级'].eq('D')
    return result.sort_values(['违约概率', 'TOPSIS得分'], ascending=[True, False]).reset_index(drop=True)


def risk_model_pipeline(df1, df2):
    print('\n' + '=' * 60)
    print('阶段3：监督违约概率模型 + 固定尺度TOPSIS')
    print('=' * 60)
    y = (df1['是否违约'].astype(str) == '是').astype(int).to_numpy()

    oof_probability, selected_lambdas = nested_oof_predictions(df1, y)
    metrics = evaluate_predictions(y, oof_probability)
    final_lambda, lambda_losses = select_lambda(df1, y, n_splits=5, seed=RANDOM_SEED + 77)
    preprocessor = fit_preprocessor(df1)
    params = fit_logistic(apply_preprocessor(df1, preprocessor), y, final_lambda)
    probability2 = _prior_corrected_probability(
        apply_preprocessor(df2, preprocessor), params, y.mean()
    )

    topsis_model = fit_topsis(df1)
    score1, dpos1, dneg1 = topsis_scores(df1, topsis_model)
    score2, dpos2, dneg2 = topsis_scores(df2, topsis_model)

    result1 = _score_frame(df1, oof_probability, score1, dpos1, dneg1)
    result2 = _score_frame(df2, probability2, score2, dpos2, dneg2)

    coefficients = pd.DataFrame({
        '特征': preprocessor['features'],
        '标准化系数': params[1:],
        '优势比': np.exp(np.clip(params[1:], -20, 20)),
        '风险方向': np.where(params[1:] > 0, '提高违约风险', '降低违约风险'),
    }).sort_values('标准化系数', key=np.abs, ascending=False)

    model_package = {
        'model_type': 'L2-balanced-logistic-with-prior-correction',
        'random_seed': RANDOM_SEED,
        'preprocessor': preprocessor,
        'parameters': params.tolist(),
        'prior': float(y.mean()),
        'lambda': float(final_lambda),
        'lambda_cv_losses': {str(k): v for k, v in lambda_losses.items()},
        'nested_selected_lambdas': [float(v) for v in selected_lambdas],
        'grade_thresholds': GRADE_THRESHOLDS,
        'topsis': topsis_model,
    }

    print(f"  样本外 ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  样本外 PR-AUC:  {metrics['pr_auc']:.4f}")
    print(f"  Brier分数:      {metrics['brier']:.4f}")
    print(f"  最终L2系数:     {final_lambda}")
    print(f"  附件1等级: {result1['风险等级'].value_counts().to_dict()}")
    print(f"  附件2等级: {result2['风险等级'].value_counts().to_dict()}")
    return result1, result2, model_package, metrics, coefficients


if __name__ == '__main__':
    from config import OUTPUT_DIR

    f1 = pd.read_csv(OUTPUT_DIR / 'features_附件1.csv', index_col='企业代号')
    f2 = pd.read_csv(OUTPUT_DIR / 'features_附件2.csv', index_col='企业代号')
    s1, s2, package, validation, coef = risk_model_pipeline(f1, f2)
    s1.to_csv(OUTPUT_DIR / 'scores_附件1.csv', index=False, encoding='utf-8-sig')
    s2.to_csv(OUTPUT_DIR / 'scores_附件2.csv', index=False, encoding='utf-8-sig')
    coef.to_csv(OUTPUT_DIR / 'model_coefficients.csv', index=False, encoding='utf-8-sig')
    with open(OUTPUT_DIR / 'model_package.json', 'w', encoding='utf-8') as handle:
        json.dump(package, handle, ensure_ascii=False, indent=2)
