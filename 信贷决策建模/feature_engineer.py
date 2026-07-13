"""从发票净额与完整月历构建企业级风险特征。"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import OUTPUT_DIR, ensure_directories


def _invoice_summary(conn, table, partner_col, prefix):
    sql = f'''SELECT
        企业代号,
        COUNT(*) AS {prefix}发票总数,
        SUM(CASE WHEN 发票状态='有效发票' THEN 1 ELSE 0 END) AS {prefix}有效发票数,
        SUM(CASE WHEN 发票状态='作废发票' THEN 1 ELSE 0 END) AS {prefix}作废发票数,
        SUM(CASE WHEN 发票状态='有效发票' AND 金额<0 THEN 1 ELSE 0 END) AS {prefix}负数发票数,
        SUM(CASE WHEN 发票状态='有效发票' THEN COALESCE(金额,0) ELSE 0 END) AS {prefix}净额,
        SUM(CASE WHEN 发票状态='有效发票' AND 金额>0 THEN 金额 ELSE 0 END) AS {prefix}正数金额,
        SUM(CASE WHEN 发票状态='有效发票' AND 金额<0 THEN -金额 ELSE 0 END) AS {prefix}负数金额,
        COUNT(DISTINCT CASE WHEN 发票状态='有效发票' THEN {partner_col} END) AS {prefix}交易对手数,
        MIN(CASE WHEN 发票状态='有效发票' THEN 开票日期 END) AS {prefix}首票日期,
        MAX(CASE WHEN 发票状态='有效发票' THEN 开票日期 END) AS {prefix}末票日期
    FROM {table}
    GROUP BY 企业代号'''
    return pd.read_sql(sql, conn).set_index('企业代号')


def _monthly_net(conn, table, value_name):
    sql = f'''SELECT 企业代号, substr(开票日期,1,7) AS 月份,
        SUM(CASE WHEN 发票状态='有效发票' THEN COALESCE(金额,0) ELSE 0 END) AS {value_name}
    FROM {table}
    GROUP BY 企业代号, substr(开票日期,1,7)'''
    df = pd.read_sql(sql, conn)
    df['月份'] = pd.PeriodIndex(df['月份'], freq='M')
    return df


def _safe_ratio(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.zeros_like(num), where=np.abs(den) > 1e-12)


def _normalized_trend(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    slope = np.polyfit(x, values, 1)[0]
    scale = np.mean(np.abs(values))
    return float(slope / (scale + 1e-9))


def _trailing_zeros(values):
    count = 0
    for value in np.asarray(values)[::-1]:
        if abs(float(value)) <= 1e-12:
            count += 1
        else:
            break
    return count


def _series_groups(df, value_col):
    return {
        eid: group.set_index('月份')[value_col]
        for eid, group in df.groupby('企业代号', sort=False)
    }


def build_features_for_dataset(conn, data_prefix='1'):
    input_table = f'input_invoice_{data_prefix}'
    output_table = f'output_invoice_{data_prefix}'
    info_table = f'enterprise_info_{data_prefix}'

    info = pd.read_sql(f'SELECT * FROM {info_table}', conn, index_col='企业代号')
    inp = _invoice_summary(conn, input_table, '销方单位代号', '进项')
    out = _invoice_summary(conn, output_table, '购方单位代号', '销项')
    features = info.join(inp, how='left').join(out, how='left')

    numeric_summary = [c for c in features.columns if c.endswith(('总数', '发票数', '净额', '正数金额', '负数金额', '交易对手数'))]
    features[numeric_summary] = features[numeric_summary].fillna(0)

    monthly_cost = _monthly_net(conn, input_table, '月成本净额')
    monthly_revenue = _monthly_net(conn, output_table, '月收入净额')
    cost_groups = _series_groups(monthly_cost, '月成本净额')
    revenue_groups = _series_groups(monthly_revenue, '月收入净额')

    calendar_rows = []
    for eid in features.index:
        date_values = []
        for col in ['进项首票日期', '进项末票日期', '销项首票日期', '销项末票日期']:
            value = features.at[eid, col] if col in features.columns else None
            if pd.notna(value):
                date_values.append(pd.Timestamp(value).to_period('M'))
        if date_values:
            start, end = min(date_values), max(date_values)
            calendar = pd.period_range(start, end, freq='M')
        else:
            calendar = pd.period_range('1970-01', periods=1, freq='M')

        revenue = revenue_groups.get(eid, pd.Series(dtype=float)).reindex(calendar, fill_value=0.0).to_numpy(float)
        cost = cost_groups.get(eid, pd.Series(dtype=float)).reindex(calendar, fill_value=0.0).to_numpy(float)
        active = (np.abs(revenue) + np.abs(cost)) > 1e-12
        months = len(calendar)

        revenue_scale = np.mean(np.abs(revenue))
        cost_scale = np.mean(np.abs(cost))
        calendar_rows.append({
            '企业代号': eid,
            '观察月数': months,
            '综合活跃月数': int(active.sum()),
            '活跃月份比例': float(active.mean()),
            '最近断档月数': _trailing_zeros(active.astype(float)),
            '月均收入净额': float(revenue.mean()),
            '月均成本净额': float(cost.mean()),
            '收入波动率': float(revenue.std(ddof=1) / (revenue_scale + 1e-9)) if months > 1 else 0.0,
            '成本波动率': float(cost.std(ddof=1) / (cost_scale + 1e-9)) if months > 1 else 0.0,
            '收入趋势率': _normalized_trend(revenue),
            '成本趋势率': _normalized_trend(cost),
        })

    calendar_df = pd.DataFrame(calendar_rows).set_index('企业代号')
    features = features.join(calendar_df)

    years = np.maximum(features['观察月数'].to_numpy(float) / 12.0, 1 / 12)
    features['年化销项净额'] = features['销项净额'].to_numpy(float) / years
    features['年化进项净额'] = features['进项净额'].to_numpy(float) / years
    features['年化利润'] = features['年化销项净额'] - features['年化进项净额']
    features['利润率'] = _safe_ratio(features['年化利润'], np.abs(features['年化销项净额'])).clip(-2, 2)
    features['年化客户数'] = features['销项交易对手数'].to_numpy(float) / years
    features['年化供应商数'] = features['进项交易对手数'].to_numpy(float) / years

    for prefix in ['进项', '销项']:
        total = features[f'{prefix}发票总数'].to_numpy(float)
        valid = features[f'{prefix}有效发票数'].to_numpy(float)
        void = features[f'{prefix}作废发票数'].to_numpy(float)
        negative_count = features[f'{prefix}负数发票数'].to_numpy(float)
        positive_amount = features[f'{prefix}正数金额'].to_numpy(float)
        negative_amount = features[f'{prefix}负数金额'].to_numpy(float)
        features[f'{prefix}有效率'] = _safe_ratio(valid, total)
        features[f'{prefix}作废率'] = _safe_ratio(void, total)
        features[f'{prefix}负数发票率'] = _safe_ratio(negative_count, valid)
        features[f'{prefix}负数金额率'] = _safe_ratio(negative_amount, positive_amount + negative_amount)

    date_cols = [c for c in features.columns if c.endswith('日期')]
    features = features.drop(columns=date_cols)
    numeric = features.select_dtypes(include=[np.number]).columns
    features[numeric] = features[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features


def build_all_features(conn):
    print('\n' + '=' * 60)
    print('阶段2：特征工程（净额、连续月历与年化指标）')
    print('=' * 60)
    df1 = build_features_for_dataset(conn, '1')
    df2 = build_features_for_dataset(conn, '2')
    print(f'  附件1: {len(df1)}家企业, {len(df1.columns)}列')
    print(f'  附件2: {len(df2)}家企业, {len(df2.columns)}列')
    print(f"  附件1年化销项净额合计: {df1['年化销项净额'].sum()/1e8:.2f}亿元")
    print(f"  附件2年化销项净额合计: {df2['年化销项净额'].sum()/1e8:.2f}亿元")
    return df1, df2


if __name__ == '__main__':
    from data_loader import load_all_data

    ensure_directories()
    connection = load_all_data()
    f1, f2 = build_all_features(connection)
    f1.to_csv(OUTPUT_DIR / 'features_附件1.csv', encoding='utf-8-sig')
    f2.to_csv(OUTPUT_DIR / 'features_附件2.csv', encoding='utf-8-sig')
    connection.close()
