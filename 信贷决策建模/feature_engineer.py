"""
特征工程模块 - 从发票数据构建企业级风险指标
使用SQL聚合大幅加速处理

构建指标：
  企业实力: 收入、成本、利润、利润率、交易活跃度、交易网络广度
  企业稳定性: 收入/成本波动率
  企业信誉: 有效发票率、作废发票率、负数发票率
  企业成长: 收入/利润环比增长率
"""
import sqlite3
import os
import numpy as np
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), 'credit_data.db')


def build_features_for_dataset(conn, data_prefix='1'):
    """
    对指定数据集（附件1或附件2）构建特征
    data_prefix: '1' 或 '2'
    返回: DataFrame, 每行一个企业, 列为各指标
    """
    input_table = f'input_invoice_{data_prefix}'
    output_table = f'output_invoice_{data_prefix}'

    # ---- 1. 有效发票聚合（进项）----
    sql_input = f'''
        SELECT
            企业代号,
            COUNT(*) AS 进项发票总数,
            SUM(CASE WHEN 发票状态='有效发票' THEN 1 ELSE 0 END) AS 进项有效发票数,
            SUM(CASE WHEN 发票状态='作废发票' THEN 1 ELSE 0 END) AS 进项作废发票数,
            SUM(CASE WHEN 金额 < 0 THEN 1 ELSE 0 END) AS 进项负数发票数,
            SUM(CASE WHEN 发票状态='有效发票' THEN ABS(价税合计) ELSE 0 END) AS 进项总金额,
            SUM(CASE WHEN 发票状态='有效发票' THEN ABS(税额) ELSE 0 END) AS 进项总税额,
            COUNT(DISTINCT 销方单位代号) AS 进项供应商数,
            COUNT(DISTINCT substr(开票日期,1,7)) AS 进项活跃月数,
            AVG(CASE WHEN 发票状态='有效发票' THEN ABS(价税合计) ELSE NULL END) AS 进项平均单笔金额
        FROM {input_table}
        GROUP BY 企业代号
    '''
    df_input = pd.read_sql(sql_input, conn, index_col='企业代号')

    # ---- 2. 有效发票聚合（销项）----
    sql_output = f'''
        SELECT
            企业代号,
            COUNT(*) AS 销项发票总数,
            SUM(CASE WHEN 发票状态='有效发票' THEN 1 ELSE 0 END) AS 销项有效发票数,
            SUM(CASE WHEN 发票状态='作废发票' THEN 1 ELSE 0 END) AS 销项作废发票数,
            SUM(CASE WHEN 金额 < 0 THEN 1 ELSE 0 END) AS 销项负数发票数,
            SUM(CASE WHEN 发票状态='有效发票' THEN ABS(价税合计) ELSE 0 END) AS 销项总金额,
            SUM(CASE WHEN 发票状态='有效发票' THEN ABS(税额) ELSE 0 END) AS 销项总税额,
            COUNT(DISTINCT 购方单位代号) AS 销项客户数,
            COUNT(DISTINCT substr(开票日期,1,7)) AS 销项活跃月数,
            AVG(CASE WHEN 发票状态='有效发票' THEN ABS(价税合计) ELSE NULL END) AS 销项平均单笔金额
        FROM {output_table}
        GROUP BY 企业代号
    '''
    df_output = pd.read_sql(sql_output, conn, index_col='企业代号')

    # ---- 3. 月度收入/成本波动率 ----
    sql_monthly = f'''
        SELECT
            企业代号,
            substr(开票日期,1,7) AS 月份,
            SUM(CASE WHEN 发票状态='有效发票' THEN ABS(价税合计) ELSE 0 END) AS 月收入
        FROM {output_table}
        GROUP BY 企业代号, 月份
    '''
    df_monthly_revenue = pd.read_sql(sql_monthly, conn)
    # 计算每家企业的月度收入波动率
    monthly_stats = df_monthly_revenue.groupby('企业代号')['月收入'].agg(['std', 'mean'])
    monthly_stats['收入波动率'] = monthly_stats['std'] / (monthly_stats['mean'] + 1e-6)
    monthly_stats = monthly_stats.rename(columns={'mean': '月均收入', 'std': '月收入标准差'})

    sql_monthly_cost = f'''
        SELECT
            企业代号,
            substr(开票日期,1,7) AS 月份,
            SUM(CASE WHEN 发票状态='有效发票' THEN ABS(价税合计) ELSE 0 END) AS 月成本
        FROM {input_table}
        GROUP BY 企业代号, 月份
    '''
    df_monthly_cost = pd.read_sql(sql_monthly_cost, conn)
    monthly_cost_stats = df_monthly_cost.groupby('企业代号')['月成本'].agg(['std', 'mean'])
    monthly_cost_stats['成本波动率'] = monthly_cost_stats['std'] / (monthly_cost_stats['mean'] + 1e-6)
    monthly_cost_stats = monthly_cost_stats.rename(columns={'mean': '月均成本', 'std': '月成本标准差'})

    # ---- 4. 合并所有特征 ----
    df_features = pd.concat([df_input, df_output], axis=1)
    df_features = df_features.join(monthly_stats[['月均收入', '收入波动率']])
    df_features = df_features.join(monthly_cost_stats[['月均成本', '成本波动率']])

    # ---- 5. 计算派生指标 ----
    # 利润率
    df_features['利润'] = df_features['销项总金额'] - df_features['进项总金额']
    df_features['利润率'] = df_features['利润'] / (df_features['销项总金额'] + 1e-6)
    df_features['利润率'] = df_features['利润率'].clip(-1, 1)  # 限制在合理范围

    # 有效发票率
    df_features['进项有效率'] = df_features['进项有效发票数'] / (df_features['进项发票总数'] + 1e-6)
    df_features['销项有效率'] = df_features['销项有效发票数'] / (df_features['销项发票总数'] + 1e-6)

    # 作废率
    df_features['进项作废率'] = df_features['进项作废发票数'] / (df_features['进项发票总数'] + 1e-6)
    df_features['销项作废率'] = df_features['销项作废发票数'] / (df_features['销项发票总数'] + 1e-6)

    # 负数率
    df_features['进项负数率'] = df_features['进项负数发票数'] / (df_features['进项发票总数'] + 1e-6)
    df_features['销项负数率'] = df_features['销项负数发票数'] / (df_features['销项发票总数'] + 1e-6)

    # 交易网络广度（供应商+客户总数）
    df_features['交易网络广度'] = df_features['进项供应商数'] + df_features['销项客户数']

    # 综合活跃度
    df_features['综合活跃月数'] = df_features[['进项活跃月数', '销项活跃月数']].max(axis=1)

    # 税负率
    df_features['进项税负率'] = df_features['进项总税额'] / (df_features['进项总金额'] + 1e-6)
    df_features['销项税负率'] = df_features['销项总税额'] / (df_features['销项总金额'] + 1e-6)

    # 填充NaN为0
    df_features = df_features.fillna(0)

    # ---- 6. 计算成长性指标（环比增长率）----
    # 基于月度数据计算最近两个月的收入环比增长
    growth_data = []
    for eid, group in df_monthly_revenue.groupby('企业代号'):
        group = group.sort_values('月份')
        if len(group) >= 2:
            recent = group['月收入'].iloc[-1]
            prev = group['月收入'].iloc[-2]
            if prev > 0:
                growth_data.append({'企业代号': eid, '收入环比增长率': (recent - prev) / prev})
            else:
                growth_data.append({'企业代号': eid, '收入环比增长率': 0})
        else:
            growth_data.append({'企业代号': eid, '收入环比增长率': 0})

    df_growth = pd.DataFrame(growth_data).set_index('企业代号')
    df_features = df_features.join(df_growth)
    df_features['收入环比增长率'] = df_features['收入环比增长率'].fillna(0)

    return df_features


def build_all_features(conn):
    """
    为所有数据集构建特征
    返回: (df_features_1, df_features_2)
    """
    print("\n" + "=" * 60)
    print("阶段2：特征工程（发票→企业级指标）")
    print("=" * 60)

    # 附件1特征（123家）
    print("\n[1/2] 构建附件1企业特征...")
    df1 = build_features_for_dataset(conn, '1')
    # 合并企业信息（信誉评级、是否违约）
    info1 = pd.read_sql("SELECT * FROM enterprise_info_1", conn, index_col='企业代号')
    df1 = df1.join(info1)
    print(f"  附件1: {len(df1)} 家企业, {len(df1.columns)} 个特征")

    # 附件2特征（302家）
    print("[2/2] 构建附件2企业特征...")
    df2 = build_features_for_dataset(conn, '2')
    info2 = pd.read_sql("SELECT * FROM enterprise_info_2", conn, index_col='企业代号')
    df2 = df2.join(info2)
    print(f"  附件2: {len(df2)} 家企业, {len(df2.columns)} 个特征")

    # 打印特征统计摘要
    print("\n特征统计摘要（附件1）：")
    numeric_cols = df1.select_dtypes(include=[np.number]).columns
    stats = df1[numeric_cols].describe().round(2)
    print(stats.to_string())

    # 检查数据质量
    print("\n数据质量检查：")
    print(f"  附件1 NaN值: {df1[numeric_cols].isna().sum().sum()}")
    print(f"  附件2 NaN值: {df2.select_dtypes(include=[np.number]).isna().sum().sum()}")
    # 检查是否有极端利润率的
    extreme_profit = df1[df1['利润率'].abs() > 0.8]
    if len(extreme_profit) > 0:
        print(f"  附件1 利润率极端值(>80%或<-80%): {len(extreme_profit)} 家")

    print("\n[OK] 特征工程完成！")
    return df1, df2


if __name__ == '__main__':
    from data_loader import load_all_data
    conn = load_all_data()
    df1, df2 = build_all_features(conn)
    # 保存特征供后续使用
    out = os.path.join(os.path.dirname(__file__), 'output')
    df1.to_csv(os.path.join(out, 'features_附件1.csv'), encoding='utf-8-sig')
    df2.to_csv(os.path.join(out, 'features_附件2.csv'), encoding='utf-8-sig')
    print("\n特征数据已保存到 CSV")
    conn.close()
