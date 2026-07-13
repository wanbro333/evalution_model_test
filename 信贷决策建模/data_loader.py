"""
数据加载与SQLite入库模块
功能：
1. 读取附件1-3的Excel数据
2. 数据清洗（去作废发票、处理缺失值等）
3. 建立SQLite数据库加速查询
4. 导出企业级汇总数据供特征工程使用
"""
import sqlite3
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from config import DATA_DIR, DB_PATH as CONFIG_DB_PATH, ensure_directories

BASE_DIR = str(DATA_DIR)
DB_PATH = str(CONFIG_DB_PATH)


def find_file(pattern):
    """在原始数据目录中查找匹配pattern的文件，排除临时文件。"""
    files = glob.glob(os.path.join(BASE_DIR, pattern))
    files = [f for f in files if '~$' not in f]
    if not files:
        raise FileNotFoundError(f"在 {BASE_DIR} 中未找到文件: {pattern}")
    return files[0]


def load_excel_safe(filepath, sheet_name=0):
    """安全加载Excel sheet"""
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    # 删除完全为空的列
    df = df.dropna(axis=1, how='all')
    return df


def create_database():
    """创建SQLite数据库和表结构"""
    conn = sqlite3.connect(DB_PATH)

    # 附件1：企业信息表（含信贷记录）
    conn.execute('''
        CREATE TABLE IF NOT EXISTS enterprise_info_1 (
            企业代号 TEXT PRIMARY KEY,
            企业名称 TEXT,
            信誉评级 TEXT,
            是否违约 TEXT
        )
    ''')

    # 附件2：企业信息表（无信贷记录）
    conn.execute('''
        CREATE TABLE IF NOT EXISTS enterprise_info_2 (
            企业代号 TEXT PRIMARY KEY,
            企业名称 TEXT
        )
    ''')

    # 进项发票表（附件1的进项）
    conn.execute('''
        CREATE TABLE IF NOT EXISTS input_invoice_1 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            企业代号 TEXT,
            发票号码 TEXT,
            开票日期 TEXT,
            销方单位代号 TEXT,
            金额 REAL,
            税额 REAL,
            价税合计 REAL,
            发票状态 TEXT
        )
    ''')

    # 销项发票表（附件1的销项）
    conn.execute('''
        CREATE TABLE IF NOT EXISTS output_invoice_1 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            企业代号 TEXT,
            发票号码 TEXT,
            开票日期 TEXT,
            购方单位代号 TEXT,
            金额 REAL,
            税额 REAL,
            价税合计 REAL,
            发票状态 TEXT
        )
    ''')

    # 进项发票表（附件2的进项）
    conn.execute('''
        CREATE TABLE IF NOT EXISTS input_invoice_2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            企业代号 TEXT,
            发票号码 TEXT,
            开票日期 TEXT,
            销方单位代号 TEXT,
            金额 REAL,
            税额 REAL,
            价税合计 REAL,
            发票状态 TEXT
        )
    ''')

    # 销项发票表（附件2的销项）
    conn.execute('''
        CREATE TABLE IF NOT EXISTS output_invoice_2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            企业代号 TEXT,
            发票号码 TEXT,
            开票日期 TEXT,
            购方单位代号 TEXT,
            金额 REAL,
            税额 REAL,
            价税合计 REAL,
            发票状态 TEXT
        )
    ''')

    # 附件3：利率与流失率
    conn.execute('''
        CREATE TABLE IF NOT EXISTS rate_churn (
            贷款年利率 REAL,
            客户流失率_A REAL,
            客户流失率_B REAL,
            客户流失率_C REAL
        )
    ''')

    conn.commit()
    return conn


def load_all_data():
    """
    主函数：加载所有数据到SQLite
    返回: conn (数据库连接)
    """
    print("=" * 60)
    print("阶段1：数据加载与SQLite入库")
    print("=" * 60)

    ensure_directories()
    # 创表
    conn = create_database()

    # ---- 加载附件1 ----
    file1 = find_file('附件1*')
    print(f"\n[1/3] 加载附件1: {os.path.basename(file1)}")

    # 企业信息
    df = load_excel_safe(file1, '企业信息')
    print(f"  企业信息: {len(df)} 条记录")
    df.to_sql('enterprise_info_1', conn, if_exists='replace', index=False)

    # 统计信誉评级分布
    print(f"  信誉评级分布: {df['信誉评级'].value_counts().to_dict()}")

    # 进项发票
    df_in = load_excel_safe(file1, '进项发票信息')
    df_in['开票日期'] = pd.to_datetime(df_in['开票日期']).astype(str)
    print(f"  进项发票: {len(df_in)} 条, 日期范围: {pd.to_datetime(df_in['开票日期']).min()} ~ {pd.to_datetime(df_in['开票日期']).max()}")
    print(f"  发票状态: {df_in['发票状态'].value_counts().to_dict()}")
    df_in.to_sql('input_invoice_1', conn, if_exists='replace', index=False)

    # 销项发票
    df_out = load_excel_safe(file1, '销项发票信息')
    df_out['开票日期'] = pd.to_datetime(df_out['开票日期']).astype(str)
    print(f"  销项发票: {len(df_out)} 条, 日期范围: {pd.to_datetime(df_out['开票日期']).min()} ~ {pd.to_datetime(df_out['开票日期']).max()}")
    print(f"  发票状态: {df_out['发票状态'].value_counts().to_dict()}")
    df_out.to_sql('output_invoice_1', conn, if_exists='replace', index=False)

    # ---- 加载附件2 ----
    file2 = find_file('附件2*')
    print(f"\n[2/3] 加载附件2: {os.path.basename(file2)}")

    df = load_excel_safe(file2, '企业信息')
    print(f"  企业信息: {len(df)} 条记录")
    df.to_sql('enterprise_info_2', conn, if_exists='replace', index=False)

    df_in2 = load_excel_safe(file2, '进项发票信息')
    df_in2['开票日期'] = pd.to_datetime(df_in2['开票日期']).astype(str)
    print(f"  进项发票: {len(df_in2)} 条")
    print(f"  发票状态: {df_in2['发票状态'].value_counts().to_dict()}")
    df_in2.to_sql('input_invoice_2', conn, if_exists='replace', index=False)

    df_out2 = load_excel_safe(file2, '销项发票信息')
    df_out2['开票日期'] = pd.to_datetime(df_out2['开票日期']).astype(str)
    print(f"  销项发票: {len(df_out2)} 条")
    print(f"  发票状态: {df_out2['发票状态'].value_counts().to_dict()}")
    df_out2.to_sql('output_invoice_2', conn, if_exists='replace', index=False)

    # ---- 加载附件3 ----
    file3 = find_file('附件3*')
    print(f"\n[3/3] 加载附件3: {os.path.basename(file3)}")

    df3 = load_excel_safe(file3, 'Sheet1')
    # 附件3格式: 第1行=列名(贷款年利率,客户流失率,None,None)
    #             第2行=子列名(None,信誉评级A,信誉评级B,信誉评级C)
    # 重构数据
    clean_data = []
    for _, row in df3.iterrows():
        rate = row.iloc[0]
        if pd.notna(rate) and isinstance(rate, (int, float)):
            clean_data.append({
                '贷款年利率': float(rate),
                '客户流失率_A': float(row.iloc[1]) if pd.notna(row.iloc[1]) else 0,
                '客户流失率_B': float(row.iloc[2]) if pd.notna(row.iloc[2]) else 0,
                '客户流失率_C': float(row.iloc[3]) if pd.notna(row.iloc[3]) else 0,
            })
    df3_clean = pd.DataFrame(clean_data)
    print(f"  利率-流失率数据: {len(df3_clean)} 行, 利率范围 {df3_clean['贷款年利率'].min():.2%}~{df3_clean['贷款年利率'].max():.2%}")
    df3_clean.to_sql('rate_churn', conn, if_exists='replace', index=False)

    # 创建索引加速查询
    print("\n创建数据库索引...")
    for table in ['input_invoice_1', 'output_invoice_1', 'input_invoice_2', 'output_invoice_2']:
        conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_eid ON {table}(企业代号)')
        conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_date ON {table}(开票日期)')
        conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_status ON {table}(发票状态)')
    conn.commit()

    print("\n[OK] 数据加载完成！")
    return conn


if __name__ == '__main__':
    conn = load_all_data()
    conn.close()
