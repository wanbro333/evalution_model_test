"""
可视化模块
生成：排名柱状图、雷达图、权重分布图、利率-流失率曲线、灵敏度分析图
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互模式
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_ranking_bar(df_scores, title_prefix, filename, top_n=30):
    """风险评分排名柱状图"""
    fig, ax = plt.subplots(figsize=(14, 6))

    df_plot = df_scores.head(top_n).copy()
    labels = [str(eid) for eid in df_plot['企业代号']]
    scores = df_plot['TOPSIS得分'].values
    colors = []
    for grade in df_plot['风险等级']:
        color_map = {'A': '#2ecc71', 'B': '#3498db', 'C': '#f39c12', 'D': '#e74c3c'}
        colors.append(color_map.get(grade, '#95a5a6'))

    bars = ax.bar(range(len(labels)), scores, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('TOPSIS得分')
    ax.set_title(f'{title_prefix} - 企业信贷风险评分排名 (Top {top_n})')
    ax.set_ylim(0, max(scores) * 1.1)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='A级(低风险)'),
        Patch(facecolor='#3498db', label='B级'),
        Patch(facecolor='#f39c12', label='C级'),
        Patch(facecolor='#e74c3c', label='D级(高风险)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


def plot_radar(df_scores, title_prefix, filename):
    """各风险等级的雷达图 - 展示各等级的指标均值"""
    # 简化：用风险等级分布饼图代替
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 饼图：等级分布
    grade_counts = df_scores['风险等级'].value_counts()
    colors_pie = {'A': '#2ecc71', 'B': '#3498db', 'C': '#f39c12', 'D': '#e74c3c'}
    pie_colors = [colors_pie.get(g, '#95a5a6') for g in grade_counts.index]
    ax1.pie(grade_counts.values, labels=grade_counts.index, colors=pie_colors,
            autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'{title_prefix}\n风险等级分布')

    # 柱状图：各等级统计
    grades = ['A', 'B', 'C', 'D']
    counts = [grade_counts.get(g, 0) for g in grades]
    ax2.bar(grades, counts, color=[colors_pie.get(g) for g in grades])
    ax2.set_ylabel('企业数量')
    ax2.set_title('各等级企业数量')
    for i, v in enumerate(counts):
        ax2.text(i, v + 0.5, str(v), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


def plot_rate_churn(coeffs, filename):
    """利率-流失率拟合曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))

    rates = np.linspace(0.04, 0.15, 100)
    grade_colors = {'A': '#2ecc71', 'B': '#3498db', 'C': '#f39c12'}

    for grade in ['A', 'B', 'C']:
        if grade in coeffs:
            churn = np.polyval(coeffs[grade], rates)
            churn = np.clip(churn, 0, 1)
            ax.plot(rates * 100, churn * 100, label=f'信誉评级{grade}',
                    color=grade_colors[grade], linewidth=2)

    ax.set_xlabel('贷款年利率(%)')
    ax.set_ylabel('客户流失率(%)')
    ax.set_title('贷款利率与客户流失率关系')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(4, 15)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


def plot_weight_distribution(w_dict, filename):
    """权重分布图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    items = sorted(w_dict.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in items]
    weights = [x[1] for x in items]

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(labels)))
    bars = ax.barh(range(len(labels)), weights, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('权重')
    ax.set_title('指标权重分布 (AHP+熵权组合)')
    ax.invert_yaxis()

    for i, (label, w) in enumerate(zip(labels, weights)):
        ax.text(w + 0.002, i, f'{w:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


def plot_emergency_comparison(df_strategy_adj, filename):
    """突发因素调整前后对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 各行业调整前后平均额度
    ax1 = axes[0]
    industries = ['严重负面', '中度负面', '其他', '正面/轻影响']
    orig_loans = []
    adj_loans = []
    for ind in industries:
        mask = df_strategy_adj['行业类别'] == ind
        if mask.any():
            orig_loans.append(df_strategy_adj.loc[mask, '贷款额度_万元'].mean())
            adj_loans.append(df_strategy_adj.loc[mask, '调整后额度'].mean())
        else:
            orig_loans.append(0)
            adj_loans.append(0)

    x = np.arange(len(industries))
    width = 0.35
    ax1.bar(x - width/2, orig_loans, width, label='调整前', color='#3498db')
    ax1.bar(x + width/2, adj_loans, width, label='调整后', color='#e74c3c')
    ax1.set_xticks(x)
    ax1.set_xticklabels(industries, fontsize=9)
    ax1.set_ylabel('平均贷款额度(万元)')
    ax1.set_title('疫情调整前后各行业平均贷款额度')
    ax1.legend()

    # 右图: 等级变化流向
    ax2 = axes[1]
    if '风险等级' in df_strategy_adj.columns and '调整后等级' in df_strategy_adj.columns:
        changed = (df_strategy_adj['风险等级'] != df_strategy_adj['调整后等级'])
        change_counts = df_strategy_adj[changed].groupby(['风险等级', '调整后等级']).size()

        levels = ['A', 'B', 'C', 'D']
        changes_label = []
        changes_count = []
        for old in levels:
            for new in levels:
                count = change_counts.get((old, new), 0)
                if count > 0:
                    changes_label.append(f'{old}->{new}')
                    changes_count.append(count)

        if changes_count:
            ax2.bar(range(len(changes_label)), changes_count, color='#9b59b6')
            ax2.set_xticks(range(len(changes_label)))
            ax2.set_xticklabels(changes_label, fontsize=8)
            ax2.set_ylabel('企业数量')
            ax2.set_title('风险等级调整流向')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


def create_all_visualizations(df_scores_1, df_scores_2, df_strategy_2_adj, coeffs, w_dict):
    """生成所有可视化图表"""
    print("\n[可视化] 生成图表...")

    # 1. 附件1排名
    plot_ranking_bar(df_scores_1, '附件1(123家)', 'ranking_附件1.png', top_n=30)

    # 2. 附件2排名
    plot_ranking_bar(df_scores_2, '附件2(302家)', 'ranking_附件2.png', top_n=30)

    # 3. 等级分布
    plot_radar(df_scores_1, '附件1', 'grade_dist_附件1.png')
    plot_radar(df_scores_2, '附件2', 'grade_dist_附件2.png')

    # 4. 利率-流失率曲线
    if coeffs:
        plot_rate_churn(coeffs, 'rate_churn_curve.png')

    # 5. 权重分布
    if w_dict:
        plot_weight_distribution(w_dict, 'weight_distribution.png')

    # 6. 疫情调整对比
    if df_strategy_2_adj is not None:
        plot_emergency_comparison(df_strategy_2_adj, 'emergency_adjustment.png')

    print(f"  [OK] 图表已保存到: {OUTPUT_DIR}")
