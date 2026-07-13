"""从真实输出生成论文和Dashboard共用图表。"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import CHART_DIR, ensure_directories


COLORS = {'A': '#0f766e', 'B': '#2563a6', 'C': '#d97706', 'D': '#b42318'}


def setup_style():
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 140
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.titleweight'] = 'bold'


def save(fig, name):
    fig.tight_layout()
    fig.savefig(CHART_DIR / name, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_default_rate(scores1):
    y = (scores1['是否违约'] == '是').astype(int)
    data = scores1.assign(_default=y).groupby('风险等级')['_default'].agg(['count', 'sum', 'mean']).reindex(['A', 'B', 'C', 'D']).fillna(0)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    x = np.arange(4)
    ax.bar(x, data['count'], color='#d9e2e7', width=0.62, label='企业数')
    ax.bar(x, data['sum'], color=[COLORS[g] for g in data.index], width=0.62, label='违约数')
    ax.set_xticks(x, [f'{g}级' for g in data.index])
    ax.set_ylabel('企业数')
    ax2 = ax.twinx()
    ax2.plot(x, data['mean'] * 100, color='#8b1e1e', marker='o', linewidth=2.2, label='实际违约率')
    ax2.set_ylabel('实际违约率 (%)')
    ax2.set_ylim(0, max(100, data['mean'].max() * 120))
    for i, value in enumerate(data['mean'] * 100):
        ax2.text(i, value + 3, f'{value:.1f}%', ha='center', fontsize=9)
    ax.set_title('样本外PD分档与实际违约率')
    ax.legend(loc='upper left', frameon=False)
    ax2.legend(loc='upper right', frameon=False)
    save(fig, 'fig_default_rate.png')


def plot_ranking(scores1):
    top = scores1.nsmallest(20, '违约概率').sort_values('违约概率', ascending=True)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    color = [COLORS[g] for g in top['风险等级']]
    ax.barh(top['企业代号'], top['违约概率'] * 100, color=color)
    ax.invert_yaxis()
    ax.set_xlabel('样本外违约概率 (%)')
    ax.set_title('附件1低风险企业排名 Top 20')
    save(fig, 'fig_ranking.png')


def plot_rate_churn(churn_model):
    rates = np.asarray(churn_model['rates']) * 100
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for grade, color in [('A', '#0f766e'), ('B', '#2563a6'), ('C', '#d97706')]:
        ax.plot(rates, np.asarray(churn_model['curves'][grade]) * 100, marker='o', markersize=3, linewidth=2, label=f'{grade}级', color=color)
    ax.set_xlabel('贷款年利率 (%)')
    ax.set_ylabel('客户流失率 (%)')
    ax.set_title('单调约束下的利率与客户流失率')
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    save(fig, 'fig_rate_churn.png')


def plot_coefficients(coefficients):
    data = coefficients.head(12).sort_values('标准化系数')
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    colors = np.where(data['标准化系数'] > 0, '#b42318', '#0f766e')
    ax.barh(data['特征'], data['标准化系数'], color=colors)
    ax.axvline(0, color='#64748b', linewidth=0.8)
    ax.set_xlabel('标准化逻辑回归系数')
    ax.set_title('监督违约模型主要特征方向')
    save(fig, 'fig_weights.png')


def plot_crosstab(scores1):
    table = pd.crosstab(scores1['信誉评级'], scores1['风险等级']).reindex(index=['A', 'B', 'C', 'D'], columns=['A', 'B', 'C', 'D'], fill_value=0)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(table.to_numpy(), cmap='GnBu')
    for i in range(4):
        for j in range(4):
            value = int(table.iloc[i, j])
            ax.text(j, i, value, ha='center', va='center', color='white' if value > table.to_numpy().max() * 0.55 else '#12212f')
    ax.set_xticks(range(4), [f'模型{g}' for g in table.columns])
    ax.set_yticks(range(4), [f'银行{g}' for g in table.index])
    ax.set_title('模型风险等级与银行原评级')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save(fig, 'fig_crosstab.png')


def plot_sensitivity(trials):
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(trials['试验编号'], trials['Spearman相关系数'], color='#0f766e', linewidth=1.4)
    mean = trials['Spearman相关系数'].mean()
    minimum = trials['Spearman相关系数'].min()
    ax.axhline(mean, color='#2563a6', linestyle='--', label=f'均值 {mean:.4f}')
    ax.axhline(minimum, color='#b42318', linestyle=':', label=f'最小值 {minimum:.4f}')
    ax.set_xlabel('真实权重扰动试验编号')
    ax.set_ylabel('Spearman相关系数')
    ax.set_ylim(max(0.9, minimum - 0.01), 1.001)
    ax.set_title('100次TOPSIS权重扰动稳定性')
    ax.legend(frameon=False)
    save(fig, 'fig_sensitivity.png')


def plot_covid(strategy3):
    data = strategy3.groupby('行业类别')[['原贷款额度', '调整后额度']].mean().sort_values('调整后额度')
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    y = np.arange(len(data))
    ax.barh(y - 0.18, data['原贷款额度'], height=0.34, color='#cbd5e1', label='调整前')
    ax.barh(y + 0.18, data['调整后额度'], height=0.34, color='#0f766e', label='调整后')
    ax.set_yticks(y, data.index)
    ax.set_xlabel('平均贷款额度 (万元)')
    ax.set_title('疫情赔率冲击前后行业平均额度')
    ax.legend(frameon=False)
    save(fig, 'fig_covid.png')


def plot_case_radar(scores1, features1):
    ordered = scores1.sort_values('违约概率')
    selected = pd.concat([ordered.head(1), ordered.iloc[[len(ordered) // 2]], ordered.tail(1)])
    axes = ['PD安全度', 'TOPSIS', '盈利能力', '经营活跃', '收入稳定', '发票合规']
    merged = selected.merge(features1, left_on='企业代号', right_index=True)
    profit = np.clip((merged['利润率'].to_numpy() + 1) / 2, 0, 1)
    stability = 1 / (1 + merged['收入波动率'].to_numpy())
    compliance = 1 - np.clip((merged['销项作废率'] + merged['销项负数金额率']).to_numpy(), 0, 1)
    values = np.c_[1 - merged['违约概率'], merged['TOPSIS得分'], profit, merged['活跃月份比例'], stability, compliance]
    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False)
    angles = np.r_[angles, angles[0]]
    fig, ax = plt.subplots(figsize=(6.4, 6.0), subplot_kw={'polar': True})
    for i, row in merged.reset_index(drop=True).iterrows():
        points = np.r_[values[i], values[i, 0]]
        ax.plot(angles, points, linewidth=2, label=f"{row['企业代号']} ({row['风险等级']}级)")
        ax.fill(angles, points, alpha=0.08)
    ax.set_xticks(angles[:-1], axes)
    ax.set_ylim(0, 1)
    ax.set_title('低、中、高风险企业经营画像')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    save(fig, 'fig_case_radar.png')


def plot_data_scale(data_summary):
    labels = ['附件1', '附件2']
    input_counts = np.array([
        data_summary['attachment1_input_invoices'],
        data_summary['attachment2_input_invoices'],
    ]) / 10000
    output_counts = np.array([
        data_summary['attachment1_output_invoices'],
        data_summary['attachment2_output_invoices'],
    ]) / 10000
    enterprises = [
        data_summary['attachment1_enterprises'],
        data_summary['attachment2_enterprises'],
    ]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.bar(x - 0.18, input_counts, width=0.36, color='#365d73', label='进项发票')
    ax.bar(x + 0.18, output_counts, width=0.36, color='#0f766e', label='销项发票')
    for i in range(2):
        ax.text(i - 0.18, input_counts[i] + 0.6, f'{input_counts[i]:.1f}万', ha='center', fontsize=9)
        ax.text(i + 0.18, output_counts[i] + 0.6, f'{output_counts[i]:.1f}万', ha='center', fontsize=9)
        ax.text(i, -6.2, f'{enterprises[i]}家企业', ha='center', color='#475569', fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel('发票记录数 (万条)')
    ax.set_ylim(-8, max(input_counts.max(), output_counts.max()) * 1.18)
    ax.set_title('两组企业数据规模')
    ax.legend(frameon=False, ncol=2, loc='upper left')
    save(fig, 'fig_data_scale.png')


def plot_pd_distribution(scores1, scores2):
    bins = np.linspace(0, 1, 16)
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.hist(scores1['违约概率'], bins=bins, alpha=0.72, color='#365d73', label='附件1', edgecolor='white')
    ax.hist(scores2['违约概率'], bins=bins, histtype='step', linewidth=2.3, color='#0f766e', label='附件2')
    for threshold, label in [(0.10, 'A/B'), (0.20, 'B/C'), (0.40, 'C/D')]:
        ax.axvline(threshold, color='#64748b', linewidth=0.9, linestyle='--')
        ax.text(threshold + 0.008, ax.get_ylim()[1] * 0.88, label, rotation=90, va='top', fontsize=8, color='#475569')
    ax.set_xlabel('预测违约概率 PD')
    ax.set_ylabel('企业数')
    ax.set_title('附件1与附件2预测PD分布')
    ax.legend(frameon=False)
    save(fig, 'fig_pd_distribution.png')


def plot_profit_breakdown(strategy1, strategy2, strategy3):
    frames = [strategy1, strategy2, strategy3]
    labels = ['问题1', '问题2', '问题3']
    interest = np.array([frame['预计留存后利息'].sum() for frame in frames])
    default_loss = np.array([frame['预计违约损失'].sum() for frame in frames])
    funding = np.array([frame['预计资金成本'].sum() for frame in frames])
    net = interest - default_loss - funding
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(x, interest, width=0.56, color='#0f766e', label='留存后利息')
    ax.bar(x, -default_loss, width=0.56, color='#8da3ad', label='违约损失')
    ax.bar(x, -funding, width=0.56, bottom=-default_loss, color='#cbd5e1', label='资金成本')
    ax.axhline(0, color='#64748b', linewidth=0.8)
    for i, value in enumerate(net):
        ax.text(i, interest[i] + max(interest) * 0.035, f'净收益 {value:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x, labels)
    ax.set_ylabel('金额 (万元)')
    ax.set_title('三问期望收益构成')
    ax.legend(frameon=False, ncol=3, loc='lower center')
    save(fig, 'fig_profit_breakdown.png')


def create_all_charts(scores1, scores2, features1, coefficients, churn_model, sensitivity_trials,
                      strategy1, strategy2, strategy3, data_summary):
    ensure_directories()
    setup_style()
    plot_default_rate(scores1)
    plot_ranking(scores1)
    plot_rate_churn(churn_model)
    plot_coefficients(coefficients)
    plot_crosstab(scores1)
    plot_sensitivity(sensitivity_trials)
    plot_covid(strategy3)
    plot_case_radar(scores1, features1)
    plot_data_scale(data_summary)
    plot_pd_distribution(scores1, scores2)
    plot_profit_breakdown(strategy1, strategy2, strategy3)
    print(f'  图表已写入: {CHART_DIR}')
