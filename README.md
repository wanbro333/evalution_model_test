# 中小微企业信贷决策建模
## 2020年数学建模国赛C题

### 项目结构
```
信贷决策建模/
├── main.py              # 主控脚本
├── data_loader.py       # 数据加载与SQLite入库
├── feature_engineer.py  # 特征工程
├── risk_model.py        # 风险评估(AHP+熵权+TOPSIS)
├── credit_strategy.py   # 信贷策略与疫情调整
├── sensitivity.py       # 灵敏度分析
├── visualize.py         # 可视化
├── utils.py             # 评价模型工具库
├── charts/              # 生成的图表
├── credit_data.db       # SQLite数据库
└── 信贷决策建模结果.xlsx # 最终结果
```

### 方法
- 特征工程: 从发票数据构建15个企业级指标
- 风险评估: AHP(主观) + 熵权法(客观) → 组合赋权 → TOPSIS评分
- 信贷策略: 按TOPSIS得分分配额度(10-100万)和利率(4%-15%)
- 突发因素: 行业分类 + 疫情冲击系数调整

### 运行
```bash
cd 信贷决策建模
python main.py
```

### 结果
- 问题1: 123家企业风险评估与信贷策略(总额8000万)
- 问题2: 302家企业风险评估与信贷策略(总额1亿)
- 问题3: 考虑疫情影响的信贷调整策略
