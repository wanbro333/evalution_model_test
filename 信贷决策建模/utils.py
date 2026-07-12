"""
公共工具函数 - 评价模型通用工具
包含：中文字体配置、数据标准化、AHP、熵权法、TOPSIS、GRA、FCE
参考：D:\shumo\evalution models\数学建模评价类模型建立方法.md
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========== 中文字体配置 ==========
def setup_chinese_font():
    """配置matplotlib中文字体"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


# ========== 数据预处理 ==========
def minmax_normalize(data):
    """
    Min-Max归一化到[0,1]
    data: (n_samples, n_indicators) numpy array
    """
    data = np.array(data, dtype=float)
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    # 防止除零
    diff = max_vals - min_vals
    diff[diff == 0] = 1e-12
    return (data - min_vals) / diff


def vector_normalize(data):
    """
    向量归一化: z = x / sqrt(sum(x²))
    data: (n_samples, n_indicators) numpy array
    """
    data = np.array(data, dtype=float)
    norm = np.sqrt(np.sum(data ** 2, axis=0))
    norm[norm == 0] = 1e-12
    return data / norm


# ========== 指标正向化 ==========
def positive_transform(data, indicator_types):
    """
    将各类指标统一转换为极大型（越大越好）
    data: (n_samples, n_indicators) numpy array
    indicator_types: list of str, 每个指标的类别
        支持: 'positive'(极大型), 'negative'(极小型),
              'central'(中间型,用均值作为最优),
              'range'(区间型,用[a,b]设定)
    返回: 正向化后的矩阵
    """
    data = np.array(data, dtype=float)
    n_samples, n_indicators = data.shape
    result = np.zeros_like(data)

    for j in range(n_indicators):
        col = data[:, j]
        itype = indicator_types[j] if j < len(indicator_types) else 'positive'

        if itype == 'positive':
            # 已经是极大型，Min-Max归一化
            col_min, col_max = np.min(col), np.max(col)
            diff = col_max - col_min
            if diff == 0:
                diff = 1e-12
            result[:, j] = (col - col_min) / diff

        elif itype == 'negative':
            # 极小型 → 极大型
            col_max = np.max(col)
            diff = col_max - np.min(col)
            if diff == 0:
                diff = 1e-12
            result[:, j] = (col_max - col) / diff

        elif itype == 'central':
            # 中间型: 越接近best越好
            best = np.mean(col)  # 默认用均值作为最优值
            max_dev = np.max(np.abs(col - best))
            if max_dev == 0:
                max_dev = 1e-12
            result[:, j] = 1 - np.abs(col - best) / max_dev

        elif itype == 'range':
            # 区间型: [a, b]内最好
            a, b = np.percentile(col, 25), np.percentile(col, 75)
            max_dev = max(a - np.min(col), np.max(col) - b)
            if max_dev == 0:
                max_dev = 1e-12
            for i in range(n_samples):
                if col[i] < a:
                    result[i, j] = 1 - (a - col[i]) / max_dev
                elif col[i] > b:
                    result[i, j] = 1 - (col[i] - b) / max_dev
                else:
                    result[i, j] = 1

    return result


# ========== AHP 层次分析法 ==========
def ahp_weights(judgment_matrix, method='geometric_mean'):
    """
    AHP层次分析法求权重
    judgment_matrix: n*n 判断矩阵(1-9标度)
    method: 'geometric_mean'(几何平均法) 或 'eigenvector'(特征值法)
    返回: weights, lambda_max, CI, CR
    """
    A = np.array(judgment_matrix, dtype=float)
    n = A.shape[0]

    # RI表 (n=1~15的随机一致性指标)
    RI = [0, 0.0001, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49,
          1.52, 1.54, 1.56, 1.58, 1.59]

    if method == 'geometric_mean':
        # 几何平均法
        prod = np.prod(A, axis=1)
        w = prod ** (1 / n)
        weights = w / np.sum(w)
    elif method == 'eigenvector':
        # 特征值法
        eigvals, eigvecs = np.linalg.eig(A)
        max_idx = np.argmax(eigvals.real)
        w = np.abs(eigvecs[:, max_idx].real)
        weights = w / np.sum(w)
    else:
        raise ValueError(f"Unknown method: {method}")

    # 一致性检验
    lambda_max = np.sum(np.sum(A, axis=0) * weights)
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0
    CR = CI / RI[n] if n <= 15 else CI / 1.59

    return weights, lambda_max, CI, CR


def print_ahp_result(name, matrix, labels, weights, lambda_max, CI, CR):
    """打印AHP结果"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"指标: {labels}")
    print(f"权重: {np.round(weights, 4)}")
    print(f"λ_max = {lambda_max:.4f}")
    print(f"CI = {CI:.4f}, CR = {CR:.4f}")
    status = "✓ 通过" if CR < 0.10 else "✗ 未通过，需调整判断矩阵"
    print(f"一致性检验: {status}")


# ========== 熵权法 ==========
def entropy_weight(data, epsilon=1e-12):
    """
    熵权法求客观权重
    data: (n_samples, n_indicators) numpy array, 已归一化到[0,1]
    返回: weights, entropy_values, diff_coeff
    """
    data = np.array(data, dtype=float)
    n_samples, n_indicators = data.shape

    # 确保非负
    data = np.clip(data, 0, None)

    # 计算比重 p_ij
    col_sums = np.sum(data, axis=0)
    col_sums[col_sums == 0] = 1e-12
    p = data / col_sums

    # 防止log(0)
    p = np.clip(p, epsilon, 1)

    # 信息熵
    k = 1.0 / np.log(n_samples)
    entropy = -k * np.sum(p * np.log(p), axis=0)

    # 差异系数
    diff = 1 - entropy

    # 权重
    weights = diff / (np.sum(diff) + epsilon)

    return weights, entropy, diff


# ========== TOPSIS ==========
def topsis_score(data, weights):
    """
    TOPSIS综合评价
    data: (n_samples, n_indicators) 已正向化+标准化的矩阵
    weights: (n_indicators,) 权重向量
    返回: scores(C值), d_pos, d_neg
    """
    data = np.array(data, dtype=float)
    weights = np.array(weights, dtype=float)

    # 加权矩阵
    weighted = data * weights

    # 正理想解(每列最大值)和负理想解(每列最小值)
    z_pos = np.max(weighted, axis=0)
    z_neg = np.min(weighted, axis=0)

    # 欧氏距离
    d_pos = np.sqrt(np.sum((weighted - z_pos) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weighted - z_neg) ** 2, axis=1))

    # 相对贴近度
    denominator = d_pos + d_neg
    denominator[denominator == 0] = 1e-12
    scores = d_neg / denominator

    return scores, d_pos, d_neg


# ========== 灰色关联分析 GRA ==========
def grey_relational_grade(data, weights, rho=0.5):
    """
    灰色关联分析
    data: (n_samples, n_indicators) 已归一化到[0,1]的矩阵
    weights: (n_indicators,) 权重
    rho: 分辨系数, 默认0.5
    返回: 关联度 grades, 关联系数矩阵
    """
    data = np.array(data, dtype=float)
    weights = np.array(weights, dtype=float)

    # 加权矩阵
    weighted = data * weights

    # 参考序列(每列最大值)
    ref = np.max(weighted, axis=0)

    # 绝对差
    delta = np.abs(weighted - ref)

    # 两级最大/最小差
    delta_max = np.max(delta)
    delta_min = np.min(delta)

    # 关联系数
    xi = (delta_min + rho * delta_max) / (delta + rho * delta_max)

    # 关联度(加权平均)
    grades = np.sum(xi * weights, axis=1) / np.sum(weights)

    return grades, xi


# ========== 模糊综合评价 FCE ==========
def trapezoid_up(x, a, b, c, d):
    """偏大型梯形隶属函数（值越大隶属度越高）"""
    if x <= a:
        return 0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1
    elif c < x < d:
        return (d - x) / (d - c)
    else:
        return 0


def trapezoid_down(x, a, b, c, d):
    """偏小型梯形隶属函数（值越小隶属度越高）"""
    if x <= a:
        return 1
    elif a < x < b:
        return (b - x) / (b - a)
    elif b <= x <= c:
        return 0
    elif c < x < d:
        return (x - c) / (d - c)
    else:
        return 0


def trapezoid_mid(x, a, b, c, d):
    """中间型梯形隶属函数（中间最优）"""
    if x <= a:
        return 0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1
    elif c < x < d:
        return (d - x) / (d - c)
    else:
        return 0
