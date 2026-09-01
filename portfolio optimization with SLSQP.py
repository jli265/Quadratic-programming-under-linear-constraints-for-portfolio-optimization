import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

# ==========================================
# 1. 模拟生成测试数据 (实际使用时替换为真实数据)
# ==========================================
np.random.seed(42)
n = 100  # 样本点个数（时间窗口长度）
m = 30   # 行业数量

y = np.random.normal(0.001, 0.02, n)           # 基金净值日增长率
u = np.random.normal(0.0002, 0.005, n)          # 国债指数收益率
X_ind = np.random.normal(0.001, 0.03, (n, m))   # 30个中信一级行业指数收益率

# ==========================================
# 2. 构建设计矩阵与时间权重矩阵
# ==========================================
# 设计矩阵 X: 第一列为常数项1，第二列为国债收益率 u，其余 30 列为行业收益率
X = np.column_stack([np.ones(n), u, X_ind])     # 维度 (n, 32)

# 权重 w_i = e^(i / n)
i_seq = np.arange(1, n + 1)
w = np.exp(i_seq / n)
W = np.diag(w)                                  # 维度 (n, n)

# ==========================================
# 3. 定义目标函数及梯度
# ==========================================
def objective(beta):
    """时间加权残差平方和 S(beta)"""
    residual = y - X @ beta
    return residual.T @ W @ residual

def objective_grad(beta):
    """目标函数的解析梯度 (Jacobian)，用于加速 SLSQP 收敛并提高精度"""
    return -2 * X.T @ W @ (y - X @ beta)

# ==========================================
# 4. 设置约束条件与变量上下界
# ==========================================
# 4.1 变量上下界 (Bounds)
# beta = [alpha, a0, a1, ..., a30]
# alpha 无约束 (-inf, inf)；a0 到 a30 限制在 [0, 1] 之间
lb_bounds = [-np.inf] + [0.0] * 31
ub_bounds = [np.inf] + [1.0] * 31
bounds = Bounds(lb_bounds, ub_bounds)

# 4.2 线性约束 (LinearConstraint)
# 约束 1: 总仓位 a0 + a1 + ... + a30 <= 1  (对应 -inf <= sum(a_j) <= 1)
# 约束 2: 股票仓位 0.60 <= a1 + ... + a30 <= 0.95 (以偏股混合型为例)
A_total = np.array([0] + [1] * 31)     # 对应 sum(a0...a30)
A_stock = np.array([0, 0] + [1] * 30)  # 对应 sum(a1...a30)

A = np.vstack([A_total, A_stock])
lb_cons = [-np.inf, 0.60]
ub_cons = [1.0, 0.95]

linear_constraint = LinearConstraint(A, lb_cons, ub_cons)

# ==========================================
# 5. 执行二次规划求解
# ==========================================
# 初始向量设为 0
beta0 = np.zeros(32)

res = minimize(
    fun=objective,
    x0=beta0,
    jac=objective_grad,
    method='SLSQP',
    bounds=bounds,
    constraints=linear_constraint,
    options={'maxiter': 1000, 'ftol': 1e-9}
)

# ==========================================
# 6. 结果输出
# ==========================================
if res.success:
    beta_opt = res.x
    print(" Optimization Success!")
    print(f"截距项 alpha  : {beta_opt[0]:.6f}")
    print(f"国债仓位 a0   : {beta_opt[1]:.6f}")
    print(f"股票总仓位Sum : {np.sum(beta_opt[2:]):.6f} (符合 60% ~ 95% 约束)")
    print(f"总仓位合计Sum : {np.sum(beta_opt[1:]):.6f} (符合 <= 100% 约束)")
else:
    print(" Optimization Failed:", res.message)