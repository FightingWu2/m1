# MCM/ICM 2026 冠军级模型算法知识库 (AI-Native Master Edition)

## [SYSTEM META]

**Role**: Math Modeling Expert System  
**Language**: Chinese (Simplified)  
**Target Audience**: AI Solver / Code Interpreter  
**Directive**: Match user problem description to [Scenario Tags], verify [Data Constraints], and execute using [Tech Stack].

---

## 1. 优化与规划 (Optimization & Programming)

**适用**: 资源分配、调度、路径规划、参数反演

### 1.1 精确规划 (Exact Programming)

**[Tags]**: #Convex #Linear #Constraints #Scipy

**Linear Programming (LP)**: scipy.optimize.linprog. 资源分配基础模型。

**Mixed-Integer Linear Programming (MILP)**: scipy.optimize.milp 或 PuLP. 涉及选址、排班、0/1 决策。

**[O-Prize Check]**: 如果变量超过 1000 个或约束极复杂，必须转为 1.2 或 7.1。

### 1.2 非线性与凸优化 (Non-Linear & Convex)

**[Tags]**: #Physics #Geometry #Gradient

**Sequential Least Squares Programming (SLSQP)**: scipy.optimize.minimize. 处理带约束的非线性目标。

**CVXPY Framework**: 专门用于凸优化问题，比 Scipy 更稳健，适合复杂的金融/工程约束。

### 1.3 多目标优化 (Multi-Objective Optimization) - 🌟 创新点

**[Tags]**: #Trade-off #Pareto

**[Scenario]**: 既要成本最低，又要效率最高。

**NSGA-II / NSGA-III**: 遗传算法变体，寻找帕累托最优前沿 (Pareto Front)。

**Weighted Sum Method**: 基础做法，将多目标转化为单目标（不推荐用于冲击 O 奖，除非配合灵敏度分析）。

**[Python Lib]**: pymoo (推荐), deap

### 1.4 线性/非线性规划 (LP/NLP)

**[Core Logic]**: 在约束条件下优化目标函数。

**[Scenario Tags]**: 生产计划 成本最小化 投资组合

**[Tech Stack]**: scipy.optimize, pulp, cvxpy

### 1.5 多目标规划 (MODM / NSGA-II)

**[Core Logic]**: 寻找多个冲突目标下的帕累托最优解集。

**[Scenario Tags]**: 权衡问题 效益与成本 工程设计

**[Tech Stack]**: pymoo (NSGA-II), geatpy

### 1.6 遗传算法 (GA) / 粒子群 (PSO) / 蚁群 (ACO)

**[Core Logic]**: 启发式智能算法，模拟生物进化或群体行为搜索全局最优。

**[Scenario Tags]**: NP 难问题 TSP 问题 复杂函数寻优 离散优化

**[Complexity]**: 高，计算量大。

**[Tech Stack]**: scikit-opt, deap

### 1.7 模拟退火 (Simulated Annealing)

**[Core Logic]**: 允许以一定概率接受差解以跳出局部最优。

**[Scenario Tags]**: 组合优化 排课问题 选址问题

**[Tech Stack]**: scipy.optimize.dual_annealing

### 1.8 动态规划 (DP)

**[Core Logic]**: 分解子问题，存储中间状态，避免重复计算。

**[Scenario Tags]**: 背包问题 最短路径 资源分配

**[Data Constraints]**: 问题需满足最优子结构和无后效性。

**关联模型**:

- [多目标规划](#15-多目标规划-modm--nsga-ii) - 当需要在动态规划中平衡多个目标时
- [智能启发式算法](#7-智能启发式算法-heuristic-algorithms) - 当动态规划问题过于复杂时的替代方案

---

## 2. 评价与决策模型 (Evaluation & Decision Making)

**适用**: 建立指标体系、排名、优选、风险评估

### 2.1 主客观赋权 (Weighting Methods)

**Entropy Weight (熵权法)**: 客观，基于数据波动。

**AHP (层次分析法)**: 主观，基于专家打分。

**[O-Prize Fusion]**: 组合赋权法 (Game Theory Combination Weighting). 利用博弈论组合 AHP 和 熵权法，避免单一方法的片面性。

### 2.2 综合评价 (Comprehensive Evaluation)

**TOPSIS (逼近理想解排序)**: 计算各评价对象与"正理想解"和"负理想解"的欧氏距离，计算相对贴近度。

- **[Core Logic]**: 计算各评价对象与"正理想解"和"负理想解"的欧氏距离，计算相对贴近度。
- **[Scenario Tags]**: 综合排名 优劣分析 多目标决策
- **[Data Constraints]**: 数据需归一化处理。
- **[Tech Stack]**: scipy.spatial.distance, numpy

**Upgrade**: Grey-TOPSIS (引入灰色关联度解决小样本不确定性)。

**Fuzzy Comprehensive Evaluation (模糊综合评价)**: 适用于描述定性指标（如"美观度"、"安全性"）。

- **[Core Logic]**: 利用模糊数学的隶属度理论，将定性指标转化为定量评价。
- **[Scenario Tags]**: 满意度调查 定性指标定量化 风险等级评估
- **[Data Constraints]**: 依赖隶属度函数设计。
- **[Tech Stack]**: skfuzzy (Python) / Matrix Operations

**PCA-DEA**: 利用主成分分析降维后，用数据包络分析 (DEA) 评估投入产出效率。

### 2.3 灰色关联分析 (GRA)

**[Core Logic]**: 测量序列曲线几何形状的相似度。

**[Scenario Tags]**: 小样本分析 关联度计算 系统因素分析

**[Data Constraints]**: 适用于样本量少、信息不完全的"灰色"系统。
**[Tech Stack]**: pandas

**关联模型**:

- [灰色预测 GM(1,1)](#33-小样本预测-small-data) - 同属灰色系统理论，适用于小样本场景

### 2.4 组合评价模型 (AHP-EWM / TOPSIS-Entropy)

**[Core Logic]**: 结合主观（AHP）与客观（熵权）权重，或用熵权优化 TOPSIS。

**[Scenario Tags]**: 高精度综合评价 比赛加分项

**[Recommendation]**: 优先推荐，鲁棒性高于单一模型。

**关联模型**:

- [AHP (层次分析法)](#21-主客观赋权-weighting-methods) - 组合模型的一部分
- [熵权法](#21-主客观赋权-weighting-methods) - 组合模型的另一部分
- [TOPSIS](#22-综合评价-comprehensive-evaluation) - 组合模型的基础

---

## 3. 预测与机器学习 (Prediction & Machine Learning)

**适用**: 时间序列、分类、回归、通过数据找规律

### 3.1 时间序列 (Time Series)

**ARIMA/SARIMA**: statsmodels. 适用于平稳、有周期性的短期数据。

- **[Core Logic]**: 差分自回归移动平均，利用历史数据的自相关性预测未来。
- **[Scenario Tags]**: 短期预测 金融数据 平稳序列 季节性数据
- **[Data Constraints]**: 数据需平稳或差分后平稳，样本量>30 为佳。
- **[Tech Stack]**: statsmodels.tsa.arima

**Prophet**: prophet. Facebook 开源，处理缺失值、节假日效应极强，E/F 题神器。

**LSTM / GRU**: pytorch / keras. 适用于长序列、非线性、高维历史数据。

- **[Core Logic]**: 引入门控机制的 RNN，解决长序列依赖问题。
- **[Scenario Tags]**: 非线性复杂时序 高频数据 股票 天气
- **[Data Constraints]**: 需大量训练数据，需归一化。
- **[Tech Stack]**: pytorch, tensorflow, keras

**关联模型**:

- [马尔可夫链](#35-马尔可夫链-markov-chain) - 适用于状态转移预测
- [蒙特卡洛模拟](#81-蒙特卡洛模拟-monte-carlo) - 适用于不确定性预测

### 3.2 回归与拟合 (Regression & Fitting)

**XGBoost / LightGBM / CatBoost**: xgboost. 结构化数据（表格数据）的 SOTA（最强）模型。比传统回归强得多，提供 Feature Importance（特征重要性）分析。

- **[Core Logic]**: 集成学习（Bagging/Boosting），通过多棵决策树拟合非线性关系。
- **[Scenario Tags]**: 多因子预测 高维特征 抗过拟合
- **[Data Constraints]**: 对缺失值不敏感，无需严格正态分布。
- **[Tech Stack]**: sklearn.ensemble, xgboost

**Gaussian Process Regression (GPR)**: sklearn.gaussian_process. 给出预测的同时提供置信区间（不确定性范围），评委非常喜欢。

### 3.3 小样本预测 (Small Data)

**Grey Prediction GM(1,1)**: 适用于数据极少（<10 个）的指数增长趋势。

- **[Core Logic]**: 累加生成数列弱化随机性，构建微分方程模型。
- **[Scenario Tags]**: 小样本预测 中长期趋势 指数增长
- **[Data Constraints]**: 样本量至少 4 个，级比检验需通过。
- **[Tech Stack]**: Custom Implementation (numpy)

**Metabolic GM(1,1) (新陈代谢灰色模型)**: 滚动预测，精度更高。

**关联模型**:

- [灰色关联分析](#23-灰色关联分析-gra) - 同属灰色系统理论

### 3.4 逻辑增长模型 (Logistic Growth)

**[Core Logic]**: 考虑环境阻力的种群增长模型（S 形曲线）。

**[Scenario Tags]**: 人口预测 产品生命周期 传染病初期

**[Data Constraints]**: 数据呈现"慢-快-慢"的 S 形特征。
**[Tech Stack]**: scipy.optimize.curve_fit

**关联模型**:

- [微分方程](#4-微分方程与动力系统-differential-equations) - 逻辑增长模型是微分方程的一种特殊形式

### 3.5 马尔可夫链 (Markov Chain)

**[Core Logic]**: 下一状态仅取决于当前状态，通过转移矩阵预测长期分布。

**[Scenario Tags]**: 市场占有率 状态演变 人才流动

**[Tech Stack]**: Matrix Power (numpy)

**关联模型**:

- [时间序列预测](#31-时间序列-time-series) - 用于状态转移预测
- [蒙特卡洛模拟](#81-蒙特卡洛模拟-monte-carlo) - 都涉及随机过程

---

## 4. 微分方程与动力系统 (Differential Equations)

**适用**: A 题/B 题核心，传染病、热传导、种群、弹道

### 4.1 常微分方程 (ODE)

**Population Models**: Malthus, Logistic, Lotka-Volterra (捕食者-猎物)。

- **[Core Logic]**: 建立变量变化率的数学方程。
- **[Models]**: Lotka-Volterra (捕食), SIR/SEIR (传染病), 热传导.
- **[Scenario Tags]**: 动态过程 物理扩散 种群竞争
- **[Tech Stack]**: scipy.integrate.odeint

**Epidemic Models**: SI, SIS, SIR, SEIR.

**Innovation**: SEIR-SD (加入社会距离/隔离参数的 SEIR)。

**[Solver]**: scipy.integrate.odeint 或 solve_ivp.

**关联模型**:

- [逻辑增长模型](#34-逻辑增长模型-logistic-growth) - 逻辑增长模型是微分方程的一种形式
- [元胞自动机](#82-元胞自动机-ca) - 离散版本的动态系统模拟

### 4.2 偏微分方程 (PDE)

**[Scenario]**: 变化涉及空间和时间（如热量在金属板上的分布）。

**Heat Equation / Wave Equation**.

**[Solver]**: Finite Difference Method (FDM, 有限差分法) - 需手写 Python 迭代代码。

**关联模型**:

- [有限元方法](#10-融合创新与前沿-fusion--sota----🏆-夺冠核心) - PDE 的另一种数值解法

---

## 5. 图论与复杂网络 (Graph Theory & Network Science)

**适用**: D 题/F 题，交通、社交网络、物流

### 5.1 路径与规划

**Dijkstra / A\***: 最短路径。

**TSP (Traveling Salesman)**: 旅行商问题（使用模拟退火或遗传算法求解）。

**关联模型**:

- [动态规划](#18-动态规划-dp) - TSP 问题的经典求解方法之一
- [智能启发式算法](#7-智能启发式算法-heuristic-algorithms) - 用于解决大规模 TSP 问题

### 5.2 网络分析 (Network Analysis) - 🌟 O 奖热点

**Centrality (中心性)**: Degree, Betweenness, Closeness, Eigenvector. 识别关键节点。

**Community Detection (社区发现)**: Louvain 算法。识别网络中的团伙。

**Robustness Analysis (鲁棒性分析)**: 随机攻击 vs 蓄意攻击节点，观察网络连通性的变化。这是 D 题拿奖的关键步骤。

**[Lib]**: networkx.

### 5.3 图论与网络流

**Models**: Dijkstra, Floyd, Max Flow, MST.

**[Scenario Tags]**: 物流网络 最短路径 管道流量

**[Tech Stack]**: networkx

**关联模型**:

- [优化规划](#1-优化与规划-optimization--programming) - 网络流问题本质上是优化问题

---

## 6. 统计与降维 (Statistics & Dimensionality Reduction)

**适用**: 数据预处理、相关性分析

### 6.1 降维 (Dimensionality Reduction)

**PCA (主成分分析)**: 线性降维。

- **[Core Logic]**: 降维算法，提取主要特征，去除共线性。
- **[Scenario Tags]**: 数据降维 指标筛选 综合评价前置
- **[Tech Stack]**: sklearn.decomposition

**t-SNE / UMAP**: 非线性流形学习，用于高维数据可视化，图表非常漂亮。

### 6.2 统计推断

**Spearman / Pearson Correlation**: 相关系数。

**Canonical Correlation Analysis (CCA)**: 典型相关分析，研究两组变量之间的关系。

### 6.3 聚类 (Clustering)

**K-Means / 层次聚类**: 基于距离的无监督分类。

- **[Core Logic]**: 基于距离的无监督分类。
- **[Scenario Tags]**: 客户细分 图像分割 类别未知
- **[Tech Stack]**: sklearn.cluster

**Support Vector Machine (SVM)**: 寻找最大间隔超平面，核函数处理非线性。

- **[Core Logic]**: 寻找最大间隔超平面，核函数处理非线性。
- **[Scenario Tags]**: 二分类 小样本分类 故障诊断
- **[Tech Stack]**: sklearn.svm

### 6.4 关联规则 (Apriori)

**[Core Logic]**: 挖掘频繁项集。

**[Scenario Tags]**: 购物篮分析 推荐系统

**[Tech Stack]**: mlxtend

### 6.5 典型相关分析 (CCA) / 结构方程 (SEM)

**[Core Logic]**: 分析两组变量间的相关性或潜在因果路径。

**[Scenario Tags]**: 问卷分析 社会科学 多变量关系

**关联模型**:

- [PCA](#61-降维-dimensionality-reduction) - 都属于多元统计分析方法
- [回归分析](#32-回归与拟合-regression--fitting) - CCA 扩展了回归的概念

---

## 7. 智能启发式算法 (Heuristic Algorithms)

**适用**: NP-Hard 问题，无法求出精确解的复杂优化

**Particle Swarm Optimization (PSO, 粒子群)**: 连续变量优化快。

**Genetic Algorithm (GA, 遗传算法)**: 离散变量，鲁棒性强。

**Simulated Annealing (SA, 模拟退火)**: 避免局部最优，适合 TSP 问题。

**Ant Colony Optimization (ACO, 蚁群)**: 专门用于路径寻找。

**[Lib]**: scikit-opt.

**关联模型**:

- [优化规划](#1-优化与规划-optimization--programming) - 智能算法是优化问题的近似解法
- [图论](#5-图论与复杂网络-graph-theory--network-science) - ACO 用于解决图论中的路径问题

---

## 8. 仿真与模拟 (Simulation)

**适用**: 随机性强、规则复杂的系统，ICM 常用

### 8.1 蒙特卡洛模拟 (Monte Carlo)

**[Core Logic]**: 大量随机抽样逼近结果。

**[Scenario Tags]**: 风险评估 不确定性分析 排队仿真

**[Tech Stack]**: numpy.random

**关联模型**:

- [马尔可夫链](#35-马尔可夫链-markov-chain) - 蒙特卡洛常用于马尔可夫链的模拟
- [预测模型](#3-预测与机器学习-prediction--machine-learning) - 用于不确定性量化

### 8.2 元胞自动机 (CA)

**[Core Logic]**: 局部规则决定全局涌现。

**[Scenario Tags]**: 交通流(NS 模型) 火灾蔓延 城市扩张

**[Tech Stack]**: Custom Grid Simulation (numpy)

**关联模型**:

- [微分方程](#4-微分方程与动力系统-differential-equations) - 元胞自动机是偏微分方程的离散版本

### 8.3 Agent-Based Modeling (ABM, 智能体建模)

**[Core Logic]**: ICM 杀手锏。模拟个体交互涌现出的宏观现象（如病毒传播、舆情发酵）。

**[Lib]**: mesa, simpy.

**关联模型**:

- [图论](#5-图论与复杂网络-graph-theory--network-science) - ABM 常用于网络上的传播模拟
- [微分方程](#4-微分方程与动力系统-differential-equations) - ABM 提供了微观基础，微分方程提供了宏观描述

---

## 9. 融合创新与前沿 (Fusion & SOTA) - 🏆 夺冠核心

**Instruction**: To get an "Outstanding", do not use a single model. Use these fusion strategies:

**Optimization + Prediction**: 使用 LSTM 预测未来的需求，将预测值作为 线性规划 的输入参数。

**Clustering + Evaluation**: 先用 K-Means 将对象分类，再对每一类分别建立 AHP 评价体系。

**Physics-Informed Neural Networks (PINNs)**: A/B 题神技。将微分方程（物理定律）作为损失函数项加入神经网络，用 AI 求解 PDE。

**Ensemble Learning (集成学习)**: 结合 ARIMA 和 神经网络 的结果，加权输出。

**关联模型**:

- [LSTM](#31-时间序列-time-series) + [线性规划](#14-线性非线性规划-lpnlp) - 预测优化融合
- [K-Means](#63-聚类-clustering) + [AHP](#21-主客观赋权-weighting-methods) - 聚类评价融合
- [神经网络](#31-时间序列-time-series) + [微分方程](#4-微分方程与动力系统-differential-equations) - PINN 融合
- [ARIMA](#31-时间序列-time-series) + [神经网络](#31-时间序列-time-series) - 集成预测融合

---

## 10. 模型检验与灵敏度分析 (Validation & Sensitivity)

**[Critical Warning]**: A paper without Sensitivity Analysis cannot win a Prize.

**Sensitivity Analysis**: 改变关键参数（±5%, ±10%），观察结果变化率。模型需证明"参数变了，结论依然稳定"。

**Morris Method / Sobol Indices**: 全局灵敏度分析（比简单的控制变量法更高级）。

**Cross-Validation**: 机器学习必须做 k-fold 交叉验证。

**关联模型**:

- 所有模型都应配合此模块使用 - 模型验证是任何建模的必要环节

---

## 11. AI 自动匹配指令 (Auto-Matching Directives)

**IF 题目涉及 "Ranking", "Best Choice", "Assessment"** → USE Section 1 (AHP, TOPSIS, EWM).

**IF 题目涉及 "Future Data", "Trend", "Next Year"** → USE Section 2 (ARIMA, Grey, LSTM).

**IF 题目涉及 "Allocation", "Schedule", "Min/Max Cost"** → USE Section 3 (Programming, GA, PSO).

**IF 题目涉及 "Process", "Spread", "Interaction"** → USE Section 5 (Differential Eq, CA).

**IF Data is "High Dimensional" or "Unlabeled"** → USE Section 6 (PCA, K-Means).

---

## [AI Action Protocol]

When processing the user's prompt, follow this Chain of Thought (CoT):

1. **Deconstruct**: Break down the problem into inputs, outputs, and constraints.

2. **Classify**: Map to Type A-F (Section 0).

3. **Select Baseline**: Choose a simple, standard model from the library (e.g., Simple Regression, AHP).

4. **Select Advanced**: Choose a specialized or fusion model (e.g., XGBoost, Entropy-TOPSIS, NSGA-II).

5. **Implementation Check**: Ensure the selected model has a corresponding Python library listed.

6. **Review Innovation**: Does the solution include Sensitivity Analysis or a Hybrid approach?

7. **Output**: Present the strategy with clear steps and Python library recommendations.
