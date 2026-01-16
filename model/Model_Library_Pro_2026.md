# MCM/ICM 2026 算法与模型知识库 (AI-Readiness Pro Version)

## [System Metadata]

- **Source Material**:
  - Theory: "Mathematical Modeling Algorithms and Applications Solutions (3rd Ed)" by Si Shoukui.
  - Implementation: "Python Mathematical Experiments and Modeling" by Si Shoukui.
- **Goal**: Select the most appropriate, innovative, and executable model for MCM/ICM problems.
- **Output Style**: Python-based solution (Scipy, NetworkX, Sklearn, PyTorch).

---

## 1. 优化与规划类 (Optimization & Programming)

**Ref**: Python Book Ch.2 (Math Programming) / Solutions Book (Linear/Integer/Non-linear)

### 1.1 线性规划 (Linear Programming, LP)

- **[Tags]**: #ResourceAllocation #CostMinimization #ExactSolution
- **[Scenario]**: 目标函数和约束条件均为线性（如生产计划、运输问题）。
- **[Solver_Logic]**: 转化为标准型 min c^T x s.t. A_ub x <= b_ub.
- **[Implementation_Key]**:
  - Library: scipy.optimize.linprog
  - Method: 'highs' (recommended for speed/accuracy).

### 1.2 非线性规划 (Non-Linear Programming, NLP)

- **[Tags]**: #ComplexSystems #GeometryOptimization #Physics
- **[Scenario]**: 目标或约束中包含非线性项（如平方、指数、三角函数）。
- **[Solver_Logic]**: 确定初始点 x0，处理等式/不等式约束 constraints。
- **[Implementation_Key]**:
  - Library: scipy.optimize.minimize
  - Method: 'SLSQP' (supports constraints) or 'BFGS'.

### 1.3 整数规划与 0-1 规划 (Integer/Binary Programming)

- **[Tags]**: #AssignmentProblem #Selection #Logistics
- **[Scenario]**: 决策变量必须是整数（选址问题、人员排班）。
- **[Innovation_Path]**: 若问题规模大（NP-Hard），放弃精确解，改用 智能算法 (See Section 7)。
- **[Implementation_Key]**:
  - Library: scipy.optimize.milp (New in Scipy) or pulp library.

### 1.4 动态规划 (Dynamic Programming)

- **[Tags]**: #MultiStageDecision #OptimalSubstructure #Recursion
- **[Scenario]**: 多阶段决策问题，具有最优子结构（背包问题、最短路径、资源分配）。
- **[Solver_Logic]**: 分解为子问题，建立递推关系式，自底向上求解。
- **[Implementation_Key]**:
  - Library: Custom implementation with memoization
  - Method: Recursion with memoization or iterative table filling.

### 1.5 多目标规划 (Multi-Objective Programming)

- **[Tags]**: #TradeOff #ParetoOptimal #MultiCriteria
- **[Scenario]**: 存在多个冲突的目标函数需要同时优化。
- **[Solver_Logic]**: 寻找 Pareto 最优解集，加权求和或约束法处理。
- **[Implementation_Key]**:
  - Library: pymoo (multi-objective optimization)
  - Method: NSGA-II, SPEA2, or weighted sum approach.

---

## 2. 评价与决策类 (Evaluation & Decision)

**Ref**: Python Book Ch.6 (Evaluation) / Solutions Book (AHP/TOPSIS)

### 2.1 层次分析法 (AHP)

- **[Tags]**: #Subjective #Weighting #Hierarchical
- **[Scenario]**: 评价指标无数据，依赖专家打分（需构建判断矩阵）。
- **[Solver_Logic]**: 计算特征值 eig(A) 得到权重，进行一致性检验 (CR < 0.1)。
- **[Implementation_Key]**:
  - Library: numpy (linear algebra).
  - Code_Hint: w, v = np.linalg.eig(matrix); Normalize v to get weights.

### 2.2 熵权法 & TOPSIS (Entropy-TOPSIS)

- **[Tags]**: #Objective #DataDriven #Ranking
- **[Scenario]**: 有具体的数据指标，需要对多个对象排序。
- **[Innovation_Path]**: 熵权-TOPSIS 组合。先用熵权法客观赋权，再用 TOPSIS 计算欧氏距离得分。
- **[Implementation_Key]**:
  - Logic: Standardize data -> Calculate Information Entropy -> Calculate Weights -> Euclidean Distance to Ideal Solution.
  - Library: numpy, pandas.

### 2.3 模糊综合评价 (Fuzzy Comprehensive Evaluation)

- **[Tags]**: #VagueDefinitions #QualitativeToQuantitative
- **[Scenario]**: 评价标准模糊（如"很好"、"一般"），涉及隶属度函数。
- **[Solver_Logic]**: 确定因素集、评语集 -> 构造模糊关系矩阵 R -> 合成运算 B = W \* R。
- **[Implementation_Key]**:
  - Library: numpy (matrix multiplication).
  - Function: Define Membership Functions (Trapezoidal/Triangular).

### 2.4 数据包络分析 (DEA - Data Envelopment Analysis)

- **[Tags]**: #EfficiencyAnalysis #RelativePerformance #Benchmarking
- **[Scenario]**: 评估多个决策单元(DMU)的相对效率（企业绩效评估、资源配置）。
- **[Solver_Logic]**: CCR 模型或 BCC 模型，线性规划求解效率值。
- **[Implementation_Key]**:
  - Library: pyDEA or custom implementation with scipy.optimize
  - Method: CCR (constant returns to scale) or BCC (variable returns to scale).

### 2.5 灰色关联分析 (Grey Relational Analysis)

- **[Tags]**: #Similarity #Correlation #SmallSample
- **[Scenario]**: 小样本情况下的关联度分析，确定影响因素重要性。
- **[Solver_Logic]**: 计算灰色关联系数，得出关联度排序。
- **[Implementation_Key]**:
  - Library: numpy
  - Method: Grey relational coefficient calculation with reference sequence.

---

## 3. 预测与时序分析 (Prediction & Time Series)

**Ref**: Python Book Ch.11 (Time Series) & Ch.6 (Prediction) / Solutions Book (Grey/Regression)

### 3.1 灰色预测 (Grey Prediction GM(1,1))

- **[Tags]**: #SmallSample #ExponentialTrend #ShortTerm
- **[Scenario]**: 数据量极少（4-10 个），呈指数增长或衰减。
- **[Implementation_Key]**:
  - Logic: Cumulative Sum (AGO) -> Least Squares -> Differential Eq solution -> Inverse AGO.
  - Library: Custom Python function (using numpy).

### 3.2 差分自回归移动平均 (ARIMA)

- **[Tags]**: #Seasonality #Stationary #Economics
- **[Scenario]**: 数据平稳或差分后平稳，有周期性。
- **[Implementation_Key]**:
  - Library: statsmodels.tsa.arima.model.ARIMA.
  - Check: ADF Test (statsmodels.tsa.stattools.adfuller) for stationarity.

### 3.3 机器学习回归 (SVR / Random Forest / XGBoost)

- **[Tags]**: #HighDim #NonLinear #BigData
- **[Scenario]**: 特征多、非线性强，需高精度预测。
- **[Implementation_Key]**:
  - Library: sklearn.svm.SVR, sklearn.ensemble.RandomForestRegressor, xgboost.XGBRegressor.
  - Innovation: Use GridSearchCV for hyperparameter tuning.

### 3.4 神经网络与深度学习 (Neural Networks & Deep Learning)

- **[Tags]**: #ComplexPatterns #DeepLearning #NonLinear
- **[Scenario]**: 复杂非线性模式识别，图像、文本、时序预测。
- **[Implementation_Key]**:
  - Library: tensorflow, pytorch, keras
  - Models: MLP, CNN, LSTM, GRU for sequential data.

### 3.5 指数平滑法 (Exponential Smoothing)

- **[Tags]**: #Trend #Seasonal #SimpleForecasting
- **[Scenario]**: 具有趋势或季节性的时间序列数据。
- **[Implementation_Key]**:
  - Library: statsmodels.tsa.holtwinters.ExponentialSmoothing
  - Methods: Simple, Double, Triple (Holt-Winters) exponential smoothing.

### 3.6 Prophet 预测模型

- **[Tags]**: #SeasonalTrends #HolidayEffects #Robust
- **[Scenario]**: 包含节假日效应和多重季节性的时间序列。
- **[Implementation_Key]**:
  - Library: fbprophet (now prophet)
  - Features: Automatic trend, seasonality, holiday detection.

---

## 4. 微分方程与动力系统 (Differential Equations)

**Ref**: Python Book Ch.4 (Diff Eq) / Solutions Book (Population/Spread Models)

### 4.1 常微分方程 (ODE) - 初值问题

- **[Tags]**: #DynamicSystems #Evolution #Continuous
- **[Scenario]**: 描述随时间连续变化的系统（传染病 SIR、种群竞争 Volterra）。
- **[Solver_Logic]**: 定义导数函数 dydt(y, t)。
- **[Implementation_Key]**:
  - Library: scipy.integrate.odeint or solve_ivp.
  - Code_Hint: sol = odeint(model_func, y0, t).

### 4.2 偏微分方程 (PDE)

- **[Tags]**: #SpatialDistribution #HeatEquation #WaveEquation
- **[Scenario]**: 涉及空间和时间的连续变化过程（热传导、波动现象）。
- **[Implementation_Key]**:
  - Library: scipy, finite difference methods
  - Methods: Finite difference, finite element (FEniCS).

### 4.3 差分方程 (Difference Equations)

- **[Tags]**: #DiscreteTime #Iterative
- **[Scenario]**: 时间离散的演化过程（每年的种群数量，非连续）。
- **[Implementation_Key]**:
  - Method: Python for loop iteration.

### 4.4 动力系统与混沌理论 (Dynamical Systems & Chaos Theory)

- **[Tags]**: #Chaos #SensitiveDependence #Attractors
- **[Scenario]**: 非线性动力系统中的混沌行为研究。
- **[Implementation_Key]**:
  - Library: numpy, matplotlib for visualization
  - Models: Logistic map, Lorenz system, Rössler attractor.

---

## 5. 图论与网络模型 (Graph Theory)

**Ref**: Python Book Ch.7 (Graph Theory) / Solutions Book (Shortest Path/MST)

### 5.1 最短路径 (Shortest Path)

- **[Tags]**: #Navigation #Logistics #CostMinimization
- **[Scenario]**: 寻找网络中两点间成本/距离最小的路径（Dijkstra, Floyd）。
- **[Implementation_Key]**:
  - Library: networkx
  - Func: nx.dijkstra_path(G, source, target), nx.shortest_path_length.

### 5.2 最小生成树 (Minimum Spanning Tree, MST)

- **[Tags]**: #Connection #Infrastructure #LowCost
- **[Scenario]**: 连通所有节点且总边权最小（铺设电缆、管道）。
- **[Implementation_Key]**:
  - Library: networkx
  - Func: nx.minimum_spanning_tree(G).

### 5.3 网络流 (Network Flow)

- **[Tags]**: #Capacity #Traffic #SupplyChain
- **[Scenario]**: 管道最大流量、物资最大运输量。
- **[Implementation_Key]**:
  - Library: networkx
  - Func: nx.maximum_flow(G, 's', 't').

### 5.4 社交网络分析 (Social Network Analysis)

- **[Tags]**: #Centrality #CommunityDetection #Influence
- **[Scenario]**: 分析网络中节点的重要性、社区结构、影响力传播。
- **[Implementation_Key]**:
  - Library: networkx, igraph
  - Metrics: Betweenness, closeness, eigenvector centrality; community detection algorithms.

### 5.5 PageRank 算法

- **[Tags]**: #Ranking #WebAnalysis #Authority
- **[Scenario]**: 网页重要性排序、影响力分析。
- **[Implementation_Key]**:
  - Library: networkx.pagerank
  - Algorithm: Iterative computation of node importance.

---

## 6. 统计与多元分析 (Statistics & Multivariate)

**Ref**: Python Book Ch.8 (Multivariate) / Solutions Book (PCA/Cluster)

### 6.1 主成分分析 (PCA)

- **[Tags]**: #DimensionalityReduction #Preprocessing
- **[Scenario]**: 变量太多且相关性强，需要降维处理。
- **[Implementation_Key]**:
  - Library: sklearn.decomposition.PCA.
  - Output: Explained variance ratio.

### 6.2 聚类分析 (K-Means / Hierarchical / DBSCAN)

- **[Tags]**: #Classification #PatternRecognition #Unsupervised
- **[Scenario]**: 将对象自动分类（客户分群、区域划分）。
- **[Implementation_Key]**:
  - Library: sklearn.cluster.KMeans, AgglomerativeClustering, DBSCAN; scipy.cluster.hierarchy.
  - Vis: Dendrogram (树状图) using scipy.

### 6.3 回归分析 (Regression Analysis)

- **[Tags]**: #Relationship #Prediction #StatisticalInference
- **[Scenario]**: 分析变量间的因果关系，进行预测。
- **[Implementation_Key]**:
  - Library: sklearn.linear_model, statsmodels
  - Types: Linear, polynomial, ridge, lasso, elastic net regression.

### 6.4 方差分析 (ANOVA)

- **[Tags]**: #SignificanceTest #GroupComparison #ExperimentalDesign
- **[Scenario]**: 比较多个组均值差异的显著性。
- **[Implementation_Key]**:
  - Library: scipy.stats.f_oneway, statsmodels.stats.anova
  - Types: One-way, two-way ANOVA.

### 6.5 判别分析 (Discriminant Analysis)

- **[Tags]**: #Classification #Separation #FeatureSelection
- **[Scenario]**: 基于特征将样本分类。
- **[Implementation_Key]**:
  - Library: sklearn.discriminant_analysis.LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
  - Method: Fisher's linear discriminant, quadratic discriminant.

### 6.6 因子分析 (Factor Analysis)

- **[Tags]**: #LatentVariables #DimensionReduction #StructureDiscovery
- **[Scenario]**: 探索观测变量背后的潜在因子结构。
- **[Implementation_Key]**:
  - Library: sklearn.decomposition.FactorAnalysis, factor_analyzer
  - Method: Principal axis factoring, maximum likelihood.

---

## 7. 概率与随机模型 (Probability & Stochastic Models)

**Ref**: Additional Statistical References

### 7.1 蒙特卡罗模拟 (Monte Carlo Simulation)

- **[Tags]**: #RandomSampling #Uncertainty #NumericalIntegration
- **[Scenario]**: 解决概率问题、数值积分、风险评估。
- **[Implementation_Key]**:
  - Library: numpy.random, scipy.stats
  - Method: Random sampling and statistical estimation.

### 7.2 马尔可夫链 (Markov Chains)

- **[Tags]**: #Memoryless #StateTransition #StochasticProcess
- **[Scenario]**: 系统状态转移只依赖当前状态（天气预测、市场状态转换）。
- **[Implementation_Key]**:
  - Library: numpy for transition matrix, custom implementation
  - Method: Transition probability matrix computation.

### 7.3 排队论 (Queuing Theory)

- **[Tags]**: #WaitingLines #ServiceSystems #PerformanceMetrics
- **[Scenario]**: 服务系统性能分析（银行排队、交通流）。
- **[Implementation_Key]**:
  - Library: Custom implementation or specialized queuing libraries
  - Models: M/M/1, M/M/c, M/G/1 queue models.

### 7.4 生存分析 (Survival Analysis)

- **[Tags]**: #TimeToEvent #Censoring #Reliability
- **[Scenario]**: 分析事件发生时间（产品寿命、患者生存期）。
- **[Implementation_Key]**:
  - Library: lifelines, scikit-survival
  - Methods: Kaplan-Meier estimator, Cox proportional hazards model.

---

## 8. 智能启发式算法 (Intelligent Algorithms)

**Ref**: Python Book Ch.10 (Intelligent Algorithms) / Solutions Book (Complex Optimization)

### 8.1 粒子群算法 (PSO)

- **[Tags]**: #SwarmIntelligence #GlobalOptimization #Continuous
- **[Scenario]**: 连续域全局优化问题。
- **[Implementation_Key]**:
  - Library: pyswarm, scipy.optimize with custom implementation
  - Logic: Initialize Population -> Evaluate Fitness -> Update Velocity/Position -> Loop.

### 8.2 遗传算法 (Genetic Algorithm, GA)

- **[Tags]**: #EvolutionaryComputation #BioInspired #GlobalSearch
- **[Scenario]**: 复杂组合优化、参数优化。
- **[Implementation_Key]**:
  - Library: DEAP, pygad, custom implementation
  - Logic: Selection -> Crossover -> Mutation -> Replacement.

### 8.3 模拟退火 (Simulated Annealing, SA)

- **[Tags]**: #Probabilistic #GlobalOptimization #Metropolis
- **[Scenario]**: 避免局部最优，寻找全局最优。
- **[Implementation_Key]**:
  - Library: scipy.optimize.basinhopping, custom implementation
  - Logic: Temperature schedule -> Neighbor selection -> Accept/reject based on Metropolis criterion.

### 8.4 蚁群算法 (Ant Colony Optimization, ACO)

- **[Tags]**: #SwarmIntelligence #PathOptimization #Pheromone
- **[Scenario]**: TSP、路径优化、组合优化问题。
- **[Implementation_Key]**:
  - Library: Custom implementation or specialized libraries
  - Logic: Pheromone trail -> Probabilistic path selection -> Trail update.

### 8.5 蜂群算法 (Artificial Bee Colony, ABC)

- **[Tags]**: #SwarmIntelligence #NatureInspired #Optimization
- **[Scenario]**: 连续域优化问题，特别是多峰函数优化。
- **[Implementation_Key]**:
  - Library: Custom implementation
  - Logic: Employed bees -> Onlooker bees -> Scout bees cycle.

---

## 9. 插值与拟合 (Interpolation & Fitting)

**Ref**: Python Book Ch.3

### 9.1 插值 (Interpolation)

- **[Tags]**: #MissingData #ContourPlot #Smooth
- **[Scenario]**: 经过已知数据点，填补空缺或绘制平滑曲线/曲面。
- **[Implementation_Key]**:
  - Library: scipy.interpolate.interp1d, griddata (for 2D), scipy.interpolate.Rbf
  - Methods: Linear, cubic, spline interpolation.

### 9.2 拟合 (Curve Fitting)

- **[Tags]**: #TrendLine #ParameterEstimation #PhysicsLaw
- **[Scenario]**: 寻找一条曲线反映数据趋势，允许有误差。
- **[Implementation_Key]**:
  - Library: scipy.optimize.curve_fit, numpy.polyfit
  - Logic: Define target function f(x, a, b), fit to get parameters.

### 9.3 样条插值 (Spline Interpolation)

- **[Tags]**: #SmoothCurves #FlexibleFitting #PiecewisePolynomial
- **[Scenario]**: 需要光滑曲线通过数据点。
- **[Implementation_Key]**:
  - Library: scipy.interpolate.splrep/splev, scipy.interpolate.UnivariateSpline
  - Method: Cubic splines, B-splines.

---

## 10. 仿真建模 (Simulation Modeling)

**Ref**: Additional Simulation References

### 10.1 系统动力学 (System Dynamics)

- **[Tags]**: #FeedbackLoops #StockFlow #PolicyAnalysis
- **[Scenario]**: 分析复杂系统的动态行为（人口增长、经济发展）。
- **[Implementation_Key]**:
  - Library: Custom implementation with differential equations
  - Components: Stocks, flows, feedback loops, delays.

### 10.2 离散事件仿真 (Discrete Event Simulation)

- **[Tags]**: #EventDriven #QueueSystems #Manufacturing
- **[Scenario]**: 模拟离散事件驱动的系统（生产线、服务系统）。
- **[Implementation_Key]**:
  - Library: simpy (process-based discrete-event simulation)
  - Elements: Events, processes, resources, stores.

### 10.3 有限元分析 (Finite Element Analysis)

- **[Tags]**: #StructuralAnalysis #PDEApproximation #Engineering
- **[Scenario]**: 结构力学、热传导等物理问题数值求解。
- **[Implementation_Key]**:
  - Library: FEniCS, deal.II (through Python interfaces)
  - Method: Domain discretization, basis functions, weak formulation.

---

## [AI Activation Instructions]

When you receive this knowledge base during a competition, use the following prompt to activate AI capabilities:

"Based on the provided 'Model Algorithm Collection', analyze the current problem [Problem Description].

1. Identify the problem type by matching tags (#Keywords) from the knowledge base
2. Recommend 1-2 most suitable baseline models and 1 advanced/innovative model
3. Provide specific Python library functions and code framework based on the implementations mentioned in the knowledge base
4. Consider hybrid approaches combining multiple techniques for innovation
5. Prioritize models based on data characteristics, problem constraints, and computational feasibility"

---

## [Model Selection Heuristics]

- **Small sample size (<20)**:优先考虑 Grey Prediction, Statistical methods, Simple ML
- **Large sample size (>1000)**: Consider Deep Learning, Advanced ML, Ensemble methods
- **Time-dependent**: ARIMA, Exponential Smoothing, Neural Networks (LSTM/GRU)
- **Multiple objectives**: Multi-objective optimization, Pareto analysis
- **Uncertainty/Risk**: Monte Carlo, Stochastic models, Robust optimization
- **Network structure**: Graph theory, Network analysis, Centrality measures
- **Complex nonlinearity**: Neural networks, SVM, Ensemble methods
- **Limited domain knowledge**: Data-driven approaches, Black-box optimization
