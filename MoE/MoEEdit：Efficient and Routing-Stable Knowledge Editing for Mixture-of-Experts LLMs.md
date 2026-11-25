文章链接：http://arxiv.org/abs/2602.10965

## 问题

Dense模型的知识编辑通常通过修改FFN的参数，将特定Key映射到新的Value，同时利用Preservation Set约束参数更新，避免影响无关知识。但直接将这类方法迁移到MoE模型，会遇到三个问题：
1. 计算开销大：MoE包含大量Expert，如果像Dense模型一样联合优化所有Expert的参数，计算和内存开销会随Expert数量急剧增加。
2. Expert Coupling：MoE的输出是多个Expert输出的Router加权组合，修改某个Expert会影响整体输出，多个Expert之间存在耦合，无法简单地独立编辑。
3. Routing Distribution Shift：修改某一层的Expert参数会改变该层输出，进而影响后续层Router的输入，使其选择不同的Expert，破坏模型原有的路由模式和知识处理路径，导致编辑局部性与模型稳定性下降。
## 总结

提出MoEEdit，一种专门针对MoE模型的参数修改型知识编辑框架。

核心是通过Per-Expert Null-Space Projection，为每个Expert构造独立的零空间投影矩阵，使参数更新在Preservation Set对应的特征方向上不产生作用，从而保持其输出不变，避免后续Router输入发生变化。

同时，利用Randomized Block Coordinate Descent（BCD），将所有Expert的联合优化问题拆分为逐Expert的小规模优化问题，降低计算和内存开销。

最终实现知识编辑效果、泛化能力、局部性、路由稳定性与计算效率之间的兼顾。

## 思路

作者首先发现，MoE知识编辑除了需要考虑目标知识是否成功修改，还需要考虑编辑是否改变后续层的Router输入。因为即使当前层成功编辑，前面层输出的变化也可能导致后续层选择不同的Expert，产生连锁影响。

因此，作者希望在编辑目标知识的同时，保证Preservation Set对应的模型输出不变。

具体分为两个部分：

- Routing Stability：通过零空间投影保证路由稳定。

  首先利用Softmax的一阶泰勒展开分析路由漂移：
\[
\delta g\approx J_{\rm sm}(s)E^\top\delta u
\]

  发现只有隐藏表示扰动在Router Embedding张成空间中的分量，才会影响路由权重。

  因此，作者进一步采用更强的约束：直接保证Preservation Set在当前层的输出不发生变化，即令其隐藏表示扰动为零，从而避免后续Router输入发生变化。

  具体做法是：

    1. 对每个Expert分别收集Preservation Set对应的Key，构造特征矩阵。
    2. 对特征矩阵的协方差进行特征值分解，提取零空间对应的特征向量。
    3. 构造投影矩阵，将Expert参数更新限制在Preservation Features张成空间的正交补空间内。

  最终通过硬约束保证编辑不会改变Preservation Set在各Expert上的输出，而不需要额外优化Preservation Loss。

- Efficient Optimization：通过随机分块坐标下降解决专家耦合和计算开销。
作者首先将所有Expert的更新矩阵拼接，推导出整个MoE层的全局闭式解，但发现需要求解规模为 \(Nd_k\) 的大型线性系统，计算开销过高。
因此提出Randomized BCD：
1. 将每个Expert的更新矩阵作为一个独立优化块。
2. 每次固定其他Expert的参数，只优化当前Expert。
3. 通过扣除其他Expert已经提供的输出，计算当前Expert需要承担的剩余目标输出（Residual）。
4. 将当前Expert的优化转化为一个小规模的岭回归问题，得到闭式解。
5. 随机遍历Expert，反复更新，直到整体目标收敛。

这样既考虑了多个Expert对最终输出的共同影响，又避免了直接求解大型联合优化问题。

## 使用方法和技巧

### Key-Value Memory：将Expert视为独立的知识存储单元

对于第 \(n\) 个Expert：

\[
E_n(u)=W_nk_n
\]

其中：
- \(k_n\)：Expert经过Up、Gate和激活函数得到的Key。
- \(W_n=W_{\rm down}^{(n)}\)：Expert的输出投影矩阵。
- \(W_nk_n\)：Expert输出的Value。

MoE整体输出：

\[
v=\sum_{n=1}^{N}g_nW_nk_n
\]

知识编辑就是修改各Expert的 \(W_n\)，使其能够产生目标输出。

### Routing Distribution Shift：分析路由漂移

定义Router Embedding矩阵：

\[
E=[e_1,e_2,\ldots,e_N]
\]

Router logits：

\[
s=E^\top u
\]

当输入隐藏表示发生扰动 \(\delta u\) 时：

\[
\delta s=E^\top\delta u
\]

进一步通过Softmax的一阶泰勒展开：

\[
\delta g\approx J_{\rm sm}(s)E^\top\delta u
\]

说明只有隐藏表示在Router Embedding张成空间中的扰动才会影响路由权重。

此外，使用两种指标衡量路由稳定性：
- Routing Similarity（RS）：使用Jaccard相似度衡量编辑前后Top-K Expert集合的重合程度。
- KL Divergence：衡量编辑前后Router权重分布的差异。

### Per-Expert Null-Space Projection：专家级零空间投影

对每个Expert分别收集Preservation Set的Key：

\[
K_n^0=[k_{i,n}]_{i\in\mathcal P}
\]

对其协方差矩阵进行特征值分解：

\[
K_n^0(K_n^0)^\top=U_n\Lambda_nU_n^\top
\]

选取接近零的特征值对应的特征向量 \(U_n^0\)，构造投影矩阵：

\[
P_n=U_n^0(U_n^0)^\top
\]

使得：

\[
P_nK_n^0\approx0
\]

将Expert更新重新参数化为：

\[
\boxed{\Delta_n=\hat{\Delta}_nP_n}
\]

从而保证：

\[
\Delta_nk_{i,n}
=\hat{\Delta}_nP_nk_{i,n}
\approx0,\quad i\in\mathcal P
\]

即Preservation Set在每个Expert上的输出不发生变化，进而保持MoE整体输出和后续Router输入稳定。

### Randomized Block Coordinate Descent：随机分块坐标下降

将全局优化问题拆分为逐Expert优化问题，每次只更新一个Expert，其他Expert保持不变。

对于当前Expert \(n\)，定义其他Expert提供输出后的剩余目标：

\[
r_i^{(-n)}
=
v_i-
\sum_{\ell\neq n}
g_{i,\ell}
(W_\ell k_{i,\ell}+
\hat{\Delta}_\ell\tilde{k}_{i,\ell})
\]

其中：

\[
\tilde{k}_{i,n}=P_nk_{i,n}
\]

将当前Expert的优化转化为岭回归：

\[
\min_{\hat{\Delta}_n}
\sum_i
\left\|
r_i^{(-n)}
-g_{i,n}\hat{\Delta}_n\tilde{k}_{i,n}
\right\|^2
+\lambda\|\hat{\Delta}_n\|_F^2
\]

通过求解正规方程得到闭式更新：

\[
\hat{\Delta}_n^\star=B_nM_n^{-1}
\]

其中：

\[
M_n=
\sum_i g_{i,n}^2
\tilde{k}_{i,n}\tilde{k}_{i,n}^{\top}
+\lambda I
\]

\[
B_n=
\sum_i g_{i,n}r_i^{(-n)}
\tilde{k}_{i,n}^{\top}
\]

最后将更新投影回允许的参数空间：

\[
\Delta_n^\star=\hat{\Delta}_n^\star P_n
\]

### Engineering Optimization：计算效率优化
- Randomized Order：随机排列Expert的更新顺序。
- Active Experts Only：只更新当前Mini-batch中实际激活的Expert。
- Streaming Accumulation：流式累计优化所需的矩阵，减少内存占用。
- Key Caching：缓存投影后的Key，避免重复计算。
- Cholesky Decomposition：使用Cholesky分解求解线性系统。
- Diagonal Loading：通过对角加载提高矩阵求解的数值稳定性。
相比全局联合求解，单个Expert只需要处理 \(d_k\times d_k\) 的矩阵，将计算规模从依赖全部Expert数量的联合优化转化为逐Expert优化。

一句话概括： MoEEdit的核心不是寻找应该修改哪个Expert，而是通过零空间投影保证无关样本的Expert输出和后续路由不变，再利用BCD高效解决多个Expert共同参与知识编辑时的参数耦合问题。

