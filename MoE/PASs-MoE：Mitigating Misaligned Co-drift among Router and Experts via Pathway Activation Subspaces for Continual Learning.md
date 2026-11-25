文章链接：https://aclanthology.org/2026.acl-long.1474/

## 问题

在多模态大模型的持续指令微调（CIT）中，MoE-LoRA 通过多个 LoRA Expert 分担不同任务的参数更新，以缓解单个 LoRA 在连续学习过程中产生的任务干扰。

但现有方法通常联合优化 Router 和 Expert 参数，却没有建立二者之间统一的功能性参照，导致 Misaligned Co-drift（错位协同漂移）：
1. Router 漂移： 随着新任务的学习，Router 参数发生变化，同一个历史任务输入可能被重新分配给其他 Expert，破坏原有的输入-Expert 专门化关系。
2. Expert 漂移： 即使历史输入仍然被分配给原来的 Expert，其内部 LoRA 参数 \(A_e,B_e\) 也会因为新任务训练而发生变化，导致原有处理能力退化。
3. 二者相互加剧： Router 的路由变化与 Expert 内部功能变化缺乏统一的功能性依据，共同导致历史知识遗忘。
此外，已有 MoE-LoRA 方法通常通过增加 Expert 数量来隔离任务，但这会增加模型参数量和存储开销；而直接约束所有 Expert 参数，又会限制模型学习新任务的能力。

## 总结

提出 PASs-MoE，一种基于 Pathway Activation Subspaces（PASs，路径激活子空间）的固定容量 MoE-LoRA 持续学习框架。

核心思想是：从每个 LoRA Expert 的低秩结构中提取其功能性激活信号，将其作为统一依据，同时指导 Expert 路由和历史知识保护。

包含两个核心模块：
- PASs-guided Reweighting（PASs-RW）： 利用各 Expert 的低秩激活能量直接计算路由权重，替代独立的 Router 网络，使路由决策与 Expert 实际功能保持一致。
- PASs-aware Rank Stabilization（PASs-RS）： 根据历史任务中各 Expert 的低秩方向激活重要性，对重要的 rank direction 施加选择性正则化，减少 Expert 内部参数漂移。
最终在不增加 Expert 数量和模型参数的情况下，同时缓解路由漂移与 Expert 功能漂移，兼顾新任务适应和旧任务保持。

## 思路

作者首先发现，现有 MoE-LoRA 的 Router 和 Expert 虽然共同训练，但二者实际上依据不同的参数空间学习：Router 在独立的路由空间中学习输入与 Expert 的对应关系，而 Expert 则通过自身的 LoRA 低秩路径学习具体功能。

因此，作者希望找到一种直接来自 Expert 自身结构的功能性信号，使路由决策和历史知识保护都能够以此为依据。

从 LoRA 的基本结构出发：

\[
\Delta y_e=B_e(A_eh)
\]

作者将其拆解为：

\[
z_e=A_eh
\]

\[
\Delta y_e=B_ez_e
\]

其中 \(A_e\) 将输入投影到低秩空间，\(z_e\) 表示输入在各个低秩方向上的响应，\(B_e\) 再将其映射回输出空间。

作者据此提出 PASs：

\[
S_e=\operatorname{span}(A_e^\top)
\]

即由 LoRA 下投影矩阵 \(A_e\) 的行向量张成的子空间。它刻画了 Expert 能够通过自身低秩路径响应的输入方向。

进一步观察到，\(z_e=A_eh\) 不仅可以反映输入在 Expert 各个低秩方向上的响应，还可以作为衡量不同 rank direction 历史重要性的依据。

因此，作者围绕同一个 PASs 信号设计了两个互补模块：
- PASs-RW： 利用整个低秩响应 \(z_e\) 计算 Expert 的激活能量，决定当前输入应该由哪些 Expert 处理，从而使路由依据与 Expert 实际功能保持一致。
- PASs-RS： 利用 \(z_e\) 的每个分量 \(z_{e,k}\)，统计各低秩方向在历史任务中的重要性，并对重要方向对应的 \(A_e,B_e\) 参数施加约束，防止其在新任务训练中发生过度变化。

这样，PASs-RW 负责解决 Expert 之间的路由错位，PASs-RS 负责解决 Expert 内部的功能漂移，二者共同缓解 Misaligned Co-drift。

## 使用的方法和技巧

### Pathway Activation Subspace（PASs）：路径激活子空间

从 LoRA 的低秩分解中提取功能性激活空间：

\[
\Delta y_e=B_e(A_eh)
\]

\[
z_e=A_eh
\]

定义：

\[
S_e=\operatorname{span}(A_e^\top)
\]

其中 \(z_e\in\mathbb R^r\)，每个分量对应一个 rank direction 的响应强度。

与从中间层激活统计中提取子空间不同，PASs 直接由 LoRA 参数矩阵 \(A_e\) 决定，使子空间中的每个方向都与具体的 LoRA 低秩通道对应。

### PASs-guided Reweighting（PASs-RW）：基于激活能量的隐式自路由

不再使用独立 Router 网络，而是根据每个 Expert 的低秩激活能量计算路由权重：

\[
s_e(h)=\frac{1}{r}\|A_eh\|_2^2
\]

然后使用 Softmax 得到混合权重：

\[
\pi_e(h)=
\frac{\exp(s_e(h))}
{\sum_{e'=1}^{E}\exp(s_{e'}(h))}
\]

激活能量越大，说明当前输入在该 Expert 的低秩路径上产生的响应越强，对应的路由权重也越大。

这里采用对全部 Expert 进行 Softmax，而不是 Top-k 选择，从而让各 Expert 根据自身响应强度参与当前输入的处理。

### PASs-aware Rank Stabilization（PASs-RS）：基于历史重要性的低秩方向选择性保护

将 \(A_e\) 按行拆分为 \(r\) 个低秩方向：

\[
z_{e,k}=a_{e,k}^{\top}h
\]

定义第 \(t\) 个任务中，第 \(e\) 个 Expert 的第 \(k\) 个 rank direction 的重要性：

\[
I_{e,k}(t)=
\mathbb E_{h\sim D_t}
\left[
\pi_e(h)z_{e,k}(h)^2
\right]
\]

其中：
- \(\pi_e(h)\)：当前输入由 Expert \(e\) 负责的程度。
- \(z_{e,k}^2\)：当前输入在第 \(k\) 个低秩方向上的响应强度。
通过二者相乘，衡量该低秩方向对当前任务的响应重要性。

进一步累积历史任务的重要性：

\[
I_{e,k}^{\mathrm{agg}}(t-1)
=
\sum_{t'=1}^{t-1}I_{e,k}(t')
\]

再将其归一化并截断，得到稳定化权重 \(w_{e,k}\)。

根据重要性对 LoRA 的输入侧和输出侧参数同时施加约束：

\[
\mathcal L_{\mathrm{stab}}^A
=
\sum_{e,k}w_{e,k}
\left\|
a_{e,k}^{(t)}-a_{e,k}^{(t-1)}
\right\|_2^2
\]

\[
\mathcal L_{\mathrm{stab}}^B
=
\sum_{e,k}w_{e,k}
\left\|
b_{e,k}^{(t)}-b_{e,k}^{(t-1)}
\right\|_2^2
\]

最终训练目标：

\[
\mathcal L=
\mathcal L_{\mathrm{task}}
+\lambda_A\mathcal L_{\mathrm{stab}}^A
+\lambda_B\mathcal L_{\mathrm{stab}}^B
\]

历史重要性越高的低秩方向，受到的约束越强；不重要的方向则保留更大的更新自由度，以适应新任务。

### Router Stability：验证路由漂移是否得到缓解

固定一批历史任务输入，分别记录模型刚学完对应任务时和完成全部任务训练后的 Expert gating distribution。

使用负平均 Jensen-Shannon Divergence 衡量两个阶段的路由分布变化：

\[
\mathrm{Stability}=-\mathrm{MeanJS}
\]

数值越大（越接近 0），说明路由分布越稳定。

实验发现，PASs-MoE 相比普通 MoE-LoRA 具有更小的路由漂移，同时保持了更高的历史任务准确率。

### Random Rank Regularization：验证重要性选择是否有效

设计随机正则化对照实验：保持其他模块不变，将 PASs-RS 根据历史重要性分配的权重替换为随机权重。

结果显示：
|方法|	AP|	BWT|
|---|---|---|
|Random Regularization|	45.12|	-4.90|
|PASs-RS（Ours）|	48.46|	-2.15|

说明随意约束部分低秩方向并不能有效保留历史知识，必须根据历史任务的实际激活情况识别重要方向。

### 消融实验与持续学习评估

在 MLLM-CTBench 上进行七个任务的连续训练，使用 LLaVA-v1.5-7B 作为基础模型，设置 6 个 Expert，每个 LoRA 的 rank 为 128。

通过 Average Performance（AP）和 Backward Transfer（BWT）评估最终任务性能与遗忘程度。

主要结果：
| 方法 | AP | BWT |
| --- | --- | --- |
| MoELoRA（Softmax） | 43.36 | -6.64|
| PASs-RW|	46.31|	-4.24|
| PASs-MoE（完整方法）|	48.46|	-2.15|

消融结果表明，PASs-RW 能够独立改善路由和持续学习表现，在此基础上加入 PASs-RS 又能进一步减少遗忘。

此外，作者测试了不同 Expert 数量和正则化强度，发现适当的稳定化约束能够改善遗忘，但约束过强会损害新任务学习能力，体现了持续学习中的稳定性与可塑性权衡。