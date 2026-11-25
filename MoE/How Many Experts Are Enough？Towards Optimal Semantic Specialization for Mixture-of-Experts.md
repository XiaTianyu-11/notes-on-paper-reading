文章链接：https://ojs.aaai.org/index.php/AAAI/article/view/39665

## 问题

MoE 通过 Router 将不同 token 分配给不同 Expert，使模型能够利用多个参数子网络学习不同的语义和功能。但 Expert 数量的设置存在两个问题：
1. Expert 数量不足： 多种不同语义的 token 被迫共享同一个 Expert，导致 Expert 需要同时学习多种功能，难以形成明确的专业化分工，也可能增加不同知识之间的参数干扰。
2. Expert 数量过多： 部分 Expert 可能学习相似的功能，产生功能冗余，增加模型参数量和资源开销。
现有 DynMoE 根据 token 是否被现有 Expert 覆盖来决定是否增加 Expert，但 token 能够被现有 Expert 接收，并不意味着该 Expert 已经充分掌握了对应的语义。

从 MoE 记忆机制的角度看，这实际上涉及一个重要问题：模型应该如何将不同语义的知识分配到不同 Expert 中，避免多个语义长期竞争同一组参数表示空间，同时保证这些知识能够通过 Router 被有效访问？

## 总结

提出 MASS（Mixture-of-Experts for Adaptive Semantic Specialization），一种根据语义分化需求自动调整 Expert 数量的 MoE 框架。

MASS 通过监测 Expert 的梯度变化和梯度方向，判断现有 Expert 是否出现语义表示能力不足。当 Expert 的梯度持续增大，并且梯度方向与当前权重方向近似正交时，复制该 Expert，并通过梯度分解和路由正则化促进新旧 Expert 形成不同的功能。

此外，采用 Top-p 动态路由，根据 token 的路由置信度决定激活的 Expert 数量，使不同 token 能够使用不同规模的专家子网络。

从记忆机制角度看，MASS 提供了一种动态扩展 MoE 参数记忆容量、促进不同语义知识分散存储的思路：当现有 Expert 难以继续承载新的语义功能时，通过复制 Expert 创建新的参数空间，并让新旧 Expert 逐渐形成不同的知识分工。

## 思路

作者首先发现，传统 MoE 的 Expert 数量通常需要人工指定，而 DynMoE 虽然能够动态扩容，但只关注 token 是否被现有 Expert 覆盖，没有考虑 Expert 是否已经达到语义表示能力的边界。

因此，作者提出通过梯度行为判断 Expert 是否需要进一步分化。

具体分为两个阶段：

1. Adaptive Expert Expansion：自适应扩容

作者通过两个条件共同判断是否需要增加 Expert：
- Gradient Surge Detection： 使用概率变化点检测（CPD）监测每个 Expert 的梯度 L2 范数。如果梯度范数持续增大，并通过右尾显著性检验，说明当前 Expert 承受了较大的参数更新压力。
- Semantic Alignment Test： 计算当前梯度矩阵与 Expert 第一层线性投影权重矩阵的余弦相似度。如果二者近似正交，说明当前梯度更新方向与 Expert 原有的权重表示方向存在明显偏移。
只有两个条件同时满足，才认为当前 Expert 可能需要承载新的语义功能，并触发扩容。

扩容后，作者不是随机初始化一个新 Expert，而是复制原有 Expert，并采用梯度分解：
- 原 Expert：只接受与原有权重方向对齐的梯度，尽可能保留原来的功能。
- 新 Expert：接受完整梯度，使其能够探索新的参数更新方向。
同时复制对应的 Router 权重，并通过冗余正则化促进新旧 Expert 的路由行为逐渐分化，避免两个 Expert 始终处理相似的 token。

为了避免无效扩容，作者进一步使用 NLL 比较新增 Expert 的实际贡献。如果屏蔽某个新 Expert 后模型损失没有增加，就删除该 Expert；如果连续多次扩容都没有带来收益，则停止扩容。

2. Adaptive Top-p Routing：动态路由

传统 Top-k 为每个 token 固定激活数量为 \(k\) 的 Expert，但不同 token 的路由置信度不同。

作者采用 Top-p 策略，按照 Router 概率从高到低排序，选择累计概率质量首次达到阈值 \(p\) 的最小 Expert 集合。

因此：
- 路由概率高度集中：少量 Expert 就能达到阈值。
- 路由概率比较分散：需要更多 Expert 共同处理。
这样可以根据 token 的路由不确定性动态调整计算资源。

整个 MASS 在训练前 10% 的 steps 中执行动态扩容，剩余 90% 固定 Expert 数量进行常规训练，但 Top-p 路由在整个训练过程中持续使用。

## 使用方法和技巧

### Gradient Surge Detection：基于梯度增长检测 Expert 容量压力

对每个 Expert 计算任务损失关于其参数的梯度 L2 范数：
\[
g_t^{(k)}=\|\nabla_{\theta_k}L_t\|_2
\]

使用滑动窗口计算近期梯度均值和标准差，对梯度进行标准化：

\[
z_t^{(k)}
=
\frac{g_t^{(k)}-\mu_t^{(k)}}{\sigma_t^{(k)}}
\]

累计窗口内的标准化梯度：

\[
s_t^{(k)}
=
\sum_{i=t-\omega+1}^{t}z_i^{(k)}
\]

再计算标准化统计量和右尾 p-value：

\[
\widetilde{s}_t^{(k)}
=
\frac{s_t^{(k)}}{\sqrt{\omega}}
\]

\[
p_t^{(k)}
=
1-\Phi(\widetilde{s}_t^{(k)})
\]

当：
\[
p_t^{(k)}\leq\alpha
\]

认为该 Expert 出现了显著的梯度增长。

### Semantic Alignment Test：检测 Expert 的潜在语义偏移

计算当前梯度矩阵与 Expert 第一层线性投影权重矩阵的余弦相似度：

\[
\cos(\nabla^{(k)},W_e^{(k)})
=
\frac{
\langle\nabla^{(k)},W_e^{(k)}\rangle
}{
\|\nabla^{(k)}\|_2\|W_e^{(k)}\|_2
}
\]

当：
\[
|\cos(\nabla^{(k)},W_e^{(k)})|<\delta
\]

认为当前梯度方向与原有权重方向近似正交，存在潜在语义偏移。

本文设置：
\[
\delta=0.001
\]

该条件必须与梯度增长检测同时满足，才会触发扩容。

### Gradient Decomposition：通过梯度分解促进 Expert 专业化

将当前梯度分解为沿权重方向的分量和正交分量：

\[
G_{\parallel}
=
\frac{\langle G,W\rangle_F}
{\|W\|_F^2}W
\]

\[
G_{\perp}=G-G_{\parallel}
\]

其中 \(G\) 是当前梯度，\(W\) 是原 Expert 的权重矩阵。
- 原 Expert 只接受 \(G_{\parallel}\)，保留原有的参数表示方向。
- 新 Expert 接受完整梯度 \(G_{\parallel}+G_{\perp}\)，从而有机会学习不同的语义功能。
对应的 Router 权重也进行复制和梯度分配，使新旧 Expert 不仅在参数功能上分化，也能逐渐形成不同的 token 路由行为。

### Redundancy Regularization：避免新旧 Expert 功能冗余

对于复制产生的 Expert 对，计算对应 Router 权重向量的余弦相似度，并构造正则化损失：

\[
L_{\mathrm{red}}
=
\frac{1}{|P|}
\sum_{(i,j)\in P}
\cos(w_i,w_j)^2
\]

通过惩罚高度相似的 Router 权重，促使新旧 Expert 对不同 token 产生不同的路由响应，进一步促进语义专业化。

该正则项只在扩容阶段使用。

### NLL-based Expansion Stopping：根据实际贡献停止扩容

通过屏蔽新增 Expert 前后的 NLL 差值判断其是否有实际贡献：
\[
\Delta L=L_{\mathrm{mask}}-L_{\mathrm{orig}}
\]

- \(\Delta L>0\)：屏蔽后损失增大，说明该 Expert 有贡献，予以保留。
- \(\Delta L\leq0\)：屏蔽后损失没有增加，删除该 Expert，并减少 patience counter。
连续多次扩容没有收益，或者 Expert 数量达到上限时，停止扩容。

### Top-p Routing：根据路由置信度动态调整记忆访问范围

将 Router 输出的概率排序：
\[
r_{(1)}\geq r_{(2)}\geq\cdots\geq r_{(K)}
\]

选择满足以下条件的最小 \(k^*\)：
\[
\sum_{i=1}^{k^*}r_{(i)}\geq p
\]

使每个 token 根据自身的路由分布决定需要访问多少个 Expert。

从 MoE 记忆机制的角度，可以将其理解为一种动态知识访问策略：当 token 的语义与某个 Expert 的功能匹配度较高时，访问少量 Expert 即可；当路由分布比较分散时，则需要访问更多 Expert，以利用不同 Expert 的互补表示能力。

## 与 MoE 记忆机制的联系：

本文值得借鉴的地方，是将 Expert 数量与语义知识的分布联系起来。可以把 Expert 理解为分布在不同参数子网络中的知识表示单元，把 Router 理解为知识访问路径。MASS 通过动态扩容和梯度分解，为新的语义功能创建独立的参数空间，从而有机会减少不同语义知识对同一 Expert 的竞争。

这对研究 MoE 的记忆容量、知识干扰、知识分工以及知识访问机制都有启发。不过，本文并没有直接研究事实记忆、知识提取或遗忘，也没有通过记忆任务验证扩容是否提高了模型的事实记忆能力；上述记忆机制联系属于基于其方法的研究启发，而不是论文已经验证的实验结论。