文章链接：https://openreview.net/forum?id=BqyPLOkxFY

## 问题

现有 MoE 路由机制的可解释性研究主要关注 Expert 本身，例如不同 Expert 的领域专门化、Token 专门化、同层 Expert 共激活，以及 Expert 权重、Router 权重和输出之间的相关性。

但这些研究忽略了一个重要问题：当前层 Router 的决策不仅受到当前 Token 的影响，还会受到此前多个 Transformer Layer 中 Attention 和 MoE 输出的影响。

这是因为 Transformer 的 Residual Stream 会持续累积前面各层的输出，而 Router 根据当前隐藏状态计算各个 Expert 的 Assignment Score。因此，早期层的 Attention Head 或 Expert 可能持续影响后续很多层的路由决策。

这产生了三个值得研究的问题：
1. 跨层影响： 不同层的 Attention 和 MoE 输出分别对后续 Router 有多大影响？这种影响会随着网络深度增加而衰减吗？
2. 组件差异： Attention Head、MoE Layer、单个 Expert 对 Router 的影响是否均匀？是否存在少数具有持续影响力的组件？
3. 促进与抑制： 不同组件究竟是在促进某些 Expert 被选中，还是抑制它们被选中？这种影响是否具有长距离传播特征？
## 总结

提出一种递归式的 MoE 路由贡献分解框架（Recursive Decomposition Framework），将某一层 Router 对各个 Expert 的 Assignment Score 分解为 Token Embedding、前面各层 Attention 和 MoE 输出的贡献，并进一步分解到 Attention Head、Query-Key 交互以及具体 Expert。

作者在 OLMoE、DeepSeek-V2-Lite、Qwen3-30B-A3B 和 Mixtral-8x7B 四个模型上进行实验，主要发现：
1. MoE 输出通常具有最强、最持久的跨层路由影响。 相比之下，Token Embedding 和 Attention Layer 的影响更加局部，尤其是 Token Embedding 对路由的影响会随着网络加深而迅速下降。但在部分浅层和特殊位置，Token 与 Attention 仍然可能具有较强影响。
2. 促进与抑制具有不同的传播模式。 正向贡献通常更加局部，而负向贡献往往能够影响更远的后续层，对部分 Experts 产生持续抑制。
3. 存在跨层路由纠缠（Cross-layer Entanglement）。 某些 MoE Layer 对后续多层 Router 具有异常强的影响，在热力图中形成明显的条纹。例如 OLMoE 的 M1、M4。
4. 少数 Attention Heads 和 Experts 主导路由影响。 一些功能性 Attention Heads 具有较高的路由贡献方差；部分 Experts 能够持续影响很深的后续层，例如 OLMoE 的 M1E9、M4E14，以及 Qwen 的 M2E92。
5. Super Expert 不一定具有最强的跨层路由影响。 Expert 输出范数大，并不意味着它一定能显著改变后续 Router 的决策，真正重要的是其输出与后续 Router 权重之间的匹配关系，以及对不同 Experts 产生的分数差异。
总体而言，本文揭示了 MoE Routing 并不是一个纯粹的局部决策过程，而是由多个层、Head 和 Expert 之间的跨层交互共同形成的。

## 思路

作者首先从 Transformer 的 Residual Stream 入手，发现当前层 Router 的输入可以分解为初始 Token Embedding、此前各层 Attention 输出和 MoE 输出的累积。因此，当前层 Expert 的 Assignment Score 实际上包含了此前多个组件的共同贡献。

由此，作者希望将原本不可直接观察的跨层路由影响拆解成各个组件的独立贡献。

具体分为以下几个步骤：

- 第一步：解决 LayerNorm 阻碍分解的问题。
Router 的输入经过 RMSNorm，而直接对不同组件分别进行归一化再相加，并不等于对完整残差流进行归一化。
因此，作者采用共享归一化因子的近似方法：所有组件都使用完整 Router 输入对应的 RMS 作为归一化分母，使归一化后的组件贡献能够相加，近似重构实际 Router Input。

- 第二步：将 Router Assignment Score 分解到不同组件。
利用 Router Score 的点积形式：
\[
S(g^{\ell,n},m_{\mathrm{in}}^\ell)
=
g^{\ell,n}\cdot m_{\mathrm{in}}^\ell
\]
将当前层 Router 的分数分解为：

\[
S=
S_{\mathrm{Token}}
+
\sum_{c\leq\ell}S_{\mathrm{Attention}}^c
+
\sum_{c<\ell}S_{\mathrm{MoE}}^c
\]
从而分别计算初始 Token Embedding、此前 Attention Layer 和 MoE Layer 对目标 Expert 的贡献。

- 第三步：进一步分解到更细粒度的组件。
对于 Attention Layer，利用多头注意力的输出分解，将贡献进一步拆解为：
Attention Head、Query Token、Key Token。
对于 MoE Layer，利用 Top-K Experts 的加权输出，将贡献进一步分解到各个被激活的 Expert。
理论上还可以继续分解到 Expert 内部的 Neuron，但本文没有实际开展这一层面的实验。

- 第四步：通过贡献分数分析路由影响。
作者提出使用 Score Variance 衡量组件对不同 Experts 的区分能力，并利用正负贡献判断组件是在促进还是抑制特定 Experts。
随后分别在 Token、Layer、Attention Head 和 Expert 四个层面开展实验，观察跨层路由影响的分布规律。

- 第五步：验证高贡献组件是否真正影响 Top-K 选择。
仅有较大的 Score Variance 并不能直接证明 Top-K 一定发生变化。
因此，作者选择 OLMoE 中的 M1E9 作为案例，将它对后续 Experts 的贡献分数置零，观察原本 Top-K Experts 的排名变化，并提出 AARV 指标验证其对后续路由选择的实际影响。

## 使用方法和技巧

### Recursive Assignment Score Decomposition：递归式路由分数分解

利用残差流的累积结构，将当前层 Router 的 Assignment Score 分解为：

\[
\begin{aligned}
S=&S_{\mathrm{Token}}
+\sum_{c=1}^{\ell}S_{\mathrm{Attention}}^c\\
&+\sum_{c=1}^{\ell-1}S_{\mathrm{MoE}}^c
\end{aligned}
\]

再利用 Attention Head 和 MoE Expert 的输出结构继续分解，实现从 Layer 到 Head、Expert 的细粒度归因。

### Score Variance：衡量组件对路由的影响程度

计算某个组件对当前层所有 Experts 的贡献分数方差：

\[
Var(c)=\frac{1}{N}\sum_{n=1}^{N}(s_n^{(c)}-\mu_c)^2
\]

方差越大，说明该组件对不同 Experts 产生的分数差异越大，对路由排序的潜在影响越强。

如果一个组件对所有 Experts 都贡献相同分数，则方差为零，不会改变它们之间的相对排序。

### APS / ANS：衡量组件的促进与抑制倾向
- APS（Average Positive Score）：组件对 Experts 的平均正向贡献。
- ANS（Average Negative Score）：组件对 Experts 的平均负向贡献。

\[
APS=\frac{1}{N}\sum_n s_n^{(c)}\mathbf{1}_{s_n^{(c)}>0}
\]

\[
ANS=\frac{1}{N}\sum_n s_n^{(c)}\mathbf{1}_{s_n^{(c)}<0}
\]

正分表示促进对应 Expert 被选中，负分表示抑制对应 Expert 被选中。

将两者分开统计，可以避免正负贡献相互抵消。

### t-SNE：分析 Token 对 Expert 的评分模式

将每个 Token 对所有 Experts 的贡献分数组成向量，再使用 t-SNE 降维可视化。

通过按照 POS（词性）和 Token Position 着色，观察不同词性和位置的 Token 是否具有相似的路由贡献模式。

实验发现 Token 的词性会影响其评分分布，且首位 Token 具有特殊的评分模式。

### IOI + Path Patching：分析功能性 Attention Head 对路由的影响

使用 IOI（Indirect Object Identification）任务，借助 Path Patching 找到参与任务计算的功能性 Attention Heads。

然后计算不同 Head 对后续 Router 的 Score Variance，并与 Attention Map 进行比较。

实验发现：
- 功能性 Attention Heads 通常具有更高的 Score Variance。
- Score Variance Map 与 Attention Map 存在相似的分布模式。
说明 Attention Head 的任务功能与其对 MoE 路由的影响存在一定联系。

### AARV：衡量组件对 Top-K 排名的实际影响

提出 Average Absolute Ranking Variation of Top-K Experts：

\[
AARV=
\frac{1}{K}
\sum_{e\in TopK}
|rank_{\mathrm{orig}}(e)-rank_{\mathrm{pert}}(e)|
\]

通过将某个组件对各 Experts 的贡献分数置零，比较扰动前后原始 Top-K Experts 的排名变化。

AARV 越大，说明移除该组件的贡献后，原本被选中的 Experts 排名变化越明显。

作者利用该指标验证 OLMoE 中 M1E9 的跨层影响，发现它对 M5、M10、M15 的 Top-K 排名均有显著影响，并进一步发现 M1E9 与 M4E14 需要共同存在，才能对 M5 及后续层产生显著影响。

### Cross-layer Entanglement：利用跨层影响优化 MoE

本文还提出了两个潜在应用方向：
- Expert Prefetching： 利用早期 MoE Layer 对后续 Router 的持续影响，提前预测并预加载可能被激活的 Experts，减少 Expert Parallelism 中的通信和调度开销。
- Selective Attention Compression： 根据不同 Attention Layer 的路由贡献，优先压缩低影响力的 Attention Layer，保留对后续路由影响较大的组件。