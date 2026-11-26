文章链接：https://aclanthology.org/2025.findings-emnlp.991/

## 问题

以往关于 MoE Expert 专门化的研究，主要发现不同 Expert 会处理不同类型的 token，例如特殊字符、助动词和特定词汇等，但对于 Expert 是否能够根据更高层次的语义信息进行分工，尚缺乏明确证据。

这产生了两个问题：
1. 词面特征与语义特征难以区分：同一个词在不同上下文中可能具有不同含义，而不同的词也可能表达相同含义。仅观察 Expert 经常处理哪些 token，无法判断 Router 究竟是在根据词面形式还是实际语义进行路由。
2. 语义专门化是否随模型规模和层深变化：过去研究的 MoE 模型通常只有 8～32 个 Expert，可能不足以形成细粒度的语义分工。现代大规模 MoE 拥有更多 Experts，是否因此出现更明显的语义路由，以及这种现象集中在哪些层，仍需要验证。
此外，即使观察到语义相似的 token 被分配给相同 Expert，也还需要进一步判断这种现象是否延伸到更高层次的推理模式，而不只是词汇语义。

## 总结

提出 Semantic Routing（语义路由）的系统性实证研究，通过两个互补的受控实验，证明现代大规模 MoE 模型的 Router 不仅根据 token 的词面特征进行路由，还会受到上下文语义的影响：相同词汇在不同含义下会产生较低的 Expert 重合度，而不同词汇在语义相近时会产生较高的 Expert 重合度。

进一步发现，语义路由在模型中间层最明显，并且通常随着模型规模增加而增强。最后，作者利用 SAE 分析 DeepSeek-R1 的 CoT，发现具有相似推理功能的 token 也倾向于激活相似的 Experts，说明 Expert 专门化可能从词汇语义延伸到更高层次的推理模式。

## 思路

作者首先关注到，过去的 MoE 研究主要发现 Expert 对特殊字符、助动词等低层次 token 特征具有专门化，但没有发现明确的高层语义专门化。

作者认为，过去使用的 MoE 模型通常只有 8～32 个 Experts，可能不足以表达细粒度的语义分工。因此，选择 DeepSeek、Mixtral 和 Llama-4 三个模型家族，共六个不同规模的模型，研究大规模 MoE 是否具有 Semantic Routing。

具体设计了两个互补的实验：
- WiC（固定词汇，改变语义）：使用同一个多义词构造不同句子，分别保持或改变目标词的含义，比较目标词在不同句子中的 Expert 重合程度。如果语义相同的情况下重合度更高，说明 Router 能够根据上下文区分同一个词的不同含义。
- SWORDS（固定语境，改变词汇）：保持句子上下文不变，将目标词替换成语义相近或不同的词，比较原始词与替换词的 Expert 重合程度。如果不同词汇在语义相近时具有更高的重合度，说明路由并不只是依赖 token 的词面身份。
为了排除不同模型的 Expert 总数和激活数量对重合度的影响，作者进一步提出基于 Cohen's κ 的 Normalized Overlap Metric，对随机情况下的 Expert 重合进行校正。

实验发现，六个模型均表现出显著的语义路由现象，并且中间层的语义路由差异更加明显，整体上也呈现随模型规模增大而增强的趋势。

在此基础上，作者进一步研究语义路由是否能够延伸到更高层次的推理模式。

为此，使用 DeepSeek-R1 在 DiscoveryWorld 的 Reactor Lab 环境中生成 CoT，通过 SAE 提取模型内部的潜在特征，将具有共同特征的 token 归纳为相似的推理模式，再观察这些 token 对应的 Expert 分配。

例如，SAE 识别出与不确定性有关的推理特征，其对应 token 包括 Wait、bet、probably、attempt 等。这些词虽然词面不同，却具有较高的 Expert 重合程度。

由此，作者将 Expert 专门化从词汇语义进一步扩展到推理策略层面。

## 使用方法和技巧

### WiC：固定词汇，改变语义

使用 Words-in-Context 数据集，构造两类句子对：
- Same Sense：相同目标词在两个句子中具有相同含义。
- Different Sense：相同目标词在两个句子中具有不同含义。
分别记录目标词在每一层被 Router 选择的 Expert 集合，比较两种情况下的 Expert 重合程度。

核心是排除 token 词面不同带来的影响，观察上下文语义变化是否会改变路由。

### SWORDS：固定语境，改变词汇

使用 SWORDS 词汇替换数据集，构造三元组：
- Original：原始目标词。
- Equivalent：语义相近的替换词。
- Different：语义不同的替换词。
计算两组 Expert 重合度：

\[
\begin{aligned}
S_1&=Overlap(Original,Equivalent)\\
S_2&=Overlap(Original,Different)
\end{aligned}
\]

预期：

\[
S_1>S_2
\]

使用 Paired t-test 对配对样本进行统计检验，验证语义相似的不同词汇是否具有更高的路由相似性。

### Normalized Overlap Metric：归一化 Expert 重合指标

用于衡量两个输入的 Expert 路由相似程度，并消除随机选择带来的重合。

定义：
- \(N\)：模型拥有的 Expert 总数。
- \(k\)：每个输入激活的 Expert 数量。
- \(o\)：两个输入实际重合的 Expert 数量。
假设两个输入分别从 \(N\) 个 Expert 中随机选择 \(k\) 个，则期望重合数量为：

\[
E[o]=\frac{k^2}{N}
\]

实际重合率：

\[
P_o=\frac{o}{k}
\]

随机情况下的期望重合率：

\[
P_e=\frac{k}{N}
\]

最终得到：

\[
\boxed{
Score=\frac{P_o-P_e}{1-P_e}
=\frac{o-E[o]}{k-E[o]}
}
\]

其中：
- Score = 1：两个输入选择完全相同的 Experts。
- Score = 0：重合程度与随机选择的期望一致。
- Score 小于 0：重合程度低于随机期望。
该指标类似 Cohen's κ，可以用于比较不同 Expert 数量和激活配置的 MoE 模型。

### Layer-wise Analysis：分析不同层的语义路由强度

对模型每一层分别计算 Same 和 Different 两组的归一化 Expert 重合度：

\[
\Delta Overlap_l=
Score_{\text{Same},l}
-
Score_{\text{Different},l}
\]

通过绘制不同层的曲线，观察语义路由在模型深度上的分布。

实验发现，中间层通常具有更大的 \(\Delta Overlap\)，说明语义对 Expert 选择的影响在中间层更加明显。

### Scale Analysis：研究模型规模与语义路由的关系

计算每个模型在 WiC 和 SWORDS 上的平均语义路由差异：

\[
\Delta Overlap=
Score_{\text{Same}}-Score_{\text{Different}}
\]

以模型总参数量为横轴，以语义路由差异为纵轴，比较不同规模 MoE 模型的语义路由强度。

实验发现，语义路由整体上随模型规模增加而增强，但 Llama-4-Scout 是一个比较明显的例外，说明模型规模并不是唯一影响因素。

### SAE：提取高层语义与推理模式

使用 Sparse Autoencoder 分析 DeepSeek-R1 在 CoT 中的内部激活，寻找具有共同潜在特征的 token。

给定 Router 之前的隐藏激活：

\[
x\in\mathbb R^d
\]

Encoder 将其映射到稀疏潜在空间：

\[
z=\operatorname{ReLU}(W_{\mathrm{enc}}x+b_{\mathrm{enc}})
\]

Decoder 重构原始激活：

\[
\hat{x}=W_{\mathrm{dec}}z
\]

训练目标：

\[
L=\|x-\hat{x}\|_2^2+\lambda\|z\|_1
\]

其中，重构误差保证信息保留，L1 正则鼓励潜在特征稀疏激活。

具体实验设置：
- 环境：DiscoveryWorld 的 Reactor Lab。
- 收集 100 步 CoT。
- 收集约 200,000 个 token-activation pairs。
- SAE 训练层：Layer 7。
- SAE Width：28,672。
- Batch Size：4096。
- Learning Rate：\(5\times10^{-5}\)。
- Training Steps：30,000。
- 稀疏系数：\(\lambda=5\)。
训练完成后，通过 SAE feature 找到具有共同潜在特征的 token，再统计这些 token 对应的高频 Experts。

通过这种方式，将原本难以直接观察的高层推理模式与 MoE 的 Expert 路由行为联系起来。

### 统计检验：验证语义路由不是随机现象
- WiC：比较 Same Sense 和 Different Sense 的 Expert 重合度，六个模型平均跨层的差异均达到 \(p<0.001\)。
- SWORDS：使用 Paired t-test，六个模型均达到 \(p<0.0001\)。
这两组实验共同支持语义相似性与 Expert Routing 相似性之间存在显著关联。

### 研究启发：从词汇专门化扩展到推理模式专门化

传统 MoE 专门化研究主要关注不同 Expert 对领域、语言、主题和 token 类型的分工。

本文进一步提示，可以从更高层次研究 Expert 的功能分工：
- 不仅研究 Expert 处理什么词。
- 还可以研究 Expert 处理什么语义。
- 甚至可以研究 Expert 参与什么推理过程。
这为后续从 Expert 角度研究知识存储、推理能力分工、模型可解释性以及 MoE 的定向编辑和遗忘提供了新的切入点。
