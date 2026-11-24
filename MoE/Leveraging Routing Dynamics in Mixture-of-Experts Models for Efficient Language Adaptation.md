文章链接：[http://arxiv.org/abs/2605.29714](http://arxiv.org/abs/2605.29714)

## 问题

MoE 模型通过 Router 动态选择 Top-k Expert，因此不同语言可以利用不同的 Expert 组合进行处理。理论上，可以只微调一部分 Expert，实现低资源语言适配，而不需要更新整个模型。

但这里存在两个问题：

1. **Expert 选择困难：** 多语言 MoE 并不是严格的一种语言对应一组 Expert。不同语言可能共享大量 Expert，而且不同层的语言专属性也不同。如果随机选择 Expert，或者直接选择某种语言激活最多的 Expert，未必能够有效适配目标语言。
2. **语言专属能力与跨语言共享能力的权衡：** 只更新语言专属 Expert，可能无法充分适配跨语言共享的知识和表示；如果更新大量甚至全部 Expert，又会增加计算成本，并可能导致灾难性遗忘。

因此，核心问题是：如何利用 MoE 已有的路由结构，识别语言专属 Expert 和跨语言共享 Expert，并通过选择性微调实现高效的低资源语言适配？

## 总结

研究多语言持续预训练过程中 MoE 的路由动态，发现语言专属路由主要出现在最后几层，而早期和中间层主要表现为跨语言共享。不同语言之间的 Token 词表重叠程度，是解释其路由相似性的重要因素。

基于这些发现，提出 SSFT（Selective and Shared Expert Finetuning），利用高资源语言作为 Anchor，识别与目标低资源语言相关的语言专属 Expert，同时加入跨语言共享 Expert，只微调最后两层中的这些 Expert 和对应 Router。

SSFT 在 MultiBLiMP 上达到 83.6% 的平均准确率，使用约 138M 可训练参数，占模型总参数量不到 2%，取得了接近更新 800M 参数的 AEFT 的效果。相比完整微调，GPU-hours 减少约 10 倍，FLOPs 减少约 100 倍。

## 思路

作者首先研究多语言持续预训练对 MoE 路由结构的影响。

1. **分析路由动态：** 从英语主导的 OLMoE-Base 出发，使用七种高资源语言构建 35B Token 的均衡多语言持续预训练语料，得到 OLMoE-M7。通过 Router Entropy 和 JSD 分析不同语言在不同层的 Expert 使用情况。

2. **发现语言专属路由的位置：** 多语言持续预训练使早期和中间层的路由更加分散，不同语言仍然共享大量 Expert；语言之间的路由差异主要在最后几层出现。因此，语言专属 Expert 并不是贯穿整个模型，而是主要集中在最后几层。

3. **找到影响路由相似性的因素：** 发现不同语言之间的 Token 词表重叠程度与路由相似性具有较强关联，词表重叠越高，JSD 通常越低。例如 Hindi–Marathi 的词表重叠超过 90%，其路由模式非常相似。这为使用高资源语言辅助低资源语言适配提供了依据。

4. **SEFT：选择语言专属 Expert。** 作者提出 Activation Gap，计算每个 Expert 对激活频率最高的两种语言之间的差值。差值越大，说明该 Expert 越偏向某一种语言。对于目标低资源语言，利用对应高资源 Anchor 的路由情况，在最后两层选择 Activation Gap 超过 1% 的语言专属 Expert，通常每层选择 5～7 个，然后使用低资源语言数据进行微调。

5. **SSFT：加入跨语言共享 Expert。** 作者发现仅微调语言专属 Expert 仍然不够，因此在 SEFT 的基础上，每层再加入 5 个跨语言共享 Expert。这些 Expert 根据其在七种高资源语言上的平均激活频率选择，取平均激活最高的前五个。最终同时微调语言专属 Expert 和共享 Expert。

6. **参数高效适配。** 只更新最后两层中选出的 Expert 及对应 Router，其余 Expert、Attention 和 Embedding 全部冻结。每种目标低资源语言使用约 300M Token 进行适配。

最终，SSFT 证明了同时更新语言专属 Expert 和共享 Expert，比只更新语言专属 Expert 更有效，而且不需要更新整个模型。

## 使用方法和技巧

### Router Entropy：衡量单种语言的路由集中程度

\[
H_k(\ell)=-\sum_e q^{(k)}(\ell)[e]\log q^{(k)}(\ell)[e]
\]

Entropy 越低，说明该语言的路由概率越集中在少量 Expert 上；越高则说明路由越分散。

### Jensen-Shannon Divergence：衡量不同语言的路由差异

\[
JSD_k(\ell_i,\ell_j)=
JSD(q^{(k)}(\ell_i),q^{(k)}(\ell_j))
\]

JSD 越低，说明两种语言越倾向于共享相同的 Expert；越高则说明两种语言的路由模式越不同。

### Activation Gap：识别语言专属 Expert

首先统计每种语言的 Token 被路由到某个 Expert 的频率：

\[
f(e,\ell)=
\frac{\text{被选中该 Expert 的 Token 数}}
{\text{该语言的总 Token 数}}
\]

然后找到该 Expert 激活频率最高的两种语言：

\[
G(e)=f(e,\ell_1)-f(e,\ell_2)
\]

其中，\(\ell_1\) 和 \(\ell_2\) 分别是激活频率最高和第二高的语言。

Gap 越大，说明 Expert 越具有语言专属性。

论文采用 1% 作为筛选阈值，选择与高资源 Anchor 对应的语言专属 Expert。

### Shared Expert Selection：识别跨语言共享 Expert

计算每个 Expert 在七种高资源语言上的平均激活频率，使用每种语言 5000 个 held-out 验证样本进行统计。

选择平均激活频率最高的前 \(k\) 个 Expert。

论文主实验中，每个最后层加入 5 个共享 Expert，并通过改变 \(k\) 的大小进行消融实验。

### SEFT：选择性微调语言专属 Expert

只微调目标语言相关的专属 Expert，通常每个最后层选择 5～7 个，使用 Activation Gap 进行筛选。

### SSFT：联合微调语言专属 Expert 和共享 Expert

在 SEFT 的基础上，每个最后层额外加入 5 个共享 Expert。

\[
\text{SSFT}=
\text{Language-specific Experts}
+
\text{Shared Experts}
\]

实验表明，SSFT 的提升不仅来自参数数量增加，因为它优于更新更多语言专属 Expert 的 SEFT-Top20。

### 对照实验：验证 Expert 选择是否有效

- Random-SEFT：随机选择与 SEFT 数量相同的 Expert，验证路由选择是否有效。
- SEFT-Top20：增加语言专属 Expert 数量，排除单纯增加参数量带来的影响。
- AEFT：微调最后两层的全部 Expert 和 Router。
- Full-FT：微调整个模型，作为性能上界。

### 实验指标与结果

使用 MultiBLiMP 和 Belebele 评估低资源语言适配效果。

MultiBLiMP 平均准确率：

| 方法 | 准确率 | 可训练参数 |
|---|---:|---:|
| Random-SEFT | 73.8% | 75M |
| SEFT | 78.7% | 75M |
| SEFT-Top20 | 77.7% | 251M |
| SSFT | 83.6% | 138M |
| AEFT | 84.6% | 800M |
| Full-FT | 90.5% | 7B |

SSFT 在只更新不到 2% 模型参数的情况下，取得了接近 AEFT 的效果，同时显著降低计算成本。