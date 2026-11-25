文章链接：https://proceedings.iclr.cc/paper_files/paper/2026/hash/c01b90b643d0e877cff0f34cbe27410f-Abstract-Conference.html

## 问题

现有 MoE 模型压缩方法通常根据 Expert 的激活频率、Router Score、重构误差等指标评估专家重要性，进而剪枝、合并或量化不重要的 Expert。但这些指标主要是经验性的，没有充分揭示 MoE 内部是否存在极少数具有特殊功能、对模型运行机制至关重要的 Expert。

作者通过研究 MoE 模型中的 Massive Activations（MA），发现了一个此前没有被系统研究的特殊专家子集：Super Experts（SEs，超级专家）。

这产生了三个问题：
1. 超级专家如何产生？ 为什么极少数 Expert 的输出会出现异常巨大的激活，并通过残差连接传递到后续层，形成 Massive Activations？
2. 超级专家为什么如此重要？ 为什么 Qwen3-30B-A3B 中仅剪掉 6144 个专家中的 3 个，就会导致模型性能严重下降，甚至出现重复生成？
3. 超级专家通过什么机制影响整个模型？ 为什么剪掉少数早期层的 Expert，不仅会影响局部 FFN 计算，还会导致后续大量 Attention Head 的注意力分布发生变化？
## 总结

提出并系统研究了 MoE 中一种极其稀少但具有关键作用的专家：Super Experts（SEs）。

作者发现，SE 的 Down Projection 会产生极端激活异常值，这些异常值通过残差连接进入后续层的隐藏状态，形成 Massive Activations（MAs），最终参与 Transformer 中 Attention Sink（AS）的形成。

SE 具有三个主要特征：
- 数量极少： 在作者研究的 MoE 模型中，SE 占全部专家的比例不到 0.5%。
- 分布稳定： SE 的位置由模型本身决定，在不同数据域以及后训练前后基本保持不变。
- 作用关键： 剪掉 SE 会破坏模型中的系统性异常值机制，导致 Attention Sink 消失，进而造成模型整体性能严重退化，尤其是数学推理和代码任务。
例如，Qwen3-30B-A3B 只有 3 个 SE，占全部 6144 个专家的约 0.05%。剪掉这 3 个专家后，WikiText-2 的 PPL 从 8.70 上升至 59.86，推理模式下的平均性能从 69.37 降至 4.02。

作者最终揭示了 MoE 中一条重要的机制链：
\[
\boxed{
SE\rightarrow MA\rightarrow Attention\ Sink
}
\]

其中，SE 是系统性异常值的主要来源，MA 是中间传递机制，Attention Sink 则是这种机制在 Attention 层中的表现。Su 等 - 2026 - Unveiling Super Experts in Mixture-of-Experts Large Language Models.pdf


## 思路

作者的研究思路是从 Massive Activations 的形成机制入手，逐步发现超级专家，再通过剪枝实验验证其重要性，最后解释它们如何影响整个 Transformer 的运行。

1. 从 Massive Activations 入手，发现超级专家
作者首先观察到，MoE 模型的隐藏状态中存在少数异常巨大的激活值，即 Massive Activations。
这些激活值可能比普通激活大数万甚至十万倍。
作者进一步检查各层 Expert 的 Down Projection 输出，发现异常激活并不是所有专家共同产生的，而是集中在极少数特定专家中。
例如，Qwen3-30B-A3B 中发现：
- Layer 1 Expert 68
- Layer 2 Expert 92
- Layer 3 Expert 82
这三个专家的 Down Projection 都会产生极端输出。
这些输出通过 MoE 层后的残差连接进入隐藏状态，并在后续层继续传播、放大，逐渐形成稳定的 Massive Activations。

2. 研究超级专家的分布规律
作者进一步发现，SE 并不是普通意义上的语义专门化专家，而是模型内部一种相对固定的结构性特征。
具体表现为：
- 不同 MoE 模型中的 SE 位置不同，即 Model-specific。
- 更换英文、中文、数学、代码等输入数据，SE 的分布基本不变，即 Data-agnostic。
- Base Model 与经过 Instruct、Thinking 等后训练的模型，SE 的分布基本一致。
这说明 SE 并不是因为某个领域的数据输入较多，才形成的领域专门化专家，而更像是模型在训练过程中形成的特殊计算结构。

3. 通过动态剪枝，验证超级专家的重要性
作者为了验证 SE 的实际作用，采用推理阶段动态剪枝：当 Router 选中 SE 时，跳过该专家的计算。
同时设置两个对照组：
- Random Pruning：在整个模型中随机剪掉相同数量的专家。
- Layer-Controlled Random Pruning：只在 SE 所在层随机剪掉相同数量的普通专家。
第二种对照尤其重要，因为它排除了早期层本身比较重要这一因素。
实验发现，剪掉 SE 会导致模型在常识、问答、数学和代码任务上的性能全面下降，而两个随机剪枝对照组几乎没有影响。
对于 DeepSeek-R1 和 Qwen3 Thinking Mode，剪掉 SE 后，AIME、Math-500 等任务的 Pass@1 几乎降至零，模型还会出现持续重复生成的现象。
这说明 SE 并不是简单地存储某些领域知识，而是参与了更基础的模型运行机制。

4. 揭示 SE 影响 Attention Sink 的机制
这是文章最重要的机制分析。
作者将 SE 与 Transformer 中已有的 Systematic Outliers 现象联系起来，包括：
- Weight Outliers：异常权重，也称 Super Weights。
- Activation Outliers：异常激活，包括 Activation Spikes 和 Massive Activations。
- Attention Outliers：异常注意力分布，即 Attention Sinks。
作者发现，这些现象在 MoE 中存在一条联系：
首先，序列第一个 Token 通常也是 Attention Sink Token。Router 会对它分配较高的 SE 路由分数，使其强烈激活对应的超级专家。
随后，SE 内部 Down Projection 的异常权重产生极端输出，通过残差连接进入隐藏状态，形成 MA。
这些 MA 继续传递到后续层，使对应 Token 在后续 Attention 中吸引大量注意力，逐渐形成稳定的 Attention Sink。
需要注意，Transformer 每个 Block 都是先执行 Attention，再执行 MoE。因此，这是一种跨层逐渐建立和强化的机制：某层 SE 的输出影响后续层的 Attention，而不是在同一个 Block 内先通过 SE 产生激活、再反过来影响已经计算完的 Attention。

5. 剪掉 SE，验证 Attention Sink 是否消失
作者进一步直接剪掉 SE，观察 Attention Map 的变化。
结果发现，原模型中第一个 Token 能够吸引大量 Attention，但剪掉 SE 后，这种 Attention Sink 现象几乎完全消失。
作者还定义了 Attention Sink Decay Rate，量化剪枝前后流向 Sink Token 的注意力变化。
实验发现，Qwen3 剪掉 SE 后，后续各层的 Attention Sink 衰减率普遍达到约 90% 或更高。
这说明早期 SE 的影响并不局限于所在层，而是能够通过残差流影响后续大量 Attention 层。
最终，作者将模型性能崩溃归因于 SE 被剪掉后，原本依赖 SE 建立的系统性异常值机制遭到破坏，导致 Attention Sink 消失和注意力分布发生严重变化。


## 使用方法和技巧

### 1. Massive Activations：识别超级专家的基础

Massive Activations 指隐藏状态中少数数值异常巨大的激活。
作者发现，这些异常激活主要来源于少数 Expert 的 Down Projection 输出。
MoE Expert 的计算为：

\[
FFN(X)=(\sigma(XW_g)\odot XW_u)W_d
\]

其中：
- \(W_g\)：Gate Projection。
- \(W_u\)：Up Projection。
- \(W_d\)：Down Projection。
作者重点分析 \(W_d\) 的输出，寻找产生极端激活的专家。

### 2. Super Expert Profiling：超级专家识别方法

作者提出基于 Calibration Dataset 的自动识别方法，分为两个阶段。
第一阶段：确定 MA formation layers
遍历模型各层，检测隐藏状态中是否出现 MA pattern，将参与 MA 形成的层记录为：

\[
\mathcal L
\]

第二阶段：识别超级专家
对于每个候选层中的每个专家，计算其 Down Projection 输出的最大绝对值：

\[
a_{l,e}
=
\max_{x\in D}
\left\|
h_{l,e}(x)W_d
\right\|_\infty
\]

其中，\(h_{l,e}(x)\) 是该专家 Down Projection 的输入。
最终同时满足以下三个条件，才被认定为 SE：

\[
\boxed{
\begin{cases}
a_{l,e}>P_{99.5}\\
a_{l,e}>0.1a_{\max}\\
l\in\mathcal L
\end{cases}
}
\]

其中：
- \(P_{99.5}\)：所有候选专家输出幅值的 99.5% 分位数。
- \(a_{\max}\)：所有候选专家输出幅值的最大值。
- \(\mathcal L\)：参与 MA 形成的层集合。
第三个条件用于排除那些虽然 Down Projection 输出很大，但实际上不参与 MA 形成的普通 Outlier Experts。

### 3. Dynamic Expert Pruning：验证专家重要性

通过推理阶段动态跳过指定 Expert，比较剪枝前后的模型性能。
设置三种对照：
- 原始模型。
- 剪掉 SE。
- 随机剪掉相同数量的普通专家，包括全局随机剪枝和同层随机剪枝。
这种设计能够排除专家数量减少、所在层不同等因素的干扰，更准确地验证 SE 的特殊作用。

### 4. Attention Sink Decay Rate：衡量 Attention Sink 的破坏程度

作者提出该指标，计算剪掉 SE 后 Attention Sink 的衰减程度：

\[
D_{\text{sink}}
=
1-
\frac{1}{H}
\sum_{h=1}^{H}
\frac{
\sum_{i\in S}p'_{ti}
}{
\sum_{i\in S}p_{ti}
}
\]

其中：

- \(H\)：Attention Head 的数量。
- \(S\)：Sink Token 集合。
- \(p_{ti}\)：剪枝前 Query Token \(t\) 对 Sink Token \(i\) 的注意力分数。
- \(p'_{ti}\)：剪枝后的对应注意力分数。
该指标越高，说明剪掉 SE 后，原本分配给 Sink Token 的注意力越少。

### 5. Systematic Outliers：从权重、激活到注意力的关联分析

作者将 Transformer 中三种异常现象联系起来：

|层次|	现象|	含义|
|---|---|---|
|Weight|	Super Weights|	少数权重数值异常突出|
|Activation	|Massive Activations|	少数隐藏状态维度产生极端激活|
|Attention|	Attention Sinks|	大量 Query 将注意力集中到少数 Token|

其中，Super Weights 是 SE 内部 Down Projection 的异常权重来源之一。
作者进一步使用 SAE（Sparse Autoencoder）分析与 Super Weights 相关的特征，尝试理解这些异常权重所参与的模型功能，并发现与文本结束、终止行为相关的特征。这为剪枝后模型持续重复生成提供了进一步的解释。

### 6. QKV Suppression 与 QK Cosine Similarity：解释 MA 如何影响 Attention

作者进一步分析 Massive Activations 经过 RMSNorm 和 QKV Projection 后的变化。
主要观察到两个现象：
- QKV Suppression： MA 会使对应 Token 经过 RMSNorm 后的 Q、K、V 向量范数明显减小。
- High QK Cosine Similarity： 普通 Token 的 Query 与 Sink Token 的 Key 之间仍然具有较高的余弦相似度，使 Sink Token 能够获得较大的注意力分数。
这两个现象帮助解释了为什么异常激活经过归一化和 QKV 投影后，仍然能够与 Attention Sink 保持联系。

## 对 MoE 记忆机制的启发：

这篇文章最值得关注的地方，是它发现了专家重要性并不只是知识存储能力的差异。
传统 MoE 研究往往关注不同 Expert 如何分工存储不同领域的知识，而本文发现，极少数 Expert 可能承担着更加基础的模型运行功能，其影响可以跨越多个 Transformer Block。
这意味着，在研究 MoE 的知识归属、记忆定位和遗忘时，需要区分两种专家：
1. 知识型专家： 主要影响特定知识或语义信息的处理与存储。
2. 机制型专家： 通过特殊激活和残差传播，影响模型整体的计算与注意力机制。
如果将第二类专家误认为普通知识型专家并直接剪枝或修改，可能造成远超预期的模型能力退化。
因此，后续研究 MoE 的知识编辑、遗忘和成员推断时，可以进一步考察：目标知识所在的专家与承担系统性异常值机制的专家是否重合，以及修改前者是否会意外破坏后者所维持的模型运行机制。