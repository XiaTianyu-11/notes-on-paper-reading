文章链接：https://ojs.aaai.org/index.php/AAAI/article/view/40622

## 问题

现有关于大模型知识归因（Knowledge Attribution）的研究，大多是在预训练完成后分析模型，识别哪些神经元、Attention Head 或 FFN 负责特定知识。这种静态分析只能看到最终形成的知识结构，却无法解释这些结构在预训练过程中是如何逐渐形成和稳定下来的。

对于 MoE 模型，这个问题更加值得研究。MoE 通过 Router 动态选择 Top-k Expert，使不同 Token 可以调用不同的 FFN 参数。这种稀疏激活机制可能改变知识在模型内部的组织方式，但目前还不清楚：
1. 知识如何形成： MoE 是否会在预训练过程中逐渐形成一批长期负责知识处理的核心神经元？这些神经元是否比 Dense 模型更加稳定？
2. 知识如何分布： MoE 的稳定核心是否意味着知识集中在少数神经元中？还是说它能够形成稳定但具有冗余计算路径的分布式知识结构？
3. 架构如何影响稳定性： MoE 的稀疏路由是否不仅影响 Expert 内部的知识组织，还会影响 Attention 和整个 Transformer 的层级重要性分布？
因此，作者希望从训练过程出发，追踪 MoE 与 Dense 模型的神经元重要性变化，并进一步验证这种内部结构差异是否会影响模型的功能鲁棒性。

## 总结

提出 Gated-LPI（Gated Log-Probability Increase），一种面向 MoE 和 Dense Transformer 的神经元级知识归因方法，通过将 FFN 和 Attention 的输出分解为独立神经元的贡献，追踪不同训练 checkpoint 下各神经元对目标 Token 预测的影响。

作者使用 OLMoE-1B-7B（MoE）和 OLMo-7B（Dense）进行对照实验，发现 MoE 在预训练过程中呈现出三个特点：
1. Low-entropy Backbone（低熵骨干）： MoE 的重要性增长高度集中于少量神经元。FFN 中 Top-1% 神经元吸收了约 48.9% 的重要性增长，并且这些神经元在训练过程中持续得到强化。Dense 模型则没有形成同样稳定的核心神经元集合。
2. Early Consolidation（早期稳定）： MoE 在训练早期就形成了稳定的神经元重要性分布，并进一步形成稳定的层级重要性结构。Dense 模型的重要神经元和层级重要性分布则持续波动。
3. Functional Robustness（功能鲁棒性）： 屏蔽 MoE 最重要的 10 个 Attention Head，关系知识预测性能平均下降仅 9.44%；Dense 则下降 50.43%。屏蔽 Top-1% FFN 神经元时，MoE 下降 35.47%，Dense 下降 96.19%。这表明 MoE 的知识计算具有更强的功能冗余，不会过度依赖少数不可替代的组件。
最终，作者认为 MoE 的稀疏激活不仅提高了模型的计算效率，还可能形成一种有利于稳定学习和分布式知识存储的内部计算结构。


## 思路

作者的整体思路是：先解决 MoE 中如何进行神经元级知识归因的问题，再利用这种归因方法追踪预训练过程中的知识组织变化，最后通过人为屏蔽重要组件，验证这种内部结构是否具有实际的功能鲁棒性。

1. 将传统 LPI 扩展到 MoE，解决神经元归因问题。
传统 LPI 可以通过比较神经元输出加入前后的目标 Token 对数概率变化，衡量神经元的重要性。
但 MoE 中存在 Router 和 Top-k Expert 选择机制，一个 Token 的 FFN 输出是多个 Expert 输出的加权求和。因此，不能直接将 Dense FFN 的神经元归因方式照搬过来。
作者提出 Gated-LPI，将 Expert 内部神经元的贡献与 Router Gate 结合起来，并进一步把 Attention 中的 Value Projection 和 Output Projection 也分解为独立神经元，实现对 FFN 和 Attention 的统一归因。

2. 利用大量训练 checkpoint，追踪重要神经元如何形成。
作者不再只分析训练完成后的模型，而是收集预训练过程中不同阶段的模型 checkpoint，分别计算每个神经元的重要性。
通过两个指标研究：
- Top-1% Set Stability：重要神经元是否始终是同一批？
- Positive-Gain Concentration：训练新增的重要性是否集中到少数神经元？
实验发现 MoE 呈现出 Explore–Refine–Consolidate 三阶段过程：
- Explore：逐渐形成一批初步的重要神经元。
- Refine：筛选和调整早期形成的核心，部分神经元退出，部分新神经元进入。
- Consolidate：重要神经元集合逐渐稳定，新增的重要性持续集中到这批核心神经元。
Dense 则没有形成持久的核心神经元集合，重要性经常转移到新的神经元上。

3. 将分析尺度从神经元扩展到 Transformer Layer。
前面只分析单个神经元是否稳定，但作者还希望知道：这些重要神经元所在的层级，其整体重要性分布是否也会稳定下来？
于是统计不同 checkpoint 下每一层的平均重要性，利用 Pearson 相关系数和变异系数衡量层级重要性分布的稳定程度。
结果发现，MoE 的 FFN 和 Attention 层都很早形成稳定的层级重要性分布，而 Dense 模型在整个训练过程中仍存在明显波动。

4. 通过 Causal Intervention 验证内部结构是否具有功能鲁棒性。
仅仅发现 MoE 形成稳定的重要神经元，并不能说明它的知识存储更加鲁棒。
因为稳定的核心也可能意味着模型过度依赖少数神经元，一旦这些神经元被破坏，模型就会失去相应知识。
因此作者选择最重要的 Attention Head 和 FFN 神经元进行屏蔽，再通过 HIT@10 测量关系知识预测性能的变化。
结果发现，MoE 在屏蔽重要组件后仍然保留较多预测能力，而 Dense 模型的性能则大幅下降。
由此作者将 MoE 的稳定核心与分布式知识计算、功能鲁棒性联系起来。

## 使用方法和技巧

### Gated-LPI：面向 MoE 的神经元级知识归因

将 Expert 内部的 FFN 分解为：

\[
FFN_E(x)=
\sum_i
\sigma(w_{E,i}^{T}x)v_{E,i}
\]

其中：
- \(w_{E,i}\)：输入投影矩阵第 \(i\) 行的转置，即 Subkey。
- \(v_{E,i}\)：输出投影矩阵第 \(i\) 列，即 Subvalue。
- \(\sigma(w_{E,i}^{T}x)\)：神经元的标量激活。
考虑 Router Gate 后，一个 Expert 神经元的贡献为：

\[
output_v=
\sum_{E\in Top-k}
gate_E(x)\cdot m_E\cdot v_E
\]

Attention 也采用类似的分解，将 Value Projection 的一行与 Output Projection 的一列配对，得到 Attention Neuron。

最终统一使用 Log-Probability Increase 衡量神经元的重要性：

\[
I(v)=
\log p(w\mid x+output_v)
-
\log p(w\mid x)
\]

其中：
- \(I(v)>0\)：神经元促进目标 Token 的预测。
- \(I(v)<0\)：神经元抑制目标 Token 的预测。
- \(|I(v)|\) 越大：神经元对目标 Token 预测的影响越大。
### Top-1% Set Stability：重要神经元集合稳定性

每个训练 checkpoint 根据神经元的重要性绝对值进行排序，取 Top-1% 神经元集合 \(T_t\)，然后计算相邻 checkpoint 之间的 Jaccard 相似度：

\[
J_{\rm stab}=
\frac{1}{S-1}
\sum_{t=1}^{S-1}
\frac{|T_t\cap T_{t+1}|}
{|T_t\cup T_{t+1}|}
\]

其中：
- \(T_t\)：第 \(t\) 个 checkpoint 的 Top-1% 重要神经元。
- \(S\)：checkpoint 总数。
数值越高，说明不同训练阶段的重要神经元越一致，即核心神经元越稳定。

### Positive-Gain Concentration：重要性增长集中度

衡量训练过程中新增的重要性是否集中在少量神经元。

首先计算每个神经元的重要性增长：

\[
\Delta^+|I_{i,t}|=
\max(0,|I_{i,t}|-|I_{i,t-1}|)
\]

然后选取重要性增长最大的 Top-1% 神经元集合 \(T_t\)，计算：

\[
R_t=
\frac{
\sum_{i\in T_t}\Delta^+|I_{i,t}|
}{
\sum_{j\in\mathcal N}\Delta^+|I_{j,t}|
}
\]

其中 \(\mathcal N\) 是该层全部神经元。

数值越高，说明越多的重要性增长集中在少量神经元上。

注意，这里计算的是绝对重要性增长，而不是仅统计 \(I(v)>0\) 的神经元。例如，一个神经元的 LPI 从 -0.5 变成 -1.0，也会被计入正向增益。

### Layer-Distribution Consistency：层级重要性分布一致性

将每个 checkpoint 中各层的重要性组成向量：

\[
\mathbf I_t=
[I_{t,1},I_{t,2},\ldots,I_{t,L}]
\]

计算不同 checkpoint 之间的 Pearson 相关系数，再求平均：

\[
\rho_{\rm avg}
=
\frac{2}{S(S-1)}
\sum_{i<j}Corr(\mathbf I_i,\mathbf I_j)
\]

数值越高，说明不同训练阶段的层级重要性分布越相似，即哪些层相对重要逐渐固定下来。

### Cross-step Coefficient of Variation：层级重要性波动

对每一层计算其在所有 checkpoint 上的重要性均值和标准差：

\[
\mu_l=\frac1S\sum_t I_{t,l}
\]

\[
\sigma_l=
\sqrt{\frac1S\sum_t(I_{t,l}-\mu_l)^2}
\]

最后对所有层的变异系数求平均：

\[
\sigma_{\rm rel}
=
\frac1L\sum_l
\frac{\sigma_l}{|\mu_l|}
\]

数值越低，说明各层的重要性随训练时间的相对波动越小。

### HIT@10：关系知识预测能力

使用关系事实数据集测试模型对知识的预测能力。

数据集包含：
- 906 个 Subject-Object 样本；
- 12 种关系；
- 4 类知识：Linguistic、Commonsense、Factual、Bias。
通过 Cloze Prompt 输入 Subject，让模型预测 Object。

\[
HIT@10=
\frac1N\sum_{i=1}^{N}
\mathbb I[y_i\in Top10(\hat y_i)]
\]

如果正确 Object Token 进入模型预测概率最高的 Top-10，则该样本记为 1，否则记为 0。

### Causal Intervention：重要组件屏蔽实验

根据 Gated-LPI 的重要性排序，选择三种组件进行屏蔽：
- Top-1 Head：屏蔽最重要的一个 Attention Head。
- Top-10 Heads：屏蔽最重要的十个 Attention Heads。
- Top-1% FFN Neurons：屏蔽最重要的 1% FFN 神经元。
比较屏蔽前后的 HIT@10，计算性能下降。

通过这种方式，作者不仅识别重要神经元，还进一步验证模型是否过度依赖这些组件。

补充理解： 本文最值得借鉴的是把知识归因从训练完成后的静态分析扩展到预训练全过程，并将神经元重要性、重要性增长集中度、层级稳定性和功能干预结合起来。不过，文中所谓的知识存储是根据模型对关系事实预测的归因结果推断出来的，并不等同于直接证明某条事实被物理存储在某个神经元中。另外，OLMo 和 OLMoE 的训练 Token 数、语料和层数均不完全一致，架构差异与其他训练因素的影响尚不能完全分离。对于 MoE 记忆机制研究，本文提供的是一套分析知识组织与稳定性的工具和思路，而不是直接定位某条记忆对应的 Expert 或神经元。