文章链接：https://aclanthology.org/2025.acl-long.1093/

## 问题

现有的知识归因方法主要针对 Dense Transformer，通常通过分析 Attention Head、FFN 神经元对模型预测的贡献，研究模型内部的知识存储和处理机制。

但 MoE 与 Dense 模型存在一个重要区别：MoE 通过 Router 动态选择 Top-k Experts，不同 Token 可能经过不同的 Expert。同时，部分 MoE 还包含始终参与计算的 Shared Expert。

这产生了三个问题：
1. 知识归因困难： 传统 Dense 模型的归因方法无法充分描述 Router、Routed Experts 和 Shared Experts 之间的动态交互，因此难以定位 MoE 中不同模块对知识预测的贡献。
2. Expert 分工不明确： Shared Expert 通常被认为负责通用知识，Routed Expert 负责领域专门知识，但这种功能分工缺乏充分的实验证据。不同层的 Experts 又是如何协作完成知识处理的，也不清楚。
3. 知识专门化与鲁棒性的矛盾： 如果特定知识集中在少数 Experts 中，那么这些 Experts 失效时，模型可能无法正常完成对应任务。反过来，如果知识分散在大量 Experts 中，又可能降低 Expert 的专门化程度和知识处理效率。
因此，作者希望建立一套适用于 MoE 的知识归因方法，回答三个问题：知识贡献集中在哪些层？Shared 和 Routed Experts 如何分工？知识专门化会如何影响模型鲁棒性？

## 总结

提出一种面向 MoE 的跨层级知识归因框架（Cross-level Knowledge Attribution），通过分析 Attention Head、FFN 神经元和 Expert 神经元对正确 Token 预测的贡献，研究 MoE 内部的知识处理机制。

作者通过比较 Qwen 1.5-MoE、OLMoE、Mixtral-8x7B 与 Dense 模型，得到四个主要发现：
1. MoE 具有更明显的阶段化知识处理机制。 作者发现 MoE 存在 Mid-Activation、Late-Amplification 现象：中层通过 Router 筛选与当前任务相关的 Experts，后层通过 Shared 和 Routed Experts 协作进一步精炼知识。相比 Dense 模型较为分散的层级贡献，MoE 能够实现更高的平均单层 FFN attribution gain。
2. Shared Expert 与 Routed Expert 存在基础处理—专门精炼的协作关系（Basic-Refinement Collaboration）。 Shared Expert 作为 Generalist，负责基础语义处理、实体识别等通用任务；Routed Experts 作为 Specialists，负责领域知识和具体属性关联。例如 Shared Expert 识别 Canada 是一个国家，Routed Expert 进一步建立 Canada 与 Ottawa 的首都关系。
3. Expert Routing 受到 Attention Head 的语义信息影响。 Attention Head 提取上下文语义并影响 Router 的输入表示，使 Router 能够选择与当前任务相关的 Experts。作者通过相关性分析、Attention Head 抑制、Expert 强制激活和 Integrated Gradients 验证了这一机制。
4. Expert 专门化与模型深度、冗余程度共同影响鲁棒性。 Qwen 1.5-MoE 具有较明确的层级知识分工，并通过 Shared Expert 和其他计算路径提供一定冗余；OLMoE 的 Experts 需要承担更多不同任务，缺乏足够的层级冗余；Mixtral 则通过相对粗粒度、密集的路由分配，使知识贡献更加分散，对部分 Expert 故障具有较强容忍能力。
最终，作者认为 MoE 架构需要在知识专门化、跨层协作和功能冗余之间取得平衡，而不是单纯增加 Expert 数量或追求更高的专门化程度。

## 思路

作者按照「建立知识归因方法 → 发现跨层处理规律 → 分析 Expert 协作机制 → 定位知识并研究鲁棒性」的顺序展开研究。

1. 建立 MoE 的跨层级知识归因方法
首先建立统一的 MoE 计算框架，描述 Attention、Router、Shared Expert 和 Routed Expert 的计算关系。
对于第 \(l\) 层的 Token：
\[
u_i^l=h_i^{l-1}+A_i^l
\]
Router 根据 \(u_i^l\) 计算路由概率，选择 Top-k Experts，各 Expert 计算自己的 FFN 输出，最后经过门控加权后加入 residual stream。
在此基础上，作者扩展 Dense 模型的神经元归因方法，通过向隐藏状态中加入某个神经元的输出贡献，观察正确 Token 的对数概率变化：
\[
I(v)=
\log p(y_i\mid u+v)-\log p(y_i\mid u)
\]
对于 Expert 神经元，还需要考虑 Router 的门控权重：
\[
I(v_{E_j}^l)=
\log p(y_i\mid u_i^l+g_{i,j}^lv_{E_j}^l)
-\log p(y_i\mid u_i^l)
\]
这样就可以衡量不同神经元、模块和 Expert 对知识预测的贡献。

2. 比较 Dense 与 MoE 的跨层知识处理效率
作者将神经元的重要性进一步用于层级贡献分析，比较不同模型各层 FFN 的 attribution gain。
发现 MoE 存在：
\[
\boxed{\text{Mid-Activation}+\text{Late-Amplification}}
\]
以 Qwen 1.5-MoE 为例，将24层划分为三个阶段：
- Early Stage（Layer 1–13）：提取基础特征，为后续知识处理建立表示。
- Mid Stage（Layer 14–19）：Router 开始筛选与当前任务相关的 Experts，产生明显的知识归因贡献。
- Late Stage（Layer 20–24）：Shared Expert 和 Routed Expert 协作，进一步完成领域知识和属性关联。
相比之下，Dense 模型的 FFN gain 相对分散，逐层积累；浅层 MoE（如 OLMoE）虽然能够较早集中产生较大贡献，但由于层数有限，后续精炼空间不足。
作者由此认为，MoE 可以通过阶段化的知识处理提高单层贡献效率。

3. 通过消融实验分析 Shared 和 Routed Experts 的分工
作者分别对 Qwen 1.5-MoE、OLMoE 和 Mixtral 改变参与计算的 Expert 数量，观察 HIT@10 和 MRR 的变化。
以 Qwen 1.5-MoE 为例：
- 只使用 Shared Expert：HIT@10 仅为 0.03。
- 只使用 Top-1 Routed Expert：无法完成知识预测。
- 使用 Shared + Top-1 Routed Expert：HIT@10 达到 0.82。
- 使用 Shared + Top-4 Routed Experts：HIT@10 达到 0.85。
这说明两类 Experts 存在明显互补性，而且增加参与计算的 Experts 并不会一直带来显著收益。作者将这种现象称为 Effective Expert Threshold。
随后，作者进一步研究 Attention Head 和 Expert 之间的联系。
首先筛选知识任务中重要性较高的 Attention Heads，以及被频繁激活的 Routed Experts，计算两者的激活相关性。
接着通过三种实验验证：
- Attention Head Suppression：抑制关键 Head，观察 Router 的变化和模型性能。
- Expert Forcing：绕过正常路由，强制激活指定 Expert，观察能否恢复性能。
- Integrated Gradients：分析 Attention Head 对 Expert 激活的归因贡献。
由此提出 Semantic-Driven Routing：Attention Head 提供语义信息，Router 根据这些信息选择合适的 Experts。

4. 定位知识对应的 Experts，并研究 Expert 故障的影响
作者针对不同知识关系，计算相关神经元的 importance，选出 Top-100 重要神经元，然后统计它们分别属于哪些 Experts，找出包含这些神经元最多的 Top-5 Experts。
例如，对于 country_capital_city，作者定位到 Qwen 1.5-MoE 中一些具有较高归因贡献的地理知识 Experts。
随后，作者屏蔽这些重要 Experts，重新测试模型的 MRR，研究知识是否集中在少数 Experts 中，以及其他 Experts 能否提供替代能力。
最终发现：
- Qwen 1.5-MoE：深层架构与 Shared Expert 提供一定的层级冗余，但地理知识等核心任务仍然可能依赖少数专门化 Experts。
- OLMoE：浅层架构使 Experts 需要承担更多任务，知识分工不够精细，部分重要 Expert 失效后容易出现性能下降。
- Mixtral：Expert 数量较少、Top-k 相对较大，路由参与更加密集，知识贡献分布相对分散，具有较好的故障容忍能力。
因此作者提出，MoE 架构应该根据任务特点平衡知识专门化、网络深度和 Expert 冗余。

## 使用方法和技巧

### 1. Neuron-Level Attribution：神经元级知识归因

通过将神经元的输出贡献加入当前隐藏状态，计算正确 Token 的对数概率变化，衡量神经元的重要性。

对于 Attention 神经元：
\[
I(v_A^l)=
\log p(y_i\mid h_i^{l-1}+v_A^l)
-\log p(y_i\mid h_i^{l-1})
\]

对于 Dense FFN 神经元：
\[
I(v_F^l)=
\log p(y_i\mid u_i^l+v_F^l)
-\log p(y_i\mid u_i^l)
\]

对于 MoE Expert 神经元：
\[
I(v_{E_j}^l)=
\log p(y_i\mid u_i^l+g_{i,j}^lv_{E_j}^l)
-\log p(y_i\mid u_i^l)
\]

其中 \(g_{i,j}^l\) 是 Router 对 Expert 的门控权重。

重要性分数越高，说明该神经元对正确 Token 的预测贡献越大。

### 2. Layer Efficiency：层级知识处理效率

将各层 FFN 的知识归因贡献进行统计，定义：

\[
\text{Layer Efficiency}
=
\frac{\text{Total FFN Gain}}
{\text{Number of Layers}}
\]

用于衡量模型平均每一层产生的 FFN attribution gain。

例如，论文 Table 1 中 MoE 的平均单层归因贡献更高，但这不等同于实际推理速度或 FLOPs 效率。

### 3. Expert Ablation：专家消融与有效 Expert 阈值

分别限制模型参与计算的 Experts，观察 HIT@10 和 MRR 的变化。

通过比较：
- Only Shared；
- Only Top-1；
- Shared + Top-1；
- Shared + Top-2；
- Shared + Top-4；
研究 Shared Expert 和 Routed Experts 的功能互补性。

进一步增加 Top-k，观察性能是否继续提升，从而判断是否存在 Effective Expert Threshold，即完成当前任务所需的核心知识是否主要集中在少数 Experts 中。

### 4. Semantic-Driven Routing：语义驱动的专家路由

通过 Attention Head 与 Routed Expert 的活动相关性，分析两者是否存在语义协同。

论文报告：
\[
r(H_{10}^{18},E_{59}^{20})=0.68
\]

\[
r(H_{15}^{23},E_{23}^{20})=0.62
\]

进一步采用三种验证方法：
- Head Suppression：抑制 \(H_{12}^{19}\)，Top-4 Experts 的平均 gating probability 下降54%，MRR下降29%。
- Expert Forcing：强制激活 \(E_{59}^{20}\)，恢复原始 MRR 的85%。
- Integrated Gradients：平均约28%的 Expert activation attribution 可以追溯到特定 Attention Heads。
### 5. Expert Knowledge Localization：专家知识定位

针对每种知识关系，选取 Top-100 高重要性神经元，根据神经元所属的 Expert 进行统计，找出知识关系对应的 Top-5 Experts。

通过这种方式定位不同知识关系的主要贡献模块，例如：
- country_capital_city；
- country_language；
- name_birthplace；
- fruit_inside_color；
- object_superclass。
此外，作者还统计了不同知识关系之间的神经元重叠情况，发现 Shared Expert 中同一种关系的不同样本往往共享部分重要神经元，而不同关系之间的重叠通常较小。

### 6. Expert Blocking：专家故障与鲁棒性评估

屏蔽知识归因排名靠前的 Top-1、Top-5、Top-10 Experts，重新计算 MRR，衡量模型对关键 Experts 的依赖程度。

例如：
- Qwen 1.5-MoE 的 country_capital_city：屏蔽 Top-10 后，MRR 从 0.71 降至 0.40。
- Qwen 1.5-MoE 的 name_birthplace：MRR 从 0.85 降至 0.81。
- OLMoE 的 country_language：MRR 从 0.96 降至 0.68。
- OLMoE 的 object_superclass：MRR 从 0.80 降至 0.66。
通过不同知识关系的故障测试，区分对核心 Experts 高度敏感的任务（Core-sensitive tasks）和能够由更多 Experts 分担的任务（Distributed-tolerant tasks）。

### 7. 实验设置与评价指标
- 模型： Dense 模型包括 Llama-7B、Qwen 1.5-7B、Mistral-7B；MoE 模型包括 Qwen 1.5-MoE、OLMoE、Mixtral-8x7B。
- 数据集： LRE，选择12种知识关系，覆盖语言模式、常识、事实知识等领域。
- 实验方式： 使用模型的零样本知识预测能力进行分析。
- HIT@10： 正确答案是否出现在模型预测排名前十。
- MRR： 正确答案排名倒数的平均值，越高表示正确答案整体排名越靠前。