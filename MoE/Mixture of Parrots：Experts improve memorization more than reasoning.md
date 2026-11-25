文章链接：https://proceedings.iclr.cc/paper_files/paper/2025/hash/5bc3356e0fa1753fff7e8d6628e71b22-Abstract-Conference.html

## 问题

MoE 通过增加 Expert 数量，可以在基本不增加 active parameters 和计算量的情况下，大幅增加模型的 total parameters。但这是否意味着增加 Experts 就能全面提升模型能力？

这里存在一个值得研究的问题：不同类型的任务是否具有不同的参数需求？
1. 记忆能力（Memorization）： 模型需要记住大量训练样本或事实知识。Dense 模型需要通过增加自身参数来存储这些信息，而 MoE 能否通过增加 Experts，将不同知识分散存储在不同 Expert 中，从而降低每次推理所需的 active parameters？
2. 推理能力（Reasoning）： 模型需要同时处理多个信息，并进行复杂的信息组合。MoE 虽然增加了 total parameters，但每个 token 只激活少量 Experts，那么增加 Experts 能否真正提高模型单次计算所能处理的信息量？
因此，作者希望回答：MoE 增加 Experts 带来的额外参数，究竟更适合增强记忆能力，还是也能同样有效地增强推理能力？

## 总结

本文通过理论分析、合成任务实验和真实预训练模型实验，发现 MoE 增加 Experts 对记忆能力的提升明显大于对推理能力的提升。

核心结论：
1. 对于记忆任务： MoE 可以将不同训练样本分配给不同 Experts，每个 Expert 只需要记忆其中一部分。因此，增加 Experts 能够增加模型的总存储容量，同时降低每次推理需要激活的参数量。相同 total parameters 下，MoE 可以达到 Dense 的记忆效果，但使用更少的 active parameters。
2. 对于推理任务： 推理需要在同一次计算中整合多个信息，受到模型 active width 的限制。增加 Experts 并不会增加当前计算所能使用的表示维度，因此单纯增加 Experts 无法有效突破推理能力瓶颈。
3. 真实预训练模型验证： 在 World Knowledge 任务中，MoE 可以充分利用额外 Experts 提高事实知识检索能力；但在 Commonsense 和 Math Reasoning 任务中，增加模型 width 比增加 Experts 更有效。此外，MoE 在数学任务上表现出更大的训练集与测试集性能差距，说明其可能更倾向于记忆训练数据。
因此，MoE 更适合作为一种高效的知识记忆架构，而不是单纯依靠增加 Experts 就能持续增强推理能力的架构。


## 思路

作者首先从理论上分析 Dense 和 MoE 在两类任务上的能力差异，然后通过合成任务和真实预训练模型验证。

1. 理论分析：区分记忆与推理所需的参数
作者构建一个简化的单层 Transformer，分别分析 reasoning 和 memorization。
- Reasoning： 使用图上的 Length-2 Path Problem，判断两个节点之间是否存在长度为 2 的路径。作者通过将其转化为 Set Disjointness 问题，利用通信复杂度证明：MoE 要解决该问题，必须满足 hidden width 的下界：
\[
mH=\Omega\left(\frac{|V|}{\log N}\right)
\]
其中 \(m\) 是 hidden dimension，\(H\) 是 attention head 数量，\(V\) 是图的顶点集合。
这个下界与 Expert 数量 \(K\) 无关。原因是 Router 和 Expert 都只能处理 Attention 已经生成的表示，无法恢复此前没有保留下来的信息。
接着，作者证明 Dense 模型只需要 width 为 \(|V|\) 就可以解决该问题。
进一步，在相同 total parameters 下：
\[
m_d^2\approx Km_s^2
\]
因此：
\[
m_s=\frac{m_d}{\sqrt K}
\]
随着 Experts 增加，单个 Expert 的 width 会下降。当 Experts 足够多时，MoE 的 width 可能低于推理任务要求的临界值，而 Dense 仍然可以解决问题。
- Memorization： 作者构造随机 Gaussian 输入和任意标签的记忆任务，要求模型记住每个输入对应的标签。
MoE 可以通过 Router 将不同样本分配给不同 Experts，使每个 Expert 只需要记忆约 \(n/K\) 条样本。
作者证明：
\[
P_{\text{total}}
=
\tilde O(n+Km)
\]
\[
P_{\text{active}}
=
\tilde O\left(\frac nK+Km\right)
\]
选择：
\[
K=\sqrt{\frac nm}
\]
就可以得到：
\[
P_{\text{active}}=\tilde O(\sqrt{nm})
\]
而 Dense 模型至少需要：
\[
P_{\text{Dense}}=\tilde\Omega(n)
\]
因此，MoE 可以用更少的 active parameters 完成相同的记忆任务。

2. 合成实验：验证两种任务的参数需求差异
作者设计两个任务：
- Phone-book Memorization： 生成大量姓名和电话号码，姓名由 5 个字母组成，电话号码由 8 位数字组成。模型训练时记忆整个电话簿，测试时只提供姓名，要求模型直接输出对应电话号码。通过模型能够达到 90% 准确率的最大电话簿规模衡量记忆能力。
实验发现，MoE 的记忆能力主要取决于 total parameters，而不是 active parameters。即使只激活 64 个 Experts 中的 2 个，也可以充分利用其他 Experts 中存储的信息。例如，只有 42M active parameters 的 MoE，记忆能力可以超过参数量达到其 10 倍的 Dense 模型。
- Shortest Path： 输入随机生成的有向图及起点、终点，要求模型输出两点之间的最短路径。测试时使用训练阶段没有出现过的图，衡量模型的推理和泛化能力。
实验发现，增加 Experts 并不能稳定提升准确率，模型表现更多取决于 active parameters。也就是说，增加 width 比增加 Experts 更有效。

3. 真实预训练模型：验证理论是否适用于 LLM
作者使用 Dense 和 MoE 模型进行预训练，并在三类下游任务上评估：
- World Knowledge： TriviaQA、Natural Questions、HotpotQA 等，主要考察事实知识的记忆和检索。
- Commonsense： ARC、HellaSwag、WinoGrande 等，主要考察常识推理。
- Math： GSM8K、Hendrycks-MATH、Minerva-MATH 等，主要考察数学推理。
实验发现：
- World Knowledge：MoE 可以充分利用额外 Experts，在相同 total parameters 下达到接近 Dense 的效果。
- Commonsense 和 Math：相同 total parameters 下，MoE 通常不如 Dense，增加 width 比增加 Experts 更有效。

4. 进一步分析：MoE 是否更倾向于记忆训练数据？
作者进行了两组补充实验。
- Generalization Gap： 从 OpenMathInstruct 中选取训练样本，分别计算模型在这些训练样本和 GSM8K、Hendrycks-MATH 测试集上的准确率，比较训练准确率与测试准确率之间的差距。
结果发现，在模型规模超过 159M 参数后，MoE 的训练集与测试集准确率差距通常比 Dense 更大。这说明 MoE 可能更容易记忆训练题，而不是将这些知识转化为可泛化的推理能力。
- 固定 Validation Perplexity： 不再按照参数量比较，而是选择 Validation Perplexity 相同的模型，比较其下游任务表现。
结果发现，MoE 在 World Knowledge 上优于 Dense，而在 Commonsense 和 Math 上与 Dense 大致相当。
作者据此提出 Architecture Implicit Bias：即使两个模型达到相同的训练分布拟合程度，不同架构也可能倾向于学习不同的函数。MoE 更倾向于通过记忆事实和训练样本来降低损失。

## 使用方法和技巧

### 1. Total Parameters vs Active Parameters：区分总参数和激活参数

这是全文最重要的分析视角。
- Total Parameters：模型所有 Experts 的参数总量，反映模型可以存储多少信息。
- Active Parameters：一次前向传播中实际参与计算的参数量，反映当前计算可以使用多少模型容量。
对于 MoE：

\[
P_{\text{total}}\propto K m^2
\]

\[
P_{\text{active}}\propto k m^2
\]

其中 \(K\) 是总 Expert 数量，\(k\) 是每次激活的 Expert 数量。

因此，研究 MoE 的记忆能力时，应重点关注 total parameters；研究推理能力时，则需要重点关注 active parameters 和 width。

### 2. Communication Complexity：利用通信复杂度证明推理能力下界

将 Length-2 Path 转化为 Set Disjointness：

\[
A=\{v:s\rightarrow v\}
\]

\[
B=\{v:v\rightarrow d\}
\]

判断：

\[
A\cap B\neq\emptyset
\]

就等价于判断是否存在长度为 2 的路径。

通过通信复杂度，可以证明模型必须具有足够大的 hidden representation 才能完成信息交互。这种方法可以用来分析模型架构中的信息瓶颈，而不仅仅依赖实验观察。

### 3. Routing-based Memorization：通过路由实现记忆分区

作者在理论证明中构造了一种基于 Gaussian 输入的均匀路由方法：
- 使用输入向量的部分维度进行路由。
- 将不同样本均匀分配给 \(K\) 个 Experts。
- 每个 Expert 只负责记忆分配给自己的样本。
- 推理时通过 Router 找到对应 Expert，读取其记忆的信息。
最终实现：

\[
\text{总存储容量}\propto n
\]

\[
\text{单次激活参数量}
\propto \frac nK+Km
\]

这提供了一种理解 MoE 记忆机制的视角：Router 不只是计算资源分配模块，也可以作为不同记忆单元的寻址机制。

### 4. Phone-book Task：构造闭卷检索任务衡量记忆能力

将随机生成的姓名和电话号码组成映射关系：

\[
\text{Name}\rightarrow\text{Phone Number}
\]

训练时让模型记住所有映射，测试时不提供原始电话簿，只给出姓名，让模型生成对应号码。

通过最大可记忆电话簿规模衡量模型的记忆容量。

这种方法可以排除模型直接从输入中复制答案的可能性，比较适合研究模型对任意映射关系的记忆能力。

### 5. Generalization Gap：通过训练集与测试集差距分析记忆倾向

定义：

\[
Gap=Acc_{\text{train}}-Acc_{\text{test}}
\]

如果模型在训练样本上的表现明显优于测试样本，说明模型可能更多地记住了训练数据，而没有学到可以泛化的规律。

本文利用这个指标，分析 MoE 在数学任务中是否存在更强的训练数据记忆倾向。

### 6. Fixed Perplexity Comparison：控制语言建模拟合程度，比较架构的隐式偏置

传统比较通常固定参数量或计算量，但这不能排除不同架构对训练数据的拟合程度不同。

作者改为固定 Validation Perplexity，再比较下游任务表现。

如果两个模型的 Perplexity 相同，但 MoE 的事实知识检索能力更强，就说明它们可能采用了不同的学习策略。

这为研究不同架构如何在记忆与推理之间分配学习能力提供了实验方法。