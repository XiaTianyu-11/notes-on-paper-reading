文章链接：https://www.usenix.org/conference/usenixsecurity26/presentation/wu-lichao

## 问题

现有 LLM 安全对齐研究主要集中在 Dense 模型，对 MoE 的安全机制缺乏细粒度分析。

Dense 模型的所有参数都会参与每次输入的计算，而 MoE 通过 Router 动态选择 Top-k Expert。因此，不同输入可能经过不同的计算路径，安全对齐可能集中在部分 Expert 及其内部神经元中。

这产生了两个问题：

1. **安全对齐的分布位置不明确：** 现有 MoE 安全研究主要停留在 Expert 粒度，通常假设存在专门负责安全控制的 Expert，但没有进一步研究安全功能是否实际上分布在多个 Expert 的部分神经元中。
2. **现有攻击的干预粒度过粗：** SAFEx 等方法通过屏蔽安全相关 Expert 破坏安全对齐，但整个 Expert 还承担其他正常计算任务，因此可能造成不必要的模型能力损失。

因此，需要找到 MoE 内部真正承担安全对齐功能的神经元，在尽可能少地干预模型计算的情况下，削弱其安全拒绝能力。

## 总结

提出 GateBreaker，一种针对 MoE LLM 的训练无关、推理时攻击框架。

通过 Gate-level Profiling、Expert-level Localization 和 Targeted Safety Removal 三个阶段，先利用 Router 的路由信息筛选安全相关 Expert，再通过有害输入与正常输入的神经元激活差异定位安全神经元，最后在推理时将这些神经元的激活置零。

该方法不需要修改模型权重或重新训练模型，只需要干预少量 Expert 内部神经元。

在 8 个 MoE LLM 上，GateBreaker 将平均 ASR 从 7.4% 提升到 64.9%，平均每个目标层只需要移除 2.6% 的神经元。相比已有 SAFEx，平均 ASR 从 29.9% 提升到 64.9%。

此外，安全神经元可以在同一家族模型之间迁移，方法也可以扩展到 MoE VLM。

## 思路

作者首先发现，MoE 的安全对齐可能不是均匀分布在所有 Expert 中，而是集中在部分 Expert 的局部计算结构中。

但是，作者进一步意识到，不能简单假设存在一个专门处理有害输入的 Safety Expert。由于 MoE 存在负载均衡，而且有害输入与正常输入可能经过相似的路由路径，安全功能更可能由多个 Expert 协同实现。

因此，作者提出逐级定位的思路：

- 第一阶段：利用 Router 的路由信息，统计有害 Prompt 下每个 Expert 的激活频率，筛选安全相关 Expert。
- 第二阶段：在候选 Expert 内部，比较有害 Prompt 与正常 Prompt 的神经元激活差异，利用安全权重和 Z-score 找到安全神经元。
- 第三阶段：推理时将选中的安全神经元激活置零，保留 Router 和其他神经元的正常计算。

具体分为三个方法：

**1. Gate-level Profiling：**

对于每一层的每个 Expert，统计其在有害 Prompt 中被 Router 选择的频率，并在整个有害数据集上取平均，得到 Expert Utility Score。

选择得分最高的 Top-3k Expert 作为候选安全 Expert。

**2. Expert-level Localization：**

分别使用有害 Prompt 和正常 Prompt，提取候选 Expert 内部的神经元激活。

对于每个神经元，计算有害输入和正常输入下的平均激活差异，得到安全权重：

\[
w_{l,j,n}=
\mathbb E_{D_{harm}}[v_n]
-
\mathbb E_{D_{benign}}[v_n]
\]

随后在同一个 Expert 内对安全权重进行 Z-score 标准化，选择满足 \(z_{l,j,n}>\tau\) 的神经元。

**3. Targeted Safety Removal：**

在推理过程中，将选中的安全神经元激活置零：

\[
A'_{l,j}(x)_n=
\begin{cases}
0,&n\in N_{safety}\\
A_{l,j}(x)_n,&\text{otherwise}
\end{cases}
\]

不修改模型参数，也不改变 Router 的路由机制，只削弱指定神经元对 Expert 输出的贡献。

此外，作者还提出跨模型迁移攻击：在源模型中识别安全神经元，然后直接将相同的神经元位置应用到同一家族的目标模型，无须重新执行 Profiling 和 Localization。

## 使用方法和技巧

### 1. Runtime Compute Graph Patching：运行时计算图改造

不同 MoE 模型可能将 Router 和 Expert 内部计算融合，导致无法直接获取所需的中间信息。

作者通过运行时计算图改造，将计算逻辑拆分为 Gate Module 和 Expert Module，暴露 Router Logits、Dispatch Mask 和 Expert 内部激活。

对于 GPT-OSS、Hunyuan、DeepSeek 等不同实现，分别进行融合层拆分、路由信息暴露和非标准路由逻辑适配。

### 2. Expert Utility Score：Expert 级别筛选

利用 Router 的 Top-k 选择结果，统计每个 Expert 在有害输入中的激活频率：

\[
U_{l,j}=
\frac{1}{|D_{harm}|}
\sum_{M\in D_{harm}}f_{l,j}(M)
\]

按照得分选择 Top-3k Expert，作为后续神经元分析的候选范围。

### 3. Contrastive Activation Analysis：对比激活分析

构建数量相同的有害 Prompt 和正常 Prompt 数据集。

分别提取 Expert 内部的神经元激活，并使用逐元素最大值聚合：

\[
v_n(q)=\max_{t\in q}A_n(t)
\]

这样可以保留单个 Prompt 中神经元的峰值激活，避免平均值稀释少量 Token 的强激活。

### 4. Safety Weight + Z-score：安全神经元定位

通过两类输入的平均激活差异衡量神经元与拒绝行为的关联：

\[
w_n=
\mathbb E_{D_{harm}}[v_n]
-
\mathbb E_{D_{benign}}[v_n]
\]

再使用 Z-score 标准化：

\[
z_n=\frac{w_n-\mu_w}{\sigma_w}
\]

选择 \(z_n>\tau\) 的神经元，论文默认 \(\tau=2\)。

### 5. Layer-wise Safety Removal：跨层安全功能分析

分别在前半部分 MoE 层和全部 MoE 层中移除安全神经元，比较不同干预范围下的 ASR。

用于判断安全对齐是否集中在少数层，还是分布在整个模型的多个层中。

实验中，平均 ASR 从不干预时的 7.4%，提升到前半层干预时的 40.8%，再提升到全层干预时的 64.9%。

### 6. One-shot Transfer Attack：跨模型神经元迁移

在源模型中识别安全神经元，然后直接将相同的神经元位置应用到同一家族的目标模型。

不需要重新执行安全 Expert 筛选和神经元定位。

实验表明，安全神经元在同一家族的领域微调模型、指令微调模型和推理增强模型之间具有一定的迁移能力。

### 7. Expert Ablation：验证 Expert 与安全功能的关系

根据有害输入下的 Expert Utility Score，从高到低或从低到高依次移除 Expert，观察 ASR 变化。

从高分到低分移除时，ASR 快速增加；从低分到高分移除时，ASR 提升较小，甚至可能下降。

这验证了不同 Expert 对安全行为的贡献存在差异，但安全功能并不集中在单一 Expert 中。

### 8. Evaluation Metrics：攻击效果与正常能力评估

- ASR：衡量有害 Prompt 产生有害输出的比例。
- Safety Neuron Ratio：衡量被干预神经元的数量占比。
- Utility：使用 CoLA、RTE、WinoGrande、OpenBookQA 和 ARC 评估正常任务能力。

此外，使用 SAFEx 作为对照方法，验证神经元级干预相对于整个 Expert 级干预的效果。