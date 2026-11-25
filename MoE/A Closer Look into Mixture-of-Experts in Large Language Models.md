文章链接：https://aclanthology.org/2025.findings-naacl.251/

## 问题

MoE 通过 Router 动态选择 Top-k 个 Expert 处理输入 Token，理论上可以将不同的知识和计算任务分配给不同 Expert，从而形成具有专业分工的模块化网络。

但目前仍不清楚 MoE 是否真正实现了这种模块化：不同 Expert 是否学习了不同的功能，还是仅仅具有不同的参数，却执行着相似的计算？

这产生了三个问题：

1. Expert 是否真正具有功能差异？ 不同 Expert 的参数是否相似？即使参数不同，它们对相同 Token 的输出是否也不同？仅通过权重无法判断 Expert 是否真正实现了功能分工。

2. Router 与 Expert 内部的计算是否存在关联？ Router 负责选择 Expert，而 Expert 内部的激活矩阵 \(W_{act}\) 负责选择激活哪些 Neuron。两种选择机制是否存在关联？Neuron 是否也可以被看作更细粒度的 Expert？

3. Router 究竟偏好什么样的 Expert？ Router 根据输入隐藏状态计算 Gate Score，但它是否倾向于选择输出幅度更大的 Expert？不同网络深度的 Expert 是否具有不同程度的专业化？

## 总结

本文对 Mixtral 8×7B、Mixtral 8×22B、DeepSeekMoE 和 Grok-1 四种 MoE 模型进行静态参数和动态行为分析，研究 Expert 的参数异质性、输出行为以及 Router 的选择机制。

主要发现：

1. Neuron 可以看作更细粒度的 Expert。 Router 的 Gate Embedding \(W_g\) 与 Expert 内部激活矩阵 \(W_{act}\) 的相似度具有较强相关性，说明 Expert 级别的选择与 Neuron 级别的选择可能存在共同机制。

2. Expert 的多样性总体随网络深度增加，但最后一层是例外。 Mixtral 和 DeepSeek 的 Expert 参数及输出相似度通常随深度增加而下降，但最后一层的输出相似度明显上升，表现出重新趋同的现象。

3. Router 倾向于选择输出范数较大的 Expert。 在 Mixtral 中这一现象尤其明显，DeepSeek 也存在一定程度的关联，而 Grok 没有表现出明显的对应关系。

4. 参数相似度可以在一定程度上反映行为相似度。 对大量 Token 的输出进行平均后，Expert 输出相似度与 Neuron 权重相似度具有相似的分布模式。但 Grok 是一个反例：参数差异很大，实际输出却高度相似。

基于这些发现，作者提出三方面建议：
- 从 Neuron 粒度研究 Expert 的划分、构建和组合。
- 深层增加 Expert 数量，最后一层适当减少。
- 考虑使用 Expert 输出范数作为 Router 的选择依据。
此外，作者推测，从头训练 MoE 比从 Dense 模型 Upcycling 更有利于 Expert 多样化。

## 思路

作者首先从 MoE 的模块化程度入手，将研究分为静态参数分析和动态行为分析两个阶段。

第一阶段：分析 Expert 的参数是否真正具有差异。
作者首先比较不同 Expert 的三个核心权重矩阵：
\[
W_{up},\quad W_{act},\quad W_{down}
\]

通过计算矩阵之间的余弦相似度，发现 Mixtral 的 Expert 参数相似度相对较高，而 DeepSeek 和 Grok 的相似度普遍接近零。
但直接比较整个矩阵存在一个问题：即使两个 Expert 包含相似的 Neuron，如果 Neuron 的排列顺序不同，直接展平矩阵也可能得到较低的相似度。
因此，作者进一步从 Neuron 粒度分析，采用平均和重新排序两种方法，减少 Neuron 排列顺序对相似度计算的影响。
在此基础上，作者又将 Router 的 Gate Embedding \(W_g\) 与 Expert 内部的 Neuron 权重进行比较，发现 \(W_g\) 与 \(W_{act}\) 的相似度相关性最高。
由此提出：Expert 内部的 Neuron 也可能具有类似 Expert 的选择和分工属性，可以将其看作更细粒度的计算专家。

第二阶段：分析 Expert 的实际输出行为。
仅仅发现 Expert 参数不同，还不能证明它们的实际功能不同。
因此，作者设计了两阶段推理实验：
- 第一阶段：按照模型原本的 Top-k 路由正常推理，保存每一层真实产生的隐藏状态。
- 第二阶段：逐层重新输入保存的隐藏状态，将 Top-k 设置为 ALL，使同一个 Token 分别经过当前层的所有 Expert，记录每个 Expert 的输出。
这样就可以在相同输入条件下，直接比较不同 Expert 的输出向量相似度，而不会因为修改前面层的路由而改变后续层的输入。

实验发现：
- Mixtral 和 DeepSeek 的 Expert 输出总体随网络深度增加而分化。
- 但最后一层的 Expert 输出相似度明显上升，重新趋同。
- Grok 的 Expert 参数虽然高度不同，输出却依然高度相似。
这说明参数差异不一定意味着实际功能差异。

第三阶段：研究 Router 为什么选择特定 Expert。
既然实验发现 Router 选择的 Expert 往往具有较高的输出相似度，作者进一步研究 Expert 输出的 \(L_2\) 范数与 Gate Score 的关系。
结果发现，Mixtral 中被 Router 选中的 Expert 通常也是输出范数最大的 Expert，DeepSeek 存在一定程度的类似现象。
由此推测，Expert 输出范数可能是 Router 决策的重要因素之一。

第四阶段：将观察结果转化为架构设计建议。
根据深层 Expert 差异增大、最后一层重新趋同的现象，作者提出不同网络深度可以采用不同数量的 Expert。
为验证这一想法，作者从头训练了六个 MoE 模型，每个模型包含 24 层、3.6B 参数，使用约 120B Tokens。其中一个模型的 24 层全部使用 MoE，其余五个模型将不同位置的一层 MoE 替换为普通 Dense FFN。
实验发现，将 Dense FFN 放在越深的位置，模型性能总体越差；但将其放在最后一层时，性能反而出现小幅改善。
这初步支持了深层需要保留更多 Expert 计算，而最后一层可以适当减少 Expert 的设计思路。

## 使用方法和技巧

### 1. Matrix-level Similarity：Expert 矩阵级相似度

分别提取每个 Expert 的三个权重矩阵：

\[
W_{up},W_{act},W_{down}
\]

将矩阵展平为一维向量，计算不同 Expert 之间的余弦相似度：

\[
S_{ij}^{M}
=
\frac{
\operatorname{vec}(W_i^M)
\cdot
\operatorname{vec}(W_j^M)
}{
\|\operatorname{vec}(W_i^M)\|
\|\operatorname{vec}(W_j^M)\|
}
\]

其中 \(M\) 表示三个权重矩阵之一。

将结果绘制成热力图，观察不同 Expert 的参数相似性及其随网络深度的变化。

### 2. Neuron-level Similarity：Neuron 级相似度

为解决不同 Expert 内部 Neuron 排列顺序不同导致的相似度偏差，作者采用两种方法：

- Averaging： 对 \(W_{up}\)、\(W_{act}\) 的行向量以及 \(W_{down}\) 的列向量分别求平均，再计算不同 Expert 之间的余弦相似度。

- Reordering： 使用 Jonker-Volgenant 算法解决线性分配问题，寻找两个 Expert 中 Neuron 的最佳匹配顺序，使匹配后的 Neuron 相似度最大。

其中 Reordering 更关注不同 Expert 是否包含相似的 Neuron，只是排列顺序不同。

### 3. Gate Embedding–Neuron Correlation：Router 与 Neuron 的关联

分别计算 Router 的 Gate Embedding 和 Expert 内部 Neuron 的两两余弦相似度：

\[
X_{ij}=\cos(W_g[i,:],W_g[j,:])
\]

\[
Y_{ij}^{M}=
\operatorname{NeuronSimilarity}
(W_i^M,W_j^M)
\]

然后将两组相似度配对，计算 Pearson 相关系数，并对不同层的相关系数平方取平均：

\[
R_{\mathrm{avg}}^2
=
\operatorname{Average}(R_l^2)
\]

分别比较：
- \(W_g\) 与 \(W_{up}\)
- \(W_g\) 与 \(W_{act}\)
- \(W_g\) 与 \(W_{down}\)
实验发现 \(W_g\) 与 \(W_{act}\) 的相关性最高，说明 Router 的 Expert 选择与 Expert 内部的 Neuron 激活选择具有一定关联。

### 4. Two-stage Inference：两阶段推理

用于分析不同 Expert 在相同输入下的实际输出行为。
1. 正常执行模型推理，保存每一层的隐藏状态 \(z_i\)。
2. 逐层重新输入对应的原始隐藏状态 \(z_{i-1}\)，将当前层的 Top-k 设置为 ALL。
3. 记录当前层所有 Expert 的输出：
\[
E_1(z_{i-1}),E_2(z_{i-1}),\ldots,E_N(z_{i-1})
\]

4. 计算不同 Expert 输出向量之间的余弦相似度，并绘制热力图。

实验同时使用短文本和长文本。长文本实验还使用 Angular Similarity，将余弦相似度映射到 \([0,1]\)，便于对多个 Token 的相似度求平均：

\[
\operatorname{AngularSim}(a,b)
=
1-\frac{\arccos(\cos(a,b))}{\pi}
\]

最终得到不同 Expert 在大量 Token 上的平均行为相似度。

### 5. Expert Output Norm vs. Gate Score：输出范数与路由分数

计算每个 Expert 对当前 Token 的输出向量 \(L_2\) 范数：

\[
\operatorname{Norm}_i(x)=\|E_i(x)\|_2
\]

同时提取 Router 对应的 Gate Score：

\[
g_i(x)
\]

将两者绘制在同一张图中，观察 Router 是否倾向于选择输出范数较大的 Expert。

实验发现：
- Mixtral：被选中的 Expert 通常具有最大的输出范数。
- DeepSeek：存在一定关联，但不如 Mixtral 明显。
- Grok：没有明显的正相关关系。
这一方法可以用于分析 Router 是否偏好输出幅度较大的 Expert，也为 Norm-based Routing 提供经验依据。

### 6. Dynamic Expert Allocation：不同网络深度的 Expert 分配

作者训练六个从头初始化的 MoE 模型：
- 每个模型 24 层，3.6B 总参数。
- 训练约 120B Tokens。
- 一个模型所有层均使用 MoE。
- 其余模型分别将不同位置的一层 MoE 替换为 Dense FFN。
比较模型的 PPL 和 HellaSwag、MMLU、GSM8K、CEval 等任务表现。

结果显示，深层替换为 Dense FFN 通常会损害性能，而最后一层替换为 Dense FFN 则有小幅改善。

这为按照网络深度动态调整 Expert 数量提供了初步实验依据。