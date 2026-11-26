文章链接：https://arxiv.org/abs/2604.02178v2

## 问题

Dense Transformer 中，一个 neuron 往往同时表示多个无关概念，即 Polysemanticity（多义性）。这是由 Superposition（超叠加）造成的：模型需要用有限的神经元维度表示大量特征，因此不同特征会混合在同一神经元中。虽然 SAE 等方法可以尝试拆解这些混合特征，但需要额外训练模型，计算成本较高。

MoE 通过 Router 将不同 token 分配给少量 Expert，每个 Expert 只处理整个数据分布中的一部分输入。因此，稀疏路由可能减少 Expert 内部神经元的 Superposition，使 MoE 天然具有更好的可解释性。但此前这一现象主要在 Toy Model 中得到验证，尚不清楚在大规模 LLM 中是否成立。

同时，对于 MoE Expert 的专门化方向，已有研究存在两种不同观点：
1. Domain Specialization： 不同 Expert 专门处理不同语义领域，例如 Coding、Biology、Legal 等。
2. Token/Syntactic Specialization： 不同 Expert 根据 token 类型、词法或句法特征进行分工，例如处理特定词缀、标点符号和语法结构。
这两种观点都存在局限：Expert 可能并不是负责整个领域，也不是简单处理某种 token，而是在特定输入条件下执行某种具体的计算操作。

因此，本文希望回答三个问题：
1. MoE 的稀疏路由能否降低 neuron 的 Polysemanticity，使单个 neuron 更接近 Monosemanticity（单义性）？
2. 如果 Expert 内部的 neuron 已经比较单义，能否直接将整个 Expert 作为一个功能模块进行解释，而不必逐个分析 neuron 或训练 SAE？
3. Expert 究竟是领域专家、Token 专家，还是执行特定计算操作的 Task Expert？不同层的 Expert 是否存在功能分工？
## 总结

提出一种面向 MoE Expert 的可解释性分析框架，核心观点是：Sparse Routing 不仅提高计算效率，还能减少 Expert 内部的 Superposition，使 neuron 更接近单义表示，进而让整个 Expert 成为可直接解释的功能模块。

作者通过三个层次展开验证：
1. Neuron Level： 使用 K-sparse probing 比较 MoE 与 Dense FFN 的神经元多义性。发现 MoE 的 neuron 更容易用单个神经元识别特定概念，而且路由越稀疏，这种现象越明显。
2. Expert Level： 使用 Expert 激活样本、Logit Lens 和 LLM 自动生成自然语言标签，再通过独立 LLM Scorer 和 DLA 验证标签。结果表明，大量 Expert 可以被描述为相对明确的功能模块。
3. Specialization Level： 通过 Unembedding Matrix 聚类和 Jensen-Shannon Divergence（JSD），分别衡量 Expert 的输入专门化和输出功能专门化，发现深层 Expert 更倾向于执行细粒度的计算任务，而不是仅仅负责宽泛的语义领域。
最终提出 Modular Monosemanticity 的观点：Router 限制 Expert 的输入范围，Expert 内部又具有相对单义的 neuron，两者共同使整个 Expert 呈现出连贯的计算功能。

因此，可以将 MoE 看成由大量功能模块组成的计算系统，不同 Expert 类似执行特定操作的子程序（Sub-routines），Router 则负责根据输入动态组织这些子程序。


## 思路

作者首先从 neuron 层面验证 MoE 是否比 Dense 模型更容易解释。

第一步：验证 Sparse Routing 是否降低 Polysemanticity。
使用 K-sparse probing，针对不同概念训练线性分类器，并逐渐增加参与分类的 neuron 数量。
如果只使用一个 neuron 就能准确识别某个概念，说明这个概念可能集中在一个 neuron 中；如果需要几十个 neuron 联合才能识别，则说明该概念分散在多个 neuron 中。
实验覆盖 12 个模型、58 个概念，包含词性、LaTeX、代码和自然语言文本四类。MoE 只使用实际被路由到目标 Expert 的 token 进行分析。
结果发现：
- MoE 在 \(k=1\) 时就能取得较高的 F1，Dense 模型则通常需要更多 neuron 才能达到类似效果。
- 在不同概念类别中，MoE 普遍表现出更好的单义性。
- 在控制参数量后，MoE 仍然优于 Dense 模型。例如 OLMoE-1B-7B 每个 token 激活约 1B 参数，而 OLMo-7B Dense 每个 token 激活约 7B 参数，前者仍表现出更低的 Polysemanticity。
- 路由越稀疏，单个 neuron 的概念表示越集中。例如 Qwen3-30B-A3B 的激活专家比例约为 0.06，表现优于激活比例为 0.25 的 Mixtral-8x7B。
由此说明，改善可解释性的关键不只是增加模型容量，而是稀疏路由改变了 Expert 内部的表示组织方式。

第二步：从 Neuron Level 转向 Expert Level。
既然 Expert 内部的 neuron 已经比较单义，而且 Router 会将相关输入限制在特定 Expert 中，那么整个 Expert 的计算就有可能呈现出更加连贯的功能。
作者将这种现象称为 Modular Monosemanticity：
- 内部：Neuron 更接近单义表示。
- 外部：Router 将语义或句法相关的输入分配给特定 Expert。

因此，不必再逐个解释 neuron，而是可以直接把 Expert 作为一个功能模块进行分析。

第三步：自动生成 Expert 的功能标签，并验证其合理性。
作者设计两阶段 LLM 自动解释流程：
- Explainer：找到 Expert 真正产生较大输出更新的文本样本，结合其上下文和 Logit Lens 结果，生成一句自然语言功能描述。
- Scorer：使用独立 LLM，根据正负样本判断生成的标签是否符合 Expert 的实际激活行为，并计算 F1。
实验中，大部分 Expert 的 F1 超过 0.8，最稀疏的 Qwen3-30B-A3B 经常超过 0.9。

为了进一步验证标签，作者设计 Trigger-Target 实验：根据 Expert 标签构造新的测试文本，指定触发 Expert 的 Trigger 和它应该推动的 Target Token，再使用 DLA 衡量目标 Expert 对 Target Token 的贡献，并与同层其他 Expert 比较。
Matched Prompt 中，目标 Expert 大多能进入 Top-1 或 Top-8 贡献排名；Control Prompt 中，约 80% 的情况下目标 Expert 根本没有被路由。
这表明 Expert 的自然语言标签不仅能描述其激活模式，也能得到输出归因方面的支持。

第四步：分析 Expert 的专门化究竟体现在哪些功能上。
作者根据自动生成的标签，将 Expert 归纳为不同类型：
- Morphological：处理词根、前缀、后缀和 Subword。
- Syntactic：处理句法关系和语法续写。
- Semantic：处理特定领域的语义知识。
- Operational：执行格式约束、结构闭合等具体操作。

此外还发现一些 Hyper-specialized Expert，专门处理非常狭窄的知识或表达模式。
但这些自然语言标签仍然属于定性解释。为了进一步验证，作者在 Unembedding Space 中进行 K-means 聚类，建立不同粒度的词表类别，再使用 JSD 衡量 Expert 的输入和输出分布与所在 Layer 平均分布之间的差异。
最终发现：
- 早期层的 Expert 主要表现为形态学和词汇构造相关功能。
- 中间层逐渐出现句法稳定和语义分区。
- 深层 Expert 的输出功能专门化显著增强，尤其集中在高粒度的词表类别中。

这说明深层 Expert 并不只是宽泛的领域专家，而更倾向于接收相对一般的输入，并执行精确的语义或句法操作。


## 使用方法和技巧

### 1. K-sparse Probing：衡量 Neuron 的单义性

使用 K-sparse probing 比较 MoE Expert 与 Dense FFN 的神经元表示。
首先计算每个 neuron 在正负样本上的平均激活差：

\[
a_j=
\left|
\mathbb E[h_j|y=1]
-
\mathbb E[h_j|y=0]
\right|
\]

选择差异最大的 Top-\(k\) 个 neuron，训练 Logistic Regression 分类器：

\[
\hat y=f(h_{j_1},h_{j_2},\ldots,h_{j_k})
\]

逐渐增加 \(k\)，观察 F1 的变化。
- \(k=1\) 时 F1 已经很高：说明单个 neuron 就能较好地表示该概念。
- 随着 \(k\) 增大，F1 明显提升：说明概念需要多个 neuron 联合表示。

### 2. Modular Monosemanticity：模块级单义性

通过两个条件解释 MoE Expert 的可解释性：
- Sparse Routing：限制每个 Expert 接收的输入分布。
- Monosemantic Neurons：Expert 内部的 neuron 更倾向于表示明确的概念。
两者共同使整个 Expert 形成相对连贯的计算模块。

### 3. Expert Activation Magnitude：筛选真正执行计算的样本

不能直接使用 Router Weight 判断 Expert 是否真正发挥作用。
因为被 Router 选中只意味着 Expert 获得了处理 token 的机会，并不代表它产生了有意义的输出。
因此使用：

\[
\boxed{
A_i(x)=g_i(x)\|E_i(x)\|_2
}
\]

其中：
- \(g_i(x)\)：Router 分配给 Expert 的权重。
- \(E_i(x)\)：Expert 输出向量。
- \(g_i(x)E_i(x)\)：Expert 写入 residual stream 的更新向量。
使用其 L2 范数衡量更新幅度。

对于文本序列 \(s\)，取其中最大的激活值：

\[
Score(s,E_i)
=
\max_{x\in s}g_i(x)\|E_i(x)\|_2
\]

然后选取得分最高的 20 条文本，作为 LLM Explainer 的输入。

### 4. Logit Lens：解释 Expert 输出倾向于推动哪些 Token

将 Expert 输出向量投影到 Unembedding Space：

\[
\ell_i(x)=E_i(x)W_U
\]

选取得分最高的 Top-3 Tokens，作为 Explainer 判断 Expert 功能的辅助信息。
它回答的是：
Expert 的输出方向倾向于提高哪些 Token 的 Logit？
例如某 Expert 经常在 LaTeX 数学表达式中激活，并且 Logit Lens 主要显示 }、\end、]，就可以推测它负责结构闭合，而不只是笼统地处理 LaTeX。

### 5. LLM Explainer + Scorer：自动解释 Expert

采用两个独立的 LLM：
- Explainer：输入 20 条高激活文本及对应的 Top-3 promoted tokens，生成一句 Expert 功能描述。
- Scorer：输入 10 条正样本和 10 条负样本，判断样本是否符合该描述，最终计算 F1。
负样本来自同一 Layer 中其他 Expert 激活的文本，以检验标签是否具有区分性。


### 6. DLA + Trigger-Target：验证 Expert 对输出的贡献

根据自动生成的 Expert Label，构造新的测试文本：
- Trigger：预期触发 Expert 的 token。
- Target：预期被 Expert 推动的 token。

例如：
We need to address the elephant in the room.
其中 the 是 Trigger，room 是 Target。
计算目标 Expert 对 Target Token 的 DLA：

\[
A_{E_i\rightarrow t}
=
\mathrm{LN}_{linear}(g_iE_i)^\top W_U[:,t]
\]

再与同一层其他 Expert 的 DLA 进行排序。

如果目标 Expert 在 Matched Prompt 中具有较高排名，而在 Control Prompt 中不被路由或贡献很低，就说明生成的功能标签得到了输出归因方面的支持。

### 7. Routing Specialization 与 Functional Specialization：区分输入专门化和输出专门化

这是第六节最重要的方法。
作者先对 Unembedding Matrix 的词表向量进行 K-means 聚类：

\[
k\in\{10,50,100,1000,5000\}
\]

不同聚类粒度对应不同层次的语义或句法类别：
- 低 \(k\)：宽泛类别，例如生物学。
- 高 \(k\)：细粒度类别，例如蜜蜂、GPU 纹理等。
随后分别统计两种分布。

Routing Specialization（输入专门化）
统计实际被 Router 分配给 Expert 的 token 属于哪些 cluster。
回答：
这个 Expert 通常接收什么类型的输入？

Functional Specialization（输出功能专门化）
对 Expert 输出使用 Logit Lens，统计其 Top-N promoted tokens 属于哪些 cluster。
回答：
这个 Expert 通常推动什么类型的输出？

### 8. Jensen-Shannon Divergence：量化 Expert 专门化程度

设：
- \(P_i\)：Expert \(i\) 的 cluster 分布。
- \(Q_L\)：所在 Layer 的整体基础分布。
- \(M=(P_i+Q_L)/2\)。

定义：

\[
\boxed{
S_i=JSD(P_i\|Q_L)
}
\]

其中：

\[
JSD(P_i\|Q_L)
=
\frac12 KL(P_i\|M)
+
\frac12 KL(Q_L\|M)
\]

分数越高，说明 Expert 的行为越偏离所在 Layer 的平均分布，专门化程度越高。

此外，作者还构建 Random Expert Baseline：按照 Layer 的基础分布随机抽取与真实 Expert 相同数量的 token，计算预期 JSD，并从实际 JSD 中减去该随机基线，以排除小样本造成的统计偏差。

实验结论：
- Domain Expert：预期在低 \(k\) 时就具有较高的输入和输出专门化。
- Task Expert：输入专门化可能较低，但在高 \(k\) 时输出专门化明显上升。
Figure 7 显示，深层 Expert 的 Functional Specialization 在高粒度聚类下明显增强，支持深层 Expert 主要承担细粒度任务的观点。