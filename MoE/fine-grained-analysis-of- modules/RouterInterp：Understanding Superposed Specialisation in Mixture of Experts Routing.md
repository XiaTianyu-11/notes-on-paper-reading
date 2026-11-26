文章链接：http://arxiv.org/abs/2610.11775

## 问题

MoE 模型通常认为，不同 Expert 会专门处理不同领域的数据，例如某些 Expert 负责医学知识，某些负责数学推理，某些负责代码生成。这种观点称为 Domain Specialisation Hypothesis（DSH，领域专业化假设）。

但已有研究发现，很难通过 token 统计等方法找到清晰、稳定的 Expert 领域分工。

作者认为，问题可能不在于 Expert 没有专业化，而在于我们对专业化的理解存在偏差。

一个 MoE 层通常只有几十个 Expert，但输入数据包含大量细粒度的语义、语法和计算特征（Micro-domains）。因此，多个 Micro-domains 必然需要由同一个 Expert 处理。

这就产生了两种可能：
1. 领域专业化（DSH）：语义相近的 Micro-domains 被分配给同一个 Expert，最终形成一个语义连贯的领域。例如，一个 Expert 同时负责疾病、解剖学和医疗程序。
2. 叠加专业化（SSH）：语义不相关的 Micro-domains 被分配给同一个 Expert。例如，一个 Expert 同时负责数学运算、C++ 预处理指令和医学术语。
作者认为，过去难以解释 Expert 的路由行为，是因为研究者默认采用 DSH，而实际 MoE 更符合 SSH。

因此，本文要解决两个问题：
1. 如何理解 Expert 的真实专业化方式？ 如果一个 Expert 同时处理多个不相关的 Micro-domains，就不能再用一个宏观领域概括它。
2. 如何解释 Router 的决策？ 传统 token 共现统计无法充分捕获细粒度的路由特征，需要找到真正影响 Expert 选择的内部特征。

## 总结

提出 SSH（Superposed Specialisation Hypothesis，叠加专业化假设），重新解释 MoE 中 Expert 的专业化方式，并提出 RouterInterp，一种基于 SAE feature 的 Router 可解释性方法。

核心观点是：一个 Expert 不一定负责某个语义连贯的领域，而可能负责多个语义不相关的细粒度特征。

作者从理论和实验两方面支持 SSH：
1. 理论上，语义不相关、很少同时出现的特征可以共享同一个 Expert，减少计算干扰；Load Balancing 也可能促使不同领域的输入被分配到同一个 Expert。
2. 实验上，通过 SAE 提取每个 Expert 最相关的 feature，再对这些 feature 进行语义聚类，发现同一个 Expert 对应的 feature 往往分布在多个不同的语义簇中。
在此基础上提出 RouterInterp，通过识别影响 Expert 选择的 SAE feature、收集正负激活样本，并使用 LLM 综合生成自然语言解释。

实验结果：
- gpt-oss-20b：平均解释 F1 为 0.492，相比传统 Unigram Lookup 的 0.299 提升约 65%。
- OLMoE-1B-7B：平均解释 F1 为 0.602，相比 Unigram Lookup 的 0.342 提升约 76%。
消融实验进一步表明，提升主要来自 SAE feature decomposition，而不只是增加文本上下文或使用 LLM 总结。

因此，本文的核心贡献是：证明 MoE Expert 的路由专业化可能是多种异质特征的组合，而非单一领域的专业化，并提供了一套能够解释这种复杂路由行为的方法。


## 思路

作者的研究思路可以分为三个阶段。

第一阶段：提出 SSH，解释为什么 Expert 会同时负责多个不相关领域。
作者首先发现，真实数据中的 Micro-domains 数量远远超过 MoE 每层的 Expert 数量，因此多个 Micro-domains 必然需要分配给同一个 Expert。
关键问题是：这些 Micro-domains 最终会如何组合？
作者提出两种假设：
- DSH：将语义相近的 Micro-domains 分配给同一个 Expert。
- SSH：将语义不相关的 Micro-domains 分配给同一个 Expert。
随后给出两个理论解释：
1. Interference Minimisation（干扰最小化）：语义不相关的特征通常较少同时激活，因此可以共享同一个 Expert，减少不同特征之间的干扰。一个 Expert 可以根据输入特征执行不同的计算变换，即 Computation in Superposition。
2. Load Balancing（负载均衡）：训练时的负载均衡损失鼓励不同 Expert 均匀接收 token。当 batch 较小时，其中可能只有少数几个宏观领域的数据，Router 为了均衡负载，可能将同一个宏观领域的 token 分配给不同 Expert，从而打破单一领域与 Expert 的对应关系。

第二阶段：通过 SAE 验证 SSH。
作者使用 SAE 提取隐藏激活中的稀疏 feature，将这些 feature 作为 Micro-domains 的代理。
对于每个 Expert，找到最能预测其激活的 top-20 个 SAE feature，然后对这些 feature 的自然语言解释进行语义聚类。
定义：

\[
G(E_i)=\text{Expert }E_i\text{ 的 feature 所覆盖的语义簇数量}
\]

两种假设的预测不同：
- DSH：\(G(E_i)=1\)，因为一个 Expert 的 feature 应该属于同一个语义领域。
- SSH：\(G(E_i)\) 应接近 feature 数量，因为不同 feature 对应不同的语义领域。

实验发现：
|模型|	平均语义簇数量|
|---|---|
|OLMoE-1B-7B（Layer 15）|	10.8|
|gpt-oss-20b（Layer 20）|	11.3|

每个 Expert 使用 20 个 feature，随机分配时平均约为 16.6 个簇。
因此，实际结果远高于 DSH 预测的 1，但又低于随机分配。这说明 Expert 的确覆盖多个语义不同的 Micro-domains，同时这些 feature 之间仍保留一定的主题关联。

第三阶段：提出 RouterInterp，解释每个 Expert 的路由逻辑。
既然一个 Expert 对应多个不同的 Micro-domains，就不能再简单统计激活它的 token，然后让 LLM 总结一个统一主题。
因此，作者采用 SAE feature 作为组织单位：
1. 找出对 Expert routing 最有贡献的 feature。
2. 对每个 feature 分别收集激活且 Expert 被选择、激活但 Expert 未被选择的上下文。
3. 将每个 feature 的正负样本分别交给 LLM。
4. 让 LLM 综合不同 feature 的 routing 条件，形成完整解释。
最终得到的不是一个单一领域标签，而是多个不同条件共同组成的 Expert routing 描述。

## 使用方法和技巧

### 1. Superposed Specialisation Hypothesis：叠加专业化假设

区分 DSH 和 SSH：

|对比	|DSH	|SSH|
|---|---|---|
|Expert 专业化方式|	单一语义连贯领域	|多个异质 Micro-domains|
|Feature 关系	|相互接近|	语义上相互分散|
|Expert 解释	|一个主要概念|	多个不同概念的组合|

SSH 的核心依据是两个理论机制：
- 特征之间的低共激活频率可以降低共享 Expert 时的干扰。
- Load Balancing 可能促使不同输入领域共同使用同一个 Expert。

### 2. SAE Feature：利用稀疏特征识别 Micro-domains

将隐藏激活 \(x\) 输入 SAE：

\[
z=\operatorname{TopK}
\left(W_{\mathrm{enc}}(x-b_{\mathrm{pre}})+b_{\mathrm{enc}},s\right)
\]

其中：
- \(x\)：模型隐藏激活。
- \(z\)：SAE 提取的稀疏 feature 激活。
- \(s\)：每个 token 保留的 feature 数量。
- \(W_{\mathrm{enc}}\)：Encoder。
- \(W_{\mathrm{dec}}\)：Decoder。

通过 SAE 将原本混合在隐藏空间中的特征拆分出来，利用不同 feature 的语义差异分析 Expert 的专业化方式。

### 3. Gradient-based Attribution：寻找影响 Router 选择的 Feature

作者不是简单按照 feature 激活频率排序，而是计算移除某个 feature 对 Expert routing 的近似影响。

首先定义 Router 的 routing margin：

\[
m_i(x)=h_i(x)-\tau_i(x)
\]

其中：
- \(h_i(x)\)：Expert \(E_i\) 的 Router logit。
- \(\tau_i(x)\)：其他 Expert 中第 \(k\) 大的 logit，即目标 Expert 进入 Top-k 所需超过的门槛。

接着计算 feature \(f\) 对 routing margin 的贡献：

\[
c_{if}(x)
=
z_f d_f^\top\nabla_xm_i(x)
\]

其中 \(z_f\) 是 feature 激活值，\(d_f\) 是 SAE Decoder 中对应的方向。

\(c_{if}>0\) 表示该 feature 支持 Expert \(E_i\) 被选中。

进一步定义 feature 的区分性评分：

\[
\boxed{
S_{if}=
\mathbb E[c_{if}^{+}\mid E_i\in T(x)]
-
\mathbb E[c_{if}^{+}\mid E_i\notin T(x)]
}
\]

其中：

\[
c_{if}^{+}=\max(c_{if},0)
\]

这个指标不仅考虑 feature 是否支持 Expert 被选中，还考虑这种支持是否具有区分性。

最后按照 \(S_{if}\) 排序，为每个 Expert 选择 top-\(n\) 个 feature，并保证每个 feature 最多分配给一个 Expert。

### 4. Positive-Negative Contrast：通过正负样本识别路由条件

对于每个选中的 feature \(f\)，分别构造：

\[
P_{i,f}=
\{x:f\text{ 激活且 }E_i\text{ 被选择}\}
\]

\[
N_{i,f}=
\{x:f\text{ 激活但 }E_i\text{ 未被选择}\}
\]

关键技巧是：正负样本都必须激活同一个 feature。

这样可以排除单纯由 feature 是否出现造成的差异，让 LLM 重点分析：
- feature 出现时，什么上下文会使目标 Expert 被选中？
- 同一个 feature 出现时，什么上下文又会使目标 Expert 不被选中？
最终将不同 feature 的解释合并，得到一个包含多个 Micro-domains 的 Expert routing 描述。

### 5. Explanation Detection：使用 F1 评价解释质量

将生成的自然语言解释交给另一个 LLM（Scorer），让它判断 held-out 文本窗口是否符合解释。

测试集构造：
- Positive：至少有一个 token 被路由到目标 Expert。
- Negative：没有任何 token 被路由到目标 Expert。
Scorer 根据解释判断高亮 token 周围的上下文是否符合目标 Expert 的激活条件，得到二分类结果。

最后计算：

\[
F1=\frac{2PR}{P+R}
\]

其中 \(P\) 为 Precision，\(R\) 为 Recall。

作者还通过人工评估验证 LLM Scorer 的可靠性。

### 6. Feature Selection Ablation：验证特征选择策略

作者比较了三种特征选择方式：
1. Gradient-based Attribution：根据 feature 对 routing margin 的贡献进行排序。
2. \(\rho\)-usefulness：根据 feature 在 Expert 被选中和未被选中时的平均激活差异进行排序。
3. Cosine Similarity：根据 SAE Decoder 方向与 Router 权重向量之间的余弦相似度进行排序。
实验发现，Gradient-based Attribution 在不同 feature 数量下整体表现最好。

此外，作者选择每个 Expert 的 top-45 个 feature 作为最终解释输入，因为继续增加 feature 虽然能够提高解释质量，但会使解释长度不断增长，边际收益逐渐下降。