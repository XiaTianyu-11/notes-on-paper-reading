文章链接：https://proceedings.iclr.cc/paper_files/paper/2026/hash/6a7a2238da2b755e6dc1dbe278c60c2a-Abstract-Conference.html

## 问题

MoE 模型虽然只需要激活少量 Experts 就能完成每个 token 的计算，但部署时仍然面临显存容量限制。Expert Offloading 通过将部分 Experts 缓存在 GPU 显存中，其余 Experts 放在 CPU 或磁盘中，降低模型的显存占用。

但这种方法存在一个问题：如果连续生成的 token 不断切换所需的 Experts，就会频繁发生 Cache Miss，需要从 CPU 或磁盘加载专家，降低推理速度。

已有研究发现，MoE 在连续生成 token 时可能具有局部路由一致性（Local Routing Consistency），即相邻 token 倾向于激活相同或相近的 Experts。但不同 MoE 模型的这种性质存在明显差异，不能认为所有 MoE 都适合采用相同的专家缓存策略。

因此，本文主要研究三个问题：
1. 如何衡量局部路由一致性？ 如何判断一个 MoE 模型能否用一组相对固定的 Experts 覆盖连续一段 token 的计算需求？
2. 哪些因素影响局部路由一致性？ 例如负载均衡损失、Shared Experts、专家数量、每个 token 激活的专家数量，以及专家的领域专门化和词汇专门化。
3. 如何根据模型的局部路由一致性确定缓存容量？ 缓存多少 Experts 才能在缓存命中率和显存占用之间取得较好的平衡？
## 总结

提出 Local Routing Consistency 的系统性分析框架，并设计两个指标：
- SRP（Segment Routing Best Performance）：通过模拟一个分段 Router，衡量固定一组专家覆盖连续 token 路由需求的最佳 F1，反映模型内在的局部路由一致性。
- SCH（Segment Cache Best Hit Rate）：通过模拟具有固定容量、能够利用未来一段 token 激活频率进行专家淘汰的缓存，衡量其最佳命中率，更直接地反映 Expert Offloading 的潜在效率。
作者分析了 20 个不同规模和架构的 MoE 模型，并训练了一系列 Toy Models 验证架构与训练因素的影响。

主要发现：
1. 局部负载均衡与局部路由一致性存在明显权衡，但全局负载均衡与局部路由一致性可以同时实现。
2. Shared Experts 会降低局部路由一致性；限制专家组合空间也会产生类似影响。
3. 领域专门化（Domain Specialization）比词汇专门化（Vocabulary Specialization）对局部路由一致性的贡献更大。
4. 对于多数模型，缓存容量设置为每个 token 激活专家数量的约 2 倍，可以较好地平衡缓存效果与存储开销。
本文的核心价值是：将 MoE 的专家路由行为与实际推理部署效率联系起来，为设计更适合 Expert Offloading 的 MoE 架构提供了分析依据。


## 思路

作者的整体思路是：先建立局部路由一致性的量化指标，再分析不同 MoE 模型的差异，随后通过控制变量实验寻找影响因素，最后将这些发现与真实专家缓存效果联系起来。

1. 首先，建立局部路由一致性的评价体系
作者认为，不能仅凭相邻 token 是否激活相同专家来判断模型是否适合 Expert Offloading，因为不同模型可能具有不同程度的局部路由连续性。
因此提出 SRP 和 SCH 两个指标：
- SRP：考察连续一段 token 能否由固定的专家集合覆盖。
- SCH：考察在缓存容量限制下，能否通过提前预测未来专家需求提高缓存命中率。

2. 对真实 MoE 模型进行比较，寻找影响因素
作者选取 20 个 MoE 模型，规模从 3B 到 57B，涵盖不同架构。
数据集由两部分组成：
- 通用语料：RedPajama 的 C4、CommonCrawl、Books、Wikipedia、ArXiv、StackExchange、GitHub。
- 下游任务：LMArena、OpenMath、OpenCode、OpenScience。
总计 22,528 个输入样本，每个样本包含 512 个 token。
对每个 MoE Layer 记录所有 token 的 Router 决策，然后计算不同窗口长度下的 SRP。
作者发现：
- 短期局部一致性：窗口长度 \(m=4\)，大部分模型表现相近。
- 长期局部一致性：窗口长度 \(m\geq16\)，模型之间的差异明显扩大。
这说明很多模型虽然能够在相邻几个 token 之间复用专家，但未必能够在更长的 token 序列中维持这种性质。
作者进一步将模型划分为四组：
- Group 1：长期 SRP 最高，且 \(\hat\rho\) 约为 1.25，具有较强的长期路由一致性。
- Group 2：SRP 次高，但长期 \(\hat\rho\) 约为 2.5，需要激活更多专家才能覆盖路由需求。
- Group 3：SRP 明显下降，长期 SRP 约为 0.36。
- Group 4：SRP 最低，长期 SRP 低于 0.31。

3. 通过 Toy Models 控制变量，研究架构因素的影响
仅仅比较真实模型存在一个问题：不同模型的架构、训练数据和训练目标都不相同，很难判断究竟是什么因素导致了 SRP 的差异。
因此作者以 OLMoE 为基础，从头训练一系列 Toy Models。
基准模型具有 1.43B 参数，其他模型保持总参数量不变，分别调整专家数量、激活专家数量、Shared Experts 数量、负载均衡损失系数以及 Dense Layer 的配置。
主要研究以下因素：
（1）Load Balance 与 Local Routing Consistency 的权衡
作者通过计算不同 Experts 激活频率的标准差（SD），衡量专家负载均衡程度。
发现：
- NoLB：不施加负载均衡约束，SRP 较高，但专家负载非常不均衡。
- OverLB：进一步强化负载均衡，SRP 明显下降。
这说明强制 Router 将 token 均匀分配给所有 Experts，可能破坏连续 token 对同一批 Experts 的重复使用。
但作者还发现，Qwen3 和 GRIN-MoE 能够同时获得较高 SRP 和相对合理的负载均衡。
原因在于领域专门化：不同领域的输入可以激活不同的专家集合。一个领域内部可以反复使用少数 Experts，而不同领域之间又可以使用不同 Experts，最终使整个数据集上的专家负载保持相对均衡。
因此，局部路由一致性与全局负载均衡并不矛盾。
（2）Shared Experts 与 Expert Combination Space
作者发现 Shared Experts 会降低 SRP，并提出两种可能的解释：
- Bypass Effect：更多信息由 Shared Experts 处理，使普通 Experts 所承担的路由任务变得不那么重要。
- Expert Combination Space 减小：Shared Experts 的存在减少了实际参与路由的 Experts 数量，从而减少 Router 可以选择的专家组合。
作者通过 Toy Models 进一步验证专家组合空间的影响：
- 32Exp：减少 Experts 总数，SRP 下降。
- Top2：减少每个 token 激活的 Experts 数量，SRP 下降。
- Top16：增加每个 token 激活的 Experts 数量，但仍少于专家总数，SRP 上升。
这说明更大的专家组合空间有助于提高局部路由一致性。
（3）MoE Layer 与 Dense Layer 的交替配置
作者还研究了 MoE Layer 与 Dense Layer 的交替方式。
例如：
- Skip1：第一层使用 Dense FFN。
- Sparse2：每隔一层使用 MoE。
结果发现，这些配置对 SRP 的影响相对较小，而且部分模型的困惑度明显升高。
因此，Dense Layer 的交替配置并不是影响局部路由一致性的主要因素。

4. 研究专家专门化与局部路由一致性的关系
第三节发现，部分模型能够同时实现较高的 SRP 和较好的全局负载均衡。
作者进一步提出一个问题：
为什么某些模型能够在连续 token 中重复使用一批 Experts，同时又在不同输入之间切换 Experts？
于是第四节研究 Experts 的专门化。
作者分别考虑两种专门化：
- Domain Specialization：专家倾向于处理某些特定领域的数据。
- Vocabulary Specialization：专家倾向于处理某些特定 Token ID。
首先，作者分别计算不同领域数据上的 SRP，发现不同模型呈现三种模式：
1. Phi-3.5-MoE、GRIN-MoE、OLMoE：在 ArXiv、StackExchange、GitHub、OpenMath 和 OpenCode 等领域具有更高 SRP，说明它们可能存在数学、代码、学术领域专门化的 Experts。
2. LLaMA-MoE-v2、Yuan2.0、Qwen3：在 Wikipedia 等通用领域具有更高 SRP，而在数学、代码领域相对较低，说明它们可能具有通用文本或多语言专门化的 Experts。
3. Mixtral、MiniCPM-MoE、JetMoE：不同领域的 SRP 差异不明显，而且整体 SRP 较低。
接着，作者分析专家专门化程度与 SRP 的相关性，发现：
- Domain Specialization 与 SRP 通常呈正相关。
- Vocabulary Specialization 与 SRP 的相关性较弱。
这说明领域专门化比词汇专门化更能解释模型的局部路由一致性。
5. 使用 SCH 验证局部路由一致性对真实缓存效果的影响
前面主要使用 SRP 分析路由行为，但 SRP 没有考虑缓存容量限制。
因此第五节使用 SCH，对不同缓存容量下的专家缓存命中率进行模拟。
作者发现：
- Group 1 模型在缓存容量较小时，SCH 随缓存容量增加而快速上升。
- 当缓存比例 \(\rho\) 接近 2 时，SCH 出现明显的增长拐点。
- Group 3、Group 4 的 SCH 增长相对缓慢，需要更大的缓存才能达到相近的命中率。
作者进一步比较 SCH 与真实缓存算法 LRU、LFU 的命中率，发现两者与 SCH 高度相关。

最终，作者认为对于大多数 MoE 模型，缓存容量设置为每个 token 激活专家数量的约 2 倍，可以较好地平衡缓存效果和显存开销。

## 使用的方法和技巧

### 1. Local Routing Consistency：局部路由一致性

研究连续 token 是否倾向于激活相同或相近的 Experts。

核心目的不是研究专家之间的功能相似性，而是研究 Router 在连续 token 上的专家选择是否具有重复性。

### 2. Segment Routing Best Performance（SRP）

使用分段预测模拟固定专家集合对连续 token 路由需求的覆盖能力。

对于单个 Expert：

\[
SRP(e,m)=\max_{R_e^m}F1(R_e^m)
\]

其中 \(m\) 是窗口长度，\(R_e^m\) 是分段预测器。

通过统计窗口内 Expert 的激活次数，并寻找最优阈值 \(\alpha_{e,m}\)，确定哪些窗口应该预测为激活。

SRP 越高，说明固定的专家激活模式越能覆盖连续 token 的真实路由需求。

### 3. Segment Routing Size Ratio（\(\hat\rho\)）

衡量 Segment Router 为了获得较高 F1，需要额外激活多少 Experts。

\[
\hat\rho=
\frac{\text{Segment Router 平均激活专家数}}
{\text{Original Router 平均激活专家数}}
\]

SRP 高且 \(\hat\rho\) 低，意味着可以用较少的额外专家覆盖连续 token 的计算需求。

### 4. Segment Cache Best Hit Rate（SCH）

通过模拟具有固定容量的专家缓存，衡量局部路由一致性对实际缓存效率的影响。

\[
\rho=
\frac{\text{Cache Size}}
{\text{Top-k 激活专家数量}}
\]

当 Cache Miss 时，加载当前需要的 Experts，并淘汰未来 \(m\) 个 token 中激活次数最少的专家。

\[
SCH=
\frac{\text{Cache Hit 数量}}
{\text{总专家请求数量}}
\]

SCH 越高，说明该模型越适合利用局部路由规律进行专家缓存。

### 5. Toy Model Controlled Experiments：控制变量实验

以 OLMoE-like 模型为基础，保持模型总参数量不变，分别改变架构和训练参数：
- Load Balance：改变负载均衡损失系数。
- Shared Experts：改变共享专家数量。
- Expert Combination Space：改变专家总数和 Top-k 数量。
- Dense/MoE Arrangement：改变 Dense Layer 与 MoE Layer 的排列方式。
通过比较不同模型的 SRP 和困惑度，分析各个因素对局部路由一致性的影响。

### 6. Domain / Vocabulary Specialization：专家专门化分析

使用专家在不同领域、不同 Token ID 上的归一化激活频率，衡量其专门化程度。
- Domain Specialization：使用专家在不同领域的激活频率，并通过变异系数（CV）衡量领域专门化程度。
- Vocabulary Specialization：衡量专家对特定 Token ID 的激活偏好，分别考察输入 token、模型预测 token 和 Ground Truth token。
最后计算专门化程度与 SRP 的相关性，判断哪种专门化更有利于局部路由一致性。

### 7. LRU / LFU：验证 SCH 的实际参考价值

将 SCH 与两种常见缓存算法进行比较：
- LRU（Least Recently Used）：优先淘汰最近最少使用的专家。
- LFU（Least Frequently Used）：优先淘汰历史使用频率最低的专家。
通过比较不同窗口长度下 SCH 与真实缓存命中率的相关性，验证局部路由一致性是否能够预测实际专家缓存效果。

与 MoE 记忆机制研究的联系：

这篇文章对 MoE 记忆机制的直接贡献有限，因为它研究的是专家路由的时间局部性和推理缓存效率，而不是知识存储或记忆提取。不过，它提供了一个值得借鉴的视角：专家在某个领域内持续被激活，可能意味着该专家承担了相对稳定的领域处理功能。 因此，在研究领域知识与专家分工、知识在不同专家之间的分布时，可以将 Domain Specialization 与 Local Routing Consistency 结合起来分析，区分专家是稳定承担某类知识处理任务，还是仅仅在局部 token 上偶然共同激活。