文章链接：https://proceedings.neurips.cc/paper_files/paper/2025/hash/5340b0c0b76dc0115f5cc91c20c1251d-Abstract-Conference.html

## 问题

MoE 模型通过 Router 为每个 Token 动态选择 Top-k 个 Expert，从而实现稀疏计算。但在隐私保护推理场景下，即使不泄露 Token 内容，Router 的路由信息仍然可能泄露用户输入的语义。

例如，不同任务可能倾向于激活不同的 Expert。攻击者通过观察各个 Expert 的激活频率和负载分布，就可能推断用户正在执行数学推理还是文本理解，甚至推断不同 Expert 的专业化方向。

因此，隐私保护 MoE 推理面临三个问题：
1. 路由信息泄露： 不同输入会产生不同的 Expert 激活分布，即使不公开 Token 内容，暴露 Expert 的负载和路由结果仍可能泄露输入语义。
2. 隐私与计算效率的矛盾： 如果为了隐藏路由信息，让每个 Token 都经过所有 Expert，虽然能够保护路由隐私，但会破坏 MoE 的稀疏计算优势。例如 QWenMoE 中每个 Token 只激活 4/60 个 Expert，全部计算会产生约 15 倍的 Expert 计算量。
3. 隐私与准确率的矛盾： 如果限制每个 Expert 接收的 Token 数量，就必须丢弃超出限制的 Token，而被丢弃的 Token 可能对最终输出有重要贡献，导致模型准确率下降，直接随机丢弃最多可能造成约 7% 的准确率损失。
## 总结

提出 CryptoMoE，一种面向 MoE 大语言模型的隐私保护推理框架。核心是提出 Inference-Time Balanced Expert Routing（推理时均衡专家路由），在推理阶段限制每个 Expert 接收固定数量的 Token，从而隐藏不同输入产生的 Expert 负载差异，同时保留 MoE 的稀疏计算优势。

针对固定容量导致的 Token 丢弃问题，提出 Confidence-Aware Secure Dispatch，根据 Router 的路由置信度优先保留重要 Token，并设计对应的 Token 分发和结果合并方法。

此外，针对 MoE 中不同 Expert 的矩阵计算开销，提出 Batch MatMul，将不同 Expert 的部分 Token 打包计算，减少矩阵乘法中的重复操作。

实验覆盖 DeepSeekMoE、QWenMoE 和 OLMoE。CryptoMoE 平均保留原始模型 99.2% 的准确率，相比 Dense 隐私推理基线获得 2.8～3.5 倍加速。

## 思路

作者首先发现，MoE 的 Expert 激活分布具有明显的输入依赖性，不同任务可能激活不同的 Expert。因此，即使不公开输入内容，仅暴露 Expert 的负载分布，也可能泄露输入语义。

由此产生一个思路：如果能够让所有 Expert 在推理时始终处理相同数量的 Token，那么外部就无法通过 Expert 的实际负载差异推断输入类型。

因此，提出 Inference-Time Balanced Expert Routing：

对于包含 \(m\) 个 Token、每个 Token 激活 \(k\) 个 Expert、总共有 \(n\) 个 Expert 的 MoE 层，令每个 Expert 固定接收：
\[
t=\frac{mk}{n}
\]

个 Token。

具体分为两种情况：
- 如果某个 Expert 接收的 Token 少于 \(t\)，就补充 Dummy Token。
- 如果某个 Expert 接收的 Token 多于 \(t\)，就丢弃超出部分。
这样，每个 Expert 的计算规模固定为 \(t\)，同时所有 Expert 的总计算 Token 数仍然是：
\[
nt=mk
\]

因此，能够在维持原有稀疏计算规模的同时隐藏 Expert 的实际负载差异。

但是，这种方法会丢弃部分真实 Token，导致准确率下降。作者进一步提出 Confidence-Aware Selection：不再随机选择保留的 Token，而是按照 Router 对该 Expert 的路由概率排序，优先保留置信度最高的 \(t\) 个 Token。

在此基础上，作者进一步解决 Token 分发和结果恢复问题：
- Dispatch： 对每个 Expert 分别计算候选 Token 的优先级，选出 Top-t，再根据索引提取对应的 Token Embedding。
- Combine： 复用 Dispatch 阶段产生的 Token 索引，将 Expert 输出恢复到原始 Token 的位置，并乘回对应的 Router 权重，最后汇总所有 Expert 的输出。
最后，作者发现不同 Expert 的 FFN 虽然使用不同的权重矩阵，但计算结构相同。因此提出 Batch MatMul，将不同 Expert 的部分 Token Embedding 打包到一起进行矩阵计算，减少同态加密计算中的旋转操作。

## 使用方法和技巧

### 1. Inference-Time Balanced Expert Routing：推理时均衡专家路由

通过固定每个 Expert 接收的 Token 数量，隐藏原始动态路由产生的负载差异。
\[
t=\frac{mk}{n}
\]

- 不足 \(t\)：使用 Dummy Token 补齐。
- 超过 \(t\)：丢弃多余 Token。
关键技巧：在推理阶段改变 Token 的实际执行分配方式，而不需要重新训练 Router 或 Expert。

### 2. Confidence-Aware Selection：基于路由置信度的 Token 筛选

当某个 Expert 接收的 Token 数量超过 \(t\) 时，不采用随机丢弃，而是根据 Router 对该 Expert 的路由概率进行排序，保留 Top-t。
\[
X_i=\operatorname{TopT}\big(\{(x_j,W_{j,i})\mid i\in K_j\}\big)
\]

其中 \(W_{j,i}\) 表示 Token \(x_j\) 被分配给 Expert \(i\) 的路由概率。

核心技巧：利用 Router 已经计算出的置信度作为 Token 重要性的代理指标，尽可能减少固定容量造成的信息损失。

### 3. Index-based Token Selection：基于索引的 Token 筛选

将 Token 的优先级分数与 Token Embedding 分离。

具体过程：
1. 根据 Token 是否被分配给当前 Expert，构造布尔掩码。
2. 将掩码与 Router 概率相乘，得到 Token 优先级分数。
3. 对分数执行 Top-t，得到需要保留的 Token 索引。
4. 将展开后的索引映射回原始 Token 索引。
5. 利用 One-Hot 矩阵提取对应的 Token Embedding。
核心技巧：先对低维分数和索引进行排序，再提取高维 Embedding，避免直接对高维 Token 表示进行排序和交换。

### 4. One-Hot-based Token Reordering：基于 One-Hot 的 Token 顺序恢复

由于每个 Expert 接收到的 Token 经过筛选和重新排列，计算结束后需要恢复原始 Token 顺序。

利用 Dispatch 阶段产生的 One-Hot 矩阵：
\[
O_i=\operatorname{onehot}(K_i',m)
\]

通过转置恢复原始 Token 位置：
\[
Y_i'=O_i^TY_{E_i}
\]

进一步将 One-Hot 矩阵与对应的 Router 权重结合：
\[
Y_i'=R_iY_{E_i}
\]

最后汇总所有 Expert：
\[
Y=\sum_{i=0}^{n-1}Y_i'
\]

核心技巧：复用 Token 筛选阶段产生的索引信息，同时完成 Token 顺序恢复和 Router 权重加权，避免重复计算。

### 5. Batch MatMul：跨 Expert 批量矩阵乘法

针对不同 Expert 分别执行矩阵乘法时，需要大量重复的密文旋转操作，作者提出将不同 Expert 的部分 Token Embedding 打包到同一个密文中，并相应调整明文权重矩阵。

将 HE 旋转复杂度从：
\[
O(nd_1)
\]

降低到：
\[
O(d_1)
\]

其中 \(n\) 为 Expert 数量，\(d_1\) 为 Expert 输入的隐藏维度。

核心技巧：利用不同 Expert 的矩阵计算结构相同这一特点，将原本分散的 Expert 计算转换为批量计算，减少重复的计算操作。

本文最值得借鉴的地方： 通过控制推理阶段的 Expert 输入规模来保护路由隐私，而不是简单地取消 MoE 稀疏性；同时利用路由置信度、索引筛选和跨 Expert 批量计算，尽可能弥补均衡路由带来的准确率和效率损失。