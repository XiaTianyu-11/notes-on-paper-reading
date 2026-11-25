文章链接：https://proceedings.iclr.cc/paper_files/paper/2025/hash/854a9ab0f323b841955e70ca383b27d1-Abstract-Conference.html

## 问题

MoE 的核心是 Router 根据 Token 的隐藏表示选择 Top-k 个 Expert。传统 Router 通常通过 Token 与 Expert embedding 的点积计算匹配分数：
\[
s_k=h^\top e_k
\]

这种方式隐含地假设所有隐藏维度对于 Expert 选择具有相同的重要性。但实际上，不同 Expert 对应的 Token Cluster 可能只在部分隐藏维度上具有明显的聚集特征，而其他维度可能存在较大的离散性。

这会产生三个问题：
1. Expert 专业化不足： 如果不同 Cluster 的关键特征被大量无关维度掩盖，Router 就难以识别真正的语义区域，容易将不同类型的 Token 分配给同一个 Expert，使 Expert 接收到的数据缺乏一致性。
2. 路由收敛较慢： 高维空间中的 Cluster 可能存在重叠，Router 难以快速学习正确的 Token-Expert 对应关系，导致 Expert 难以形成专业化分工。
3. 路由鲁棒性较差： 当 Token 表示受到噪声或数据污染时，原本不重要的维度可能干扰 Router 的判断，使 Token 被错误分配给不适合的 Expert，进而影响模型表现。
因此，作者希望解决的问题是：如何识别每个 Expert 所对应的 Token Cluster 中真正具有区分作用的隐藏维度，并利用这些维度改善 Router 的 Token-Expert 匹配？

## 总结

提出 AC Router（Adaptive Clustering Router）及对应的 ACMoE，通过特征加权聚类优化，为每个 Expert 动态计算一组 Cluster-specific feature weights，根据该 Expert 接收到的 Token 在不同隐藏维度上的离散程度，放大 Cluster 内部较为紧凑的维度，压低较为分散的维度。

然后将这些权重融入后续层的 Router，通过改变 Token 与 Expert embedding 的匹配空间，使不同 Cluster 在关键特征维度上更加容易区分，从而改善 Expert 专业化。

该方法具有三个主要优势：
1. 更强的 Expert 专业化： 使 Router 更容易识别不同语义区域，将相似 Token 分配给适合的 Expert。
2. 更快的收敛速度： 通过改善 Router 优化过程中的 Hessian 条件数，使路由学习更加稳定。
3. 更好的鲁棒性： 通过扩大不同 Cluster 在变换空间中的距离，降低噪声导致错误路由的概率。
ACMoE 不引入额外的可学习参数，也不改变 Expert FFN 的结构，而是通过动态统计 Token Cluster 的特征分布调整 Router 的计算方式。作者在 Switch Transformer、GLaM 和 Swin Transformer 上验证了方法在语言建模、图像分类、数据污染和 OOD 场景下的效果。


## 思路

作者首先从特征加权聚类的角度重新理解 MoE：Router 的本质是将隐藏表示划分成不同 Cluster，并将每个 Cluster 分配给对应 Expert。因此，可以借鉴聚类优化中的特征加权思想，寻找每个 Cluster 最具有识别作用的隐藏维度。

由此产生的核心思路是：
\[
\text{Cluster 内部越紧凑的维度}
\Rightarrow
\text{越有助于识别该 Cluster}
\Rightarrow
\text{应该赋予越大的权重}
\]

具体分为三个阶段：

1. Feature-weighted Clustering：推导最优特征权重
作者首先将 MoE Router 看作一个聚类分类器，定义隐藏表示之间的加权距离：
\[
D_{ij}(w)=\sum_{q=1}^{d}w_q\rho_{ijq}
\]
其中 \(w_q\) 表示第 \(q\) 个隐藏维度对距离计算的重要程度。
但不同 Cluster 可能依赖不同的特征组合，因此不能对所有 Cluster 使用同一组权重。作者进一步引入 Cluster-specific feature weights：
\[
w_k=(w_{1k},\ldots,w_{dk})
\]
通过最小化 Cluster 内部的加权距离，寻找能够使每个 Cluster 尽可能紧凑的权重。
为了避免所有权重集中到单一维度，作者使用 KL 散度约束权重分布与均匀分布之间的差异。
最终推导得到：
\[
\boxed{
w_{qk}=\frac{\lambda/d}{s_{qk}+\alpha_k}
}
\]
其中：
- \(s_{qk}\)：Cluster \(k\) 在第 \(q\) 个维度上的离散程度。
- \(\lambda\)：正则化强度。
- \(\alpha_k\)：保证权重归一化的常数。
因此：
\[
s_{qk}\downarrow
\Rightarrow
w_{qk}\uparrow
\]
即 Cluster 在某个维度上越紧凑，该维度的权重就越大。

2. Adaptive Clustering Router：将特征权重融入 Router
得到特征权重后，作者进一步将其转换成对角矩阵：
\[
M_k^\ell=
\operatorname{diag}
\left(
\frac1{s_{1k}^\ell},
\ldots,
\frac1{s_{dk}^\ell}
\right)
\]
其中 \(s_{qk}^\ell\) 是第 \(\ell\) 层 Expert \(k\) 接收到的 Token 在第 \(q\) 个维度上的离散程度。
实际实现时，作者使用 MAD（Mean Absolute Deviation，平均绝对偏差）估计离散程度：
\[
s_{qk}^\ell=
\frac1{N_k}
\sum_{i\in C_k^\ell}
|h_{iq}^\ell-\bar h_{qk}^\ell|
\]
其中 \(C_k^\ell\) 是当前层被 Router 分配给 Expert \(k\) 的 Token 集合。
随后将变换矩阵融入下一层 Router：
\[
\boxed{
s_k^\ell=
h^{\ell\top}
M_{k^*}^{\ell-1}
e_k^\ell
}
\]
其中：
- \(k^*\)：Token 在上一层被分配到的 Expert。
- \(M_{k^*}^{\ell-1}\)：上一层对应 Expert 的特征权重矩阵。
- \(e_k^\ell\)：当前层候选 Expert 的 embedding。
- \(k\)：当前层正在参与竞争的候选 Expert。
最后仍然通过 Top-k 选择 Expert，并使用原有的 Gating 和 FFN 完成计算。
这里有一个关键设计：作者并不是提前对 Token 进行语义聚类，而是先使用 Router 产生初始 Expert assignment，再根据已经形成的分组统计特征分布，利用上一层的 Cluster 信息指导下一层重新路由。
由于第一层没有上一层的 Expert assignment，因此 AC Router 从第二层开始使用。
此外，作者对变换矩阵中的缩放系数进行均值归一化，使其平均值为 1，避免不同 Cluster 的整体尺度差异影响 Router logits。

3. 利用 Cluster 几何改善鲁棒性和收敛速度
作者从两个方面分析 ACMoE 的理论优势。
- Robustness： 首先证明 AC 变换可以增加不同 Cluster 在变换空间中的中心距离。随后在 Gaussian Mixture Model（GMM）假设下，推导噪声导致错误路由的概率，说明 Cluster 间距离越大，Token 被错误分配给邻近 Expert 的概率越低。因此，AC Router 通过突出 Cluster 内部紧凑且具有区分作用的特征，改善路由鲁棒性。
- Faster Convergence： 作者从优化角度分析 Router 的 Hessian 条件数。AC 变换相当于对 Cluster 的不同维度进行自适应缩放，使原本细长、分布不均匀的 Cluster 更接近球形（Sphering）。这能够改善 Router embedding 的 Hessian 条件数，使优化过程更加稳定，进而加快路由收敛。
最后，作者将 AC Router 集成到 Switch、GLaM 和 Swin 等 MoE 架构中，在训练和推理阶段都使用该机制，使 Router 和 Expert 在新的路由方式下共同适应，而不是仅在训练完成的 MoE 上替换推理阶段的路由规则。


## 使用方法和技巧

### Feature-weighted Clustering：基于特征离散程度的聚类加权

将不同 Expert 接收到的 Token 视为不同 Cluster，通过计算各 Cluster 在不同隐藏维度上的离散程度，判断哪些维度更有助于识别该 Cluster。

核心原则：

\[
w_{qk}\propto\frac1{s_{qk}}
\]

Cluster 在某个维度上越紧凑，该维度对识别 Cluster 的贡献越大，因此赋予更高权重。

### MAD（Mean Absolute Deviation）：估计 Cluster 的特征离散程度

使用平均绝对偏差衡量每个 Expert 接收到的 Token 在不同维度上的分散程度：

\[
s_{qk}=
\frac1{N_k}
\sum_{i\in C_k}
|h_{iq}-\bar h_{qk}|
\]

相较于方差，MAD 对离群点更加不敏感。论文实验也验证了 MAD 优于直接使用 Variance 作为离散程度指标。

### Adaptive Clustering Transformation：构造 Cluster-specific 特征变换矩阵

根据各维度的 MAD 构造对角矩阵：

\[
M_k=
\operatorname{diag}
\left(
1/s_{1k},\ldots,1/s_{dk}
\right)
\]

不同 Expert 拥有不同的变换矩阵，因此可以根据各自的专业化特征调整隐藏维度的重要程度。

实际实现中，对缩放系数进行均值归一化，使平均权重为 1，避免整体尺度变化干扰路由。

### Cross-layer Adaptive Routing：利用前一层的 Expert assignment 指导下一层路由

这是 ACMoE 最核心的实现技巧。

\[
s_k^\ell=
h^{\ell\top}M_{k^*}^{\ell-1}e_k^\ell
\]

利用上一层的 Expert assignment 确定当前 Token 应采用哪一组特征权重，再与当前层所有候选 Expert 计算匹配分数。

需要注意，上一层的 Expert assignment 只是提供特征度量依据，并不固定当前层的 Expert 选择。当前层仍然独立执行 Top-k 路由。

因此，模型形成一种跨层反馈机制：

\[
\text{上一层路由}
\rightarrow
\text{Cluster 特征统计}
\rightarrow
\text{下一层路由}
\rightarrow
\text{新的 Expert 分工}
\]

### GMM-based Robustness Analysis：基于高斯混合模型分析路由鲁棒性

将不同 Expert 对应的 Token Cluster 建模为 Gaussian Mixture Model：

\[
g_k=\mathcal N(\mu_k,\Sigma_k)
\]

假设 Token 隐藏表示受到零均值噪声污染：

\[
h'=h+\epsilon
\]

通过分析变换前后的 Cluster 中心距离，推导错误路由概率：

\[
P(\text{misrouting})
=
1-\Phi\left(
\frac{\|\delta\mu\|^2}
{2\sqrt{
\delta\mu^\top
(\Sigma_{k^*}+\Sigma_\epsilon)
\delta\mu
}}
\right)
\]

其中 \(\delta\mu\) 表示正确 Cluster 与最近错误 Cluster 的中心差异，\(\Phi\) 为标准正态分布的累积分布函数。

该分析说明，增加 Cluster 间距离可以降低噪声引起的错误路由概率。需要注意，理论结论依赖于一个重要假设：Cluster 在其内部较紧凑的特征维度上，也具有较好的 Cluster 间区分性。

### Hessian Conditioning：通过改善条件数加快 Router 收敛

作者研究 Router embedding \(e_i\) 对应的 Hessian 条件数：

\[
\kappa(H)=
\frac{\lambda_{\max}(H)}
{\lambda_{\min}(H)}
\]

并给出：

\[
\boxed{
\kappa(\nabla_{e_i}^2L_{ACMoE})
\leq
\kappa(\nabla_{e_i}^2L_{MoE})
}
\]

通过调整 Cluster 不同维度的尺度，使优化空间更加规整，降低条件数，从而改善 Router embedding 的优化效率。

这里讨论的是 Router embedding 的优化条件，而不是整个 Expert FFN 参数空间的条件数。

### Cluster Weight Mixing：利用 Top-k 路由概率混合多个 Cluster 的特征权重

在 Top-k 路由中，一个 Token 可能同时选择多个 Expert。作者在附录中进一步研究了混合多个 Cluster 特征权重的方式。

例如 Top-2 路由中，两个 Expert 的匹配权重分别为 \(\alpha\) 和 \(1-\alpha\)，可以构造：

\[
M_{k^*}
=
\alpha M_k+(1-\alpha)M_{k'}
\]

这样不仅考虑最匹配 Expert 的 Cluster 特征，还考虑 Token 对其他候选 Cluster 的归属程度。

论文发现，该方法在 GLaM 上有明显收益，但 Switch 更适合只采用最高匹配 Expert 对应的特征权重。因此，作者主实验主要采用最高匹配 Expert 的方案。

### Layer Placement：确定 AC Router 的使用层数

AC Router 依赖上一层的 Expert assignment，因此不能直接用于第一层。

作者通过消融实验研究不同层使用 ACMoE 的效果，发现过早使用可能导致 Cluster membership 估计不准确。对于 Switch，先使用两层标准 MoE，再在后续层使用 AC Router 的配置取得了较好的结果。

这说明 ACMoE 需要先获得相对合理的初始 Token-Expert 分配，才能利用这些分配产生有效的 Cluster 特征统计。
