文章链接：https://proceedings.neurips.cc/paper_files/paper/2025/hash/4598de7d243d528e38eb0c5d8155fb52-Abstract-Conference.html

## 问题

MoE 通过 Router 将 Token 分配给不同 Expert，理想情况下，不同 Expert 应该逐渐形成不同的功能专长，从而提高模型容量和知识处理能力。

但现有 MoE 通常使用 Auxiliary Load Balancing Loss，要求不同 Expert 尽可能均衡地接收 Token。这可能导致不同 Expert 接收到相似的 Token，产生相似的训练梯度，最终学习到相似的功能表示，形成 Expert Overlap。

与此同时，随着 Expert 逐渐同质化，Router 也越来越难以区分不同 Expert 的功能差异，导致路由决策趋于均匀。

由此形成恶性循环：
1. Expert 同质化： 不同 Expert 接收相似的 Token，学习到相似的表示，难以形成明确的功能分工。
2. Router 路由均匀化： Expert 输出越来越相似，主任务损失提供的路由区分信号减弱，Router 越来越依赖负载均衡损失。
3. 相互强化： 路由越均匀，Expert 接收到的训练数据就越相似，进一步加剧 Expert 同质化。
从 MoE 记忆机制的角度看，这意味着不同 Expert 虽然具有独立参数，但不一定形成了明确的知识分工。知识可能以相似的方式分布在多个 Expert 中，导致知识存储缺乏明确的功能边界，也增加了后续知识定位和定向修改的难度。

## 总结

提出一种促进 MoE Expert 专门化的训练框架，通过在原有负载均衡损失之外增加两个优化目标，使不同 Expert 形成差异化的功能表示，同时让 Router 对不同 Token 作出更有区分度的路由决策。

具体包括：
1. Orthogonality Loss（正交损失）： 约束同一个 Token 经过不同 Expert 得到的输出表示，使它们尽可能正交，减少 Expert 之间的功能重叠。
2. Variance Loss（方差损失）： 增大每个 Expert 对不同 Token 的路由分数方差，使不同 Expert 对不同 Token 形成更加明确的偏好。
最终将两个损失与原有 Auxiliary Loss 结合，在保持 Expert 负载均衡的同时促进 Expert 专门化。

实验在三种 MoE 模型和 11 个 Benchmark 上进行，作者报告相较于传统 MoE 基线取得最高 23.79% 的相对性能提升，并在 92.42% 的任务上优于对比方法。

从 MoE 记忆机制来看，本文提供了一种值得关注的思路：通过促进 Expert 专门化，使模型内部不同 Expert 有机会形成更加差异化的知识表示，从而让知识在 Expert 层面具有更明确的功能分布。

## 思路

作者首先通过梯度分析发现，传统负载均衡损失可能导致 Expert 同质化和 Router 路由均匀化相互强化。

因此，作者决定分别从 Expert 和 Router 两个角度解决问题。

具体有两个方法：

- Orthogonality Loss：从 Expert 角度解决功能重叠。

  对同一个 Token，计算不同激活 Expert 的输出向量，并计算它们之间的投影分量。

  通过最小化不同 Expert 输出之间的投影，使它们尽可能正交，从而减少功能表示重叠。

  这样可以促使不同 Expert 学习不同的表示方向，形成差异化的功能。

- Variance Loss：从 Router 角度解决路由均匀化。

  计算每个 Expert 对当前 Batch 中不同 Token 的路由分数方差，并通过最小化负方差来最大化方差。

  这样可以让不同 Expert 对不同 Token 形成更加明确的偏好，避免 Router 对所有 Token 都采取近似均匀的路由策略。

两个方法还会相互促进：
1. 正交损失使不同 Expert 的输出更加独立，主任务梯度在不同 Expert 输出方向上的投影更容易产生差异，为 Router 提供更加有区分度的学习信号。
2. 方差损失使不同 Token 更倾向于选择不同 Expert，让各个 Expert 接收到更加差异化的训练数据。
3. 不同 Expert 接收到不同的数据分布后，其参数逐渐形成差异，进一步促进 Expert 输出正交化。
最终形成正向循环：

Expert 输出差异化 → Router 获得更有区分度的信号 → Token 分配差异化 → Expert 学习差异化功能。

同时，作者保留原有的 Auxiliary Loss，继续约束整体负载均衡，避免部分 Expert 过载或闲置。

## 使用方法和技巧

### 1. Orthogonality Loss：专家输出正交化

对于 Token \(x_i\)，计算其经过 Expert \(j\) 的输出：

\[
\tilde{x}_{ij}=E_j(x_i)\cdot I\{s_{ij}>0\}
\]

其中，指示函数保证只考虑当前 Token 实际激活的 Expert。

接着计算不同 Expert 输出之间的投影：

\[
\operatorname{Proj}_{\tilde{x}_{ik}}
(\tilde{x}_{ij})
=
\frac{
\langle\tilde{x}_{ij},\tilde{x}_{ik}\rangle
}{
\|\tilde{x}_{ik}\|^2+\epsilon
}
\tilde{x}_{ik}
\]

通过最小化投影分量的平方范数，使不同 Expert 的输出尽可能正交。

其核心思想是：不直接约束 Expert 参数矩阵，而是约束不同 Expert 在处理相同 Token 时产生的输出表示。

### 2. Variance Loss：路由分数方差最大化

定义：
\[
\bar{s}_j=\frac{1}{N}\sum_{i=1}^{N}s_{ij}
\]

\[
L_v=-\frac{1}{n}
\sum_{j=1}^{n}\sum_{i=1}^{N}
(s_{ij}-\bar{s}_j)^2
\]

其中，\(\bar{s}_j\) 是 Expert \(j\) 在当前 Batch 中获得的平均路由分数。

通过最小化 \(L_v\)，最大化每个 Expert 在不同 Token 上的路由分数方差。

注意，这里计算的是同一个 Expert 在不同 Token 之间的路由分数方差，而不是同一个 Token 在不同 Expert 之间的方差。

### 3. Multi-objective Optimization：多目标联合优化

最终损失函数：
\[
L=L_h+\alpha L_{\mathrm{aux}}
+\beta L_o+\gamma L_v
\]

四个目标分别是：
- \(L_h\)：保证主任务性能。
- \(L_{\mathrm{aux}}\)：保持不同 Expert 的负载均衡。
- \(L_o\)：促进 Expert 输出正交化。
- \(L_v\)：促进 Router 路由差异化。
作者通过梯度分析讨论了各项损失之间的兼容性，以及正交化和路由差异化之间的相互促进关系。

### 4. Expert Overlap：衡量专家功能重叠

使用 Expert Overlap 衡量不同 Expert 输出表示的重叠程度。

数值越低，说明不同 Expert 的输出越不相似，专家专门化程度越高。

同时使用 Silhouette Coefficient 衡量不同 Expert 所处理 Token 的聚类分离程度。

### 5. Routing Variance：衡量路由差异化

统计不同 Expert 在不同 Token 上获得的路由分数方差。

数值越高，说明 Router 对不同 Token 的路由偏好越明显。

与 MaxVioglobal 等负载均衡指标结合，可以同时观察路由差异化和整体负载均衡是否得到改善。

### 6. 对 MoE 记忆机制的启发

本文的方法对 MoE 知识存储与定位有三个值得关注的启发：

1. Expert 专门化可能促进知识的局部化存储。 不同 Expert 接收不同分布的 Token，并学习不同的表示方向，有可能使特定领域或类型的知识更多地集中在部分 Expert 中，从而形成相对明确的知识存储单元。

2. Router 可以理解为知识访问路径。 Expert 参数承载其学习到的功能和知识，Router 则决定输入能够访问哪些 Expert。提高路由差异化，可能使不同类型的输入更稳定地访问对应的知识处理单元。

3. Expert 正交化可能影响知识冗余。 如果不同 Expert 对相同 Token 的输出高度重叠，可能意味着它们存在一定的功能冗余。通过正交化减少这种重叠，有可能使不同 Expert 承担更加独立的功能，进而影响知识在 Expert 之间的分布。

由此可以进一步研究：如果模型已经形成了较强的 Expert 专门化，能否通过识别负责某类知识的 Expert，实现更精确的知识定位、定向编辑和遗忘？反过来，如果删除某个 Expert 中的知识，其他 Expert 是否仍然保留了相同知识的冗余表示？

边界说明： 本文主要研究 Expert 专门化及其对下游任务性能的影响，并没有直接研究事实知识的存储位置、知识遗忘、记忆提取或成员推理攻击。因此，上述第 6 点属于结合本文方法对 MoE 记忆机制作出的研究推论，不能直接视为论文已经验证的结论。另外，论文 3.2 节关于路由方差与负载均衡兼容性的 Lemma，其方差定义与实际使用的 Variance Loss 并不完全一致，理论证明的严谨性仍有讨论空间。