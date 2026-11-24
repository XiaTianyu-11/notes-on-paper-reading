文章链接：http://arxiv.org/abs/2601.16905

## 问题

dense模型的遗忘算法通常通过优化模型参数，使模型忘记目标知识。但在MoE中，Router会根据输入动态选择Top-k个Expert，因此遗忘算法除了可以修改Expert参数，还可以通过修改Router改变知识的访问路径。

这就产生了两个问题：
1. 遗忘不彻底： Router可能将遗忘数据重新分配给其他Expert，使模型不再访问原本存储目标知识的Expert，但知识实际上仍然保留在原来的Expert中。攻击者可以绕过Router，强制调用原来的Expert（Expert Forcing Attack），重新获取目标知识。
2. 正常能力退化： Router的变化不仅影响遗忘数据，也可能改变保留数据的Expert选择，导致模型正常能力下降。
此前的SEUF采用软约束限制Router变化，并且只更新最相关的一个Expert，但软约束无法彻底阻止路由变化，单个Expert也无法覆盖分散在多个Expert中的目标知识。

## 总结

提出GRIP，一种与具体遗忘算法无关的MoE遗忘框架，通过对Router的参数更新施加硬几何约束，保持保留数据的Expert选择不变，迫使遗忘算法真正修改相关Expert内部的知识参数，而不是通过改变路由实现表面遗忘。

## 思路

作者首先发现，传统遗忘算法在MoE上容易产生Expert Selection Shift，即遗忘前后同一个输入所选择的Expert发生变化，而且路由稳定性会随着遗忘训练显著下降。

由此推断，遗忘算法可能利用Router作为优化捷径：将遗忘数据从原本负责处理它的Expert转移出去，而不是直接修改这些Expert的知识参数。这既导致保留数据的路由变化，也留下了可以被强制访问的目标知识。

因此，作者希望限制Router的更新，但直接冻结Router或者要求所有Router logits保持不变又过于严格，会限制模型正常的参数调整空间。

于是提出GRIP：只限制Router的更新不能改变Retain Set的Top-k Expert选择，而不要求所有Router logits保持不变。

具体有两种实现：
- Method A： 在遗忘训练过程中，每一步都对Router的更新进行几何投影，使其满足路由约束。
分为两个阶段：
  1. Equality Projection： 将原始Router梯度投影到保留样本隐藏表示张成空间的正交补空间，保证原本选中的Expert分数不变。
  2. Inequality Correction： 使用Randomized Kaczmarz随机投影算法，迭代修正违反分数边界的更新，避免未选中的Expert进入Top-k。

最终只有满足约束的Router更新才会被采用，Expert参数仍按照原始遗忘算法正常更新。

- Method B： 先正常执行遗忘算法，训练结束后再通过闭式解修正Router，使保留数据恢复原来的路由分数。
先正常执行遗忘算法，得到遗忘后的Router参数和Retain Set隐藏表示，然后通过最小二乘闭式解修正Router：
\[
(\widetilde\Theta+\Delta)\widetilde X_r
\approx \Theta X_r
\]

使遗忘后的Router在新的隐藏表示上尽可能恢复原始Router的输出分数，从而恢复原有路由选择。

相比Method A，不需要在每一步训练中进行投影，计算开销更低。

这样既能限制Router通过改变路由实现表面遗忘，又允许所有相关Expert正常参与遗忘优化。

## 使用的方法和技巧

### Routing Stability：衡量路由稳定性

使用Jaccard相似度计算遗忘前后同一个输入所选择的Expert集合的重合程度：
\[
RS=\frac{1}{|Q|}\sum_{x\in Q}
\frac{|S^{pre}(x)\cap S^{post}(x)|}
{|S^{pre}(x)\cup S^{post}(x)|}
\]

RS越高，说明遗忘前后的路由选择越一致。

### Expert-specific Selection-preserving Constraints：专家级路由约束

对于每个Expert分别建立两种约束：
- 原本被Retain Set样本选中的Expert：要求其Router logit保持不变，即更新方向与对应样本的隐藏表示正交。
- 原本未被选中的Expert：允许其Router logit发生变化，但不能超过进入Top-k所需的分数边界。
\[
\begin{cases}
\Delta\theta_j^\top x_i=0,
&j\in S^{pre}(x_i)\\
\Delta\theta_j^\top x_i\leq\tau_{i,j}-\varepsilon,
&j\notin S^{pre}(x_i)
\end{cases}
\]

其中 \(\tau_{i,j}\) 是未选中Expert距离进入Top-k的分数差，\(\varepsilon\) 是安全边界。

### Expert Forcing Attack：验证是否真正遗忘

设计白盒攻击：绕过遗忘后的Router，强制使用遗忘前Router为Forget Set选择的Expert，检查能否重新获取目标知识。

如果仍能恢复，说明模型可能只是改变了知识访问路径，并没有真正删除知识。

实验中，Baseline的知识恢复率为11%，SEUF为8%，GRIP降至3%，说明GRIP对这种路由绕行攻击具有更强的抵抗能力。
