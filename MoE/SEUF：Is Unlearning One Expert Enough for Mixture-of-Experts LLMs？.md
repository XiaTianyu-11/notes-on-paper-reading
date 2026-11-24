文章链接：https://aclanthology.org/2025.acl-long.424/

## 问题
dense的遗忘中算法通常直接对模型参数进行优化，但在MoE中，FFN被拆分成多个expert，每个token 只会激活其中top-k个expert。
假设我们希望遗忘某个领域的知识，这些知识主要由少数几个expert处理。理论上，只需要针对这些expert进行遗忘即可。但传统遗忘算法并不知道知识具体存储在哪些expert中。更麻烦的是，遗忘过程中router可能改变路由选择，使原本负责目标知识的expert不再被激活。于是，遗忘算法可能更新了错误的expert。

## 总结
识别真正负责目标知识的expert，并在遗忘过程中确保它们持续参与计算

## 思路
作者首先将统一领域数据输入，发现不同层中只有个别expert被高频选中，其他的处于随机水平
让模型处理遗忘数据，比较遗忘过程中的模型和预训练模型在处理时expert的重合比例，发现比例一直在下降（即使loss在降低）。
由此推断expert selection shift导致MoE遗忘效力低（但这似乎假设了每个expert就是负责不同领域的，不过首先作者识别了高频激活的expert似乎能印证这个假设）
算法流程为：从遗忘集中抽取一部分数据计算各层路由分数，筛选出top-m expert，只允许这些expert和对应router更新，在原有遗忘算法上加上router anchor loss，确保这些expert始终被激活。这样就解决了上面提到的两个问题。

## 使用方法和技巧
### 基于gating score的task affinity方法：识别expert
对所有样本、每个样本内部所有token分配给expert_i的router权重做平均，得到分数，然后在每一层中对分数排序选择top-m expert。
### router anchor loss
优化目标是让router的输出尽可能接近目标expert对应的向量，是一个L2距离。
