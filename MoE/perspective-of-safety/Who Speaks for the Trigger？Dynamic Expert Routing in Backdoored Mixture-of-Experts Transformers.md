文章链接：NeurIPS 2025 - Who Speaks for the Trigger? Dynamic Expert Routing in Backdoored Mixture-of-Experts Transformers

## 问题

传统 LLM 后门攻击主要通过数据投毒（DPA）、权重修改（WPA）或隐藏状态操纵（HSA）植入后门，但没有充分利用 MoE 架构自身的动态路由特性。

MoE 中不同 Expert 会形成专业化分工，对不同任务和输入具有不同的响应偏好。这种专业化虽然提高了模型效率和性能，却也带来了新的攻击面：某些 Expert 可能对特定触发器表现出更高的敏感性。

因此，产生了两个问题：
1. 如何定位敏感 Expert： 不同层有大量 Expert，不能直接确定哪些 Expert 对特定触发器最敏感。随机选择 Expert 可能导致后门成功率低，或者严重损害正常任务能力。
2. 如何控制触发输入的路由路径： 即使找到了敏感 Expert，正常 Router 也不保证触发输入一定会经过它们。因此，需要设计与目标 Expert 相适配的触发器，并控制触发输入的 Top-K 路由。
## 总结

提出 BadSwitch，一种针对 MoE 架构的后门攻击框架，通过联合优化触发器 Embedding 和基于梯度敏感性的 Top-S Expert 定位，将任务相关的后门植入 MoE 的动态路由路径。

其核心是：先利用干净样本与触发样本对 Expert 参数产生的梯度差异，识别每层最敏感的 Top-S Expert；再学习适合模型内部表示的触发器；最后限制触发样本的 Top-K 选择范围，使其只能从目标敏感 Expert 集合中选择，而正常样本仍使用原始路由机制。

在 Switch Transformer、QwenMoE 和 DeepSeekMoE 三种架构上进行实验，覆盖分类、文本生成等任务，攻击成功率最高达到 100%，同时保持较好的正常任务准确率。


## 思路

作者首先发现，MoE 的 Expert 专业化会带来不同的输入敏感性。通过观察训练过程中的梯度变化，发现部分敏感 Expert 在学习触发样本时表现出较强的梯度一致性和较快的收敛，而处理干净样本时梯度变化更加不稳定。

由此产生一个思路：既然触发样本会让特定 Expert 产生明显不同的梯度响应，就可以利用这种差异反向定位敏感 Expert，而不是人为指定攻击目标。

接下来，作者需要解决如何构造适合这些 Expert 的触发器。传统固定字符串不一定符合模型的内部表示偏好，因此作者引入可学习的 Trigger Embedding，通过梯度下降优化触发器，再将其映射回真实词表中的 Token。

最后，将前面得到的敏感 Expert 和优化后的触发器结合，通过限制触发样本的 Top-K 候选 Expert 集合，控制其内部路由路径。

具体分为两个阶段：

- Pretraining：
1. Random Backdoor Injection：在训练数据中随机加入简单触发器，构造干净样本和投毒样本。
2. Expert Cluster Tracing：计算各 Expert 对干净样本和触发样本的梯度敏感性，选择每层 Top-S Expert。
3. Adaptive Trigger Construction：优化触发器 Embedding，并映射为真实 Token，构造最终后门数据集。
- Post-training：
1. Target Expert Injection：使用优化后的触发器和敏感 Expert Cluster 继续训练。
2. 对触发样本，将 Top-K 的候选范围限制在目标 Top-S Expert 中；对干净样本，保持原始的全局 Top-K 路由。

最终实现：触发样本进入指定的敏感 Expert 集合，并在其中动态选择 Top-K；干净样本则继续正常路由。Zhao 等 - 2025 - Who Speaks for the Trigger Dynamic Expert Routing in Backdoored Mixture-of-Experts Transformers.pdf


## 使用的方法和技巧

### 1. Expert Cluster Tracing：基于梯度敏感性定位 Expert

分别收集每个 Expert 在干净样本和触发样本上的梯度，并对训练过程中的梯度取平均：

\[
\overline{\nabla}_{c}
=\frac{1}{T}\sum_{t=1}^{T}\nabla_c^{(t)}
\]

\[
\overline{\nabla}_{p}
=\frac{1}{T}\sum_{t=1}^{T}\nabla_p^{(t)}
\]

然后设计 SenScore：

\[
SenScore=
\mathbb E_{D_p,D_c}
\left[
\left\|
\overline{\nabla}_{p}
-
\overline{\nabla}_{c}
\right\|_2
+
\alpha
\frac{
\|\overline{\nabla}_{p}\|_2
}{
\|\overline{\nabla}_{c}\|_2+\epsilon
}
\right]
\]

包含两个部分：
- 梯度差异： 衡量触发样本与干净样本对 Expert 参数产生的梯度差异。
- 梯度相对幅度： 衡量触发样本产生的梯度相对于干净样本有多大。

对每个 Transformer Block 独立计算 SenScore，选择分数最高的 Top-S Expert，组成目标 Expert Cluster。

技巧： 不通过 Expert 激活频率或随机选择来定位攻击目标，而是利用不同输入引起的参数梯度变化，识别对触发器最敏感的 Expert。

### 2. Adaptive Trigger Construction：可学习触发器优化

传统方法直接指定固定字符串作为触发器，可能无法充分适应模型的内部表示。

作者随机初始化可学习的触发器 Embedding：

\[
Emb_{\text{trig}}\in\mathbb R^d
\]

将其拼接到投毒样本的输入 Embedding 中：

\[
\widetilde H_x=[H_x;Emb_{\text{trig}}]
\]

然后与模型参数一起训练，使触发器 Embedding 逐渐适应模型。

训练结束后，利用余弦相似度，从模型词表中检索与优化后的 Embedding 最接近的 Top-q 个 Token：

\[
Trigger\_Tokens=
TopQ_{w\in V}
\cos(H_w,Emb'_{\text{trig}})
\]

最后将这些 Token 随机插入原始投毒样本，构造最终的后门数据集。

技巧： 将离散触发器的设计转化为连续 Embedding 的梯度优化问题，再通过词表检索映射回离散 Token，实现自适应触发器构造。

### 3. Target Expert Injection：基于 Top-S 的动态 Top-K 路由约束

设某层的全部 Expert 为 \(E\)，定位得到的敏感 Expert 集合为：

\[
E_{\text{target}}\subset E
\]

作者设计条件路由策略：

\[
\pi(X)=
\begin{cases}
TopK(E_{\text{target}};X),&X\in\hat D_p\\
TopK(E;X),&X\in D_c
\end{cases}
\]

具体来说：
- 触发样本：只能在目标 Top-S Expert 中选择 Top-K。
- 干净样本：仍然在全部 Expert 中选择 Top-K。

这里需要区分 Top-S 和 Top-K：
- Top-S：确定允许参与后门路由的候选 Expert 集合。
- Top-K：Router 在这个候选集合中实际选择的 Expert 数量。

技巧： 不直接固定触发样本的 Expert，而是限制 Router 的候选范围，保留候选集合内部的动态选择能力。同时，每层独立确定敏感 Expert，最终形成跨层的目标路由路径。

### 4. SenScore Ablation：验证敏感性指标的有效性

作者设计了四种替代指标进行对比：
|指标	|衡量内容|
|---|---|
|Gradmean	|梯度均值|
|Graddiff	|梯度差异|
|Actmean	|激活均值|
|Actdiff	|激活差异|
|SenScore	|梯度绝对差异与相对幅度的组合|

实验中，SenScore 达到：
- ACC：86.75%
- ASR：90.39%

其他指标虽然也能取得较高 ASR，但正常准确率明显下降，最低只有 55.56%。

这说明单纯考虑梯度或激活的某一项统计特征，不如同时考虑梯度差异和相对幅度，能够更好地平衡攻击成功率与正常任务能力。

### 5. Adaptive Trigger / Random Expert Selection：消融实验

作者分别验证了自适应触发器和敏感 Expert 定位的作用。
- 固定触发器： 使用固定短语替代学习得到的触发器。其中一种配置 ACC 为 82.24%、ASR 为 21.34%；另一种配置 ASR 达到 100%，但 ACC 降至 49.88%。说明固定触发器难以同时保证攻击成功率和正常能力。
- 随机 Expert 选择： 随机选择两组 Expert Cluster，得到的 ASR 分别为 53.42% 和 42.03%，明显低于 BadSwitch。

这验证了两个组件的必要性：触发器需要适应模型表示，目标 Expert 也需要通过梯度敏感性分析进行定位。


### 6. 实验评价与防御鲁棒性

作者使用以下指标：
- ACC：无触发器时的正常任务准确率。
- ASR：有触发器时成功执行目标任务的比例。
- PPL：生成任务的困惑度，用于衡量生成质量。

主要实验包括：
- 模型：Switch Transformer、QwenMoE、DeepSeekMoE。
- 数据集：SST-2、AGNews、C4、ELI5。
- 对照方法：BadNets、VPI（DPA）和 BadEdit（WPA）。

此外，作者还测试了两种防御：
1. Text-Level Detection： 使用 ONION，根据输入 Token 对困惑度的影响检测潜在触发器。
2. Model-Level Retraining： 使用干净数据对被攻击模型进行部分微调。
在 AGNews 上，经过文本级防御后，BadSwitch 仍然达到 94.07% ASR 和 87.18% ACC，说明其后门具有一定的防御鲁棒性。

补充说明： 本文的攻击假设攻击者能够访问训练数据、微调模型并观察 Expert 梯度，因此并非黑盒攻击。此外，生成任务的攻击效果存在较大差异，不能将分类任务中接近 100% 的 ASR 直接推广到所有生成任务。论文最值得借鉴的研究思路，是将 MoE 的 Expert 专业化、梯度动态和路由选择结合起来，证明动态路由本身也可以成为后门攻击的载体。