文章链接：SAFEMOE: Safe Fine-Tuning for MoE LLMs by Aligning Harmful Input Routing

## 问题

MoE 模型的安全性不仅取决于 Expert 本身的能力，还取决于 Router 能否将有害输入正确分配给负责安全行为的 Safety-critical Experts。

然而，模型微调会改变参数和隐藏表示，进而改变 Router 的决策，使原本能够激活 Safety-critical Experts 的有害输入被分配给其他 Expert，导致模型安全性下降。

这会产生两个问题：
1. 无害微调也可能破坏安全性： 即使只使用正常任务数据进行微调，也可能改变有害输入的路由分布，使原来的安全机制失效。
2. 有害微调进一步放大安全问题： HFT 攻击不仅可能改变模型的输出行为，还会进一步加剧路由偏移，使 Safety-critical Experts 更难被正确激活。
此前针对 Dense 模型的安全微调方法，主要通过添加安全数据、约束参数更新或者剪枝有害参数来维持安全性，但没有直接针对 MoE 的动态路由机制进行约束。

## 总结

提出 SAFEMOE，一种专门针对 MoE 模型的安全微调框架。

核心思想是在微调过程中，使用初始安全对齐模型在有害指令上的路由分布作为参考，通过 KL 散度约束微调模型的路由变化，使有害输入继续被分配给原本的 Safety-critical Experts。

同时采用交替优化策略，将正常任务微调和安全路由正则化分开执行，降低计算开销。

这样既允许模型学习新任务，又能尽可能保留原有的安全路由机制。

## 思路

作者首先发现，MoE 模型的安全性高度依赖特定 Safety-critical Experts。通过分析微调过程，发现无论是正常微调还是 HFT，都会使有害输入的路由分布逐渐偏离初始安全模型，这种现象被称为 Safety Routing Drift。

于是作者提出三个问题：
1. 微调是否会改变有害输入的路由分布？
2. 路由漂移是否与模型安全性退化有关？
3. 如果恢复初始安全模型的路由分布，能否恢复模型的安全性？
作者通过实验发现，三个不同的 MoE 模型在正常微调和 HFT 下都出现了明显的路由漂移，而且漂移程度与 Harmfulness Score 高度相关。

随后，通过推理阶段直接覆盖微调模型的路由权重，恢复初始安全模型的路由决策，发现模型产生的有害回复明显减少。

这说明，微调后的安全退化与路由变化密切相关，保留原有的安全路由是一个有效的防御方向。

因此，作者提出 SAFEMOE，具体分为两个部分：

- Safety Routing Drift Regularization： 在微调过程中，使用 KL 散度约束当前模型与初始安全模型在有害指令上的路由分布，使有害输入继续激活原来的 Safety-critical Experts。

正常任务微调使用 \(D_{ft}\)，安全路由约束使用独立的有害指令集 \(D_h\)。后者不参与有害回答的监督训练，只用于计算路由正则化损失。

- Bi-level Greedy Optimization： 不在每一步同时优化任务损失和路由正则化损失，而是交替执行两种更新。
1. 首先使用 \(D_{ft}\) 执行正常 SFT。
2. 每隔 \(T_{\mathrm{reg}}\) 步，使用 \(D_h\) 计算路由正则化损失。
3. 通过 Adam 更新模型，使当前模型的路由分布重新接近初始安全模型。
4. 继续正常 SFT，重复上述过程。
此外，提前缓存初始安全模型在 \(D_h\) 上的路由权重，避免重复计算。

## 使用方法和技巧

### 1. Safety Routing Drift：衡量安全路由偏移

使用 KL 散度衡量初始安全模型与微调模型面对同一有害输入时的路由分布差异：
\[
d(w_{ft},x)=
D_{KL}\left(
P_{\mathrm{align}}(x)
\parallel
P_{ft}(x)
\right)
\]

其中：
- \(P_{\mathrm{align}}\)：初始安全模型的路由概率分布。
- \(P_{ft}\)：微调模型的路由概率分布。
KL 越大，说明微调前后的路由偏移越严重。

### 2. Safety Routing Drift Regularization：安全路由正则化

在正常 SFT 损失之外，增加路由正则化损失：
\[
\mathcal L_{\mathrm{reg}}=
\mathbb E_{x\in D_h}
\mathbb E_{l\in L}
D_{KL}
\left(
\sigma\left(\frac{r^{(l)}(x|w_{\mathrm{align}})}{\tau}\right)
\parallel
\sigma\left(\frac{r^{(l)}(x|w)}{\tau}\right)
\right)
\]

最终优化目标：
\[
\min_w
\mathcal L_{\mathrm{sft}}(w)
+
\mathcal L_{\mathrm{reg}}(w)
\]

其中有两个值得注意的设计：
- Layer-wise Regularization： 对所有 Transformer 层的路由分布进行约束，而不是只约束某一层。
- Last-token Regularization： 只对每条有害指令最后一个 Token 的路由权重进行约束。
### 3. Temperature：突出安全关键 Expert

在 Softmax 中引入温度参数：
\[
P_i=
\frac{\exp(r_i/\tau)}
{\sum_j\exp(r_j/\tau)}
\]

减小 \(\tau\) 会使路由概率分布更加尖锐，从而更加关注排名靠前的 Expert。

这样可以更严格地约束原本权重较高的 Safety-critical Experts，避免微调过程中这些 Expert 的路由权重明显下降。

### 4. Bi-level Greedy Optimization：交替优化

将原本同时执行的两种梯度更新拆开：
- Task Update：使用 \(D_{ft}\) 更新模型，学习新任务。
- Safety Update：使用 \(D_h\) 更新模型，恢复有害输入的安全路由。
使用 \(T_{\mathrm{reg}}\) 控制安全更新频率，并预计算初始安全模型的路由权重，减少重复前向计算。

### 5. Initial Routing Override：通过路由覆盖验证安全机制

在推理阶段，直接使用初始安全模型的路由权重替换微调模型的路由权重。

如果这样能够明显降低 Harmfulness Score，就说明安全路由的变化确实参与了模型安全性退化。

这个实验不仅用于评估安全性，也为 SAFEMOE 选择路由保持作为防御方向提供了依据。

我认为本文最值得借鉴的地方： 它把 MoE 的安全问题从传统的参数更新和输出行为，进一步推进到了 Router 的动态决策层面。尤其是将正常任务数据与安全约束数据分离的设计：允许模型在正常输入上自由学习，同时利用有害输入作为锚点，维护原有的安全路由。这种思路也可以迁移到其他需要保持特定输入路由行为的 MoE 场景。