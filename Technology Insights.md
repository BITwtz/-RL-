# A. FlashRL 

论文：https://arxiv.org/abs/2311.06917

博客：https://fengyao.notion.site/flash-rl

github：https://github.com/yaof20/Flash-RL

### 针对问题

训练的主要瓶颈常常不在“更新参数”，而在 rollout 生成。FlashRL 旨在**让 rollout 用 INT8/FP8 做量化推理提速/省显存，但训练仍保持全精度的稳定与效果**。

### 核心观点

洞察 A：RL 更新依赖采样动作的 logprob，训练侧与推理侧数值实现不一致，会把本应 on-policy 的训练退化为 off-policy，从而带来不稳定或性能损失。

洞察 B：量化 rollout 的收益取决于“模型规模 × 生成长度”，并不绝对有益。量化有额外开销，更推荐用在 14B+（尤其 32B）且长 CoT（如 DAPO）的场景。

### 技术实现

* TIS 修正 rollout–training mismatch：基于重要性采样思想对 policy loss 施加修正系数，并对系数做截断处理防止过修正。
* INT 8 在线量化：仅在训练开始时计算一次校准结果，以供后续复用。

# B. QuRL

论文：https://openreview.net/forum?id=eG0bpCwdKn

### 针对问题

序列由量化策略生成，但更新在全精度策略上发生，这种 mismatch 导致训练后期出现 collapse。RL 每 step 的参数更新很小（受 trust-region/clip 约束），小到被量化误差淹没，量化模型几乎无法感知这些更新。

### 技术实现

* ACR：Adaptive Clipping Range，动态调整 clip 的信赖域边界。仅使用 TIS 会导致 KL 变大和梯度估计偏差。原因在于目标函数后项会吸收外面的修正因子，导致裁剪区间变窄，会裁剪一些本不需要裁剪的 token。而 ACR
  保持一个固定的上界基准。
* UAQ：Update-Aware Quantization，问题在于 RL 的每步更新太小，小到会被量化噪声淹没。如果使用 PTQ 校准，开销大。如果使用 QAT，会引入训练图与推理引擎和 vLLM 之间更多实现差异，反而加剧 rollout/训练 mismatch。UAQ 对权重和激活使用等价缩放，实践经验选择缩放因子为 1.5。
* 工程上实践发现：**当前版本 vLLM 的 FP8 KV-cache 量化收益不明显**，因此实验里不启用 KV-cache 量化。

### 量化成果

模型越大，量化带来的吞吐收益越明显。7B 的 INT8 大多带来 20%–30% 加速；32B 在 A100 上约加速 70%、H100 上约加速 90%。

# C. QeRL

论文：https://www.arxiv.org/pdf/2510.11696

github：https://github.com/NVlabs/QeRL

### 针对问题

LLM 的 RL 常见瓶颈，rollout 慢、显存压不住、训练成本高。QeRL 用硬件友好的 **NVFP4（FP4）+ LoRA 把 rollout 与 prefilling 加速**，同时把“量化噪声”从副作用变成探索增强，在效率之外提升 RL 收敛与最终效果。

### 核心观点

洞察 A：量化噪声在 RL 里不完全有害，能提高探索。SFT 追求“拟合数据分布”，噪声常是坏事；RL 追求“发现高回报轨迹”，可控噪声能变成免费的探索机制。

洞察 B：噪声必须动态可控，否则静态量化误差难以匹配探索-收敛的阶段性需求。

### 技术实现

* QeRL 用 NVFP4 存权重，并在 rollout 与 prefilling 阶段集成基于 Marlin 的推理方式，保持精度的同时提升吞吐。
* AQN（Adaptive Quantization Noise）把静态量化误差变成可调探索。对每个量化线性层，每次 forward 都采样一个随机噪声；从较大的起始噪声逐步降到较小的结束噪声，以平衡探索与收敛；实验探究验证指数衰减更利于后期稳定高 reward。

### 对比 FlashRL

QeRL 的主张更激进，用 NVFP4 让 rollout 本身更快，并把量化带来的不确定性当作探索红利。

### 量化成果

rollout 阶段可达 >1.5× 的加速；并指出随着训练后期输出更长，速度优势会更明显。相对 QLoRA 约 1.8× 的端到端训练加速。

# D. Jet-RL

论文：https://arxiv.org/abs/2601.14243

### 核心观点

洞察 A：统一训练与 rollout 的 FP8 精度流，从源头消除数值 mismatch。

### 技术实现

* 训练与推理共享一致的量化行为，恢复 on-policy 一致性。
* 讨论了线性层三类 GEMM 的 FP8 量化布局与算子设计，并在训练中对 GEMM 进行 FP8 加速，同时在关键位置保持 BF16 以稳住收敛（例如梯度传输保持 BF16）。
* 用 vLLM 做 inference engine、VeRL 做 RL 训练框架，量化 GEMM 参考 DeepGEMM，并用 Triton 实现量化/transpose/融合算子。

### 量化成果

在 rollout 阶段实现了 33% 的加速，在 training 阶段实现了 41% 的加速，相较于 BF16 训练，端到端加速了 16% ，同时在所有设置中保持稳定的收敛，导致的精度下降可忽略不计。

# E. 低精度 RL 训推经验贴总结

1. vLLM 对低精度格式的支持与使用方式

支持的低精度类型：vLLM 已支持多种 8 比特低精度推理格式，包括 FP8（8 位浮点）和 INT8 量化模式，覆盖纯权重量化及权重 + 激活同时量化。

使用方式：vLLM 提供了**离线量化**和**在线量化**两种工作流。

对于离线量化，推荐使用 llm-compressor 工具对模型进行一次性量化，然后加载量化后的模型权重。

对于在线动态量化，vLLM 支持在推理引擎中对加载的模型实时量化。用户可通过环境变量或 API 参数开启。例如，在 FlashRL 中使用环境变量控制 vLLM 在线量化：export FLASHRL_CONFIG=fp8_vllm 开启 FP8 在线量化，或 FLASHRL_CONFIG=int8 开启 INT8 量化。FP8 动态量化无需校准且硬件友好，是首选方案；INT8 则需结合校准或 QAT 技术，在推理引擎和训练框架间同步更新量化权重。

混合精度策略如 W8A16，这种权重量化方法降低了一半模型内存，但计算时需将权重反量化为 FP16 参与乘法，因而在计算密集场景收益有限。

2. 利用 FP16 克服训练-推理不匹配问题  https://www.emergentmind.com/papers/2510.26788#related-papers

训练策略和推理策略之间的数值不匹配根本原因在于浮点精度本身。广泛采用的 BF16 浮点数虽然动态范围大，但会引入较大的舍入误差，从而破坏训练和推理之间的一致性。**改回 FP16 浮点数即可有效消除这种不匹配**。

实验结论：

* 在不同设置下 BF16 和 FP16 之间的训练奖励比较，表明 FP16 具有更优异的稳定性和收敛性。
* FP16 在标记和序列级别上显著降低了训练-推理不匹配。
* 从 BF16 切换到 FP16 可以稳定和延长 RL 训练，FP16 的性能优于所有 BF16 基线。










