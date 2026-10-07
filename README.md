# Second-Me 2026

中文 | [English](README_en.md)

> **状态：已冻结。** 这个分支完成了 Second-Me 底座的升级，之后不再开发。有证据的数字分身后续在 **[twin](https://github.com/seasonsolt/twin)** 中继续推进。

本仓库是 [mindverse/Second-Me](https://github.com/mindverse/Second-Me) 的**社区分支**，不是官方版本，与 Mindverse 没有隶属关系。

上游最后一次实质更新是 2025 年 5 月。这个分支把底座升级到 2026 年的业内水平，并验证了本地 LoRA 微调架构能够端到端跑通：合成训练数据、在 Apple Silicon 上训练、合并、转成 GGUF、部署并对话。

## 改了什么：跟上底座

| | 上游（2025） | 本分支（2026） |
|---|---|---|
| 基座模型 | Qwen2.5-Instruct，默认 0.5B | 默认 **Qwen3-1.7B**，可选 Qwen3-4B-Instruct-2507 |
| Mac 上训练 | 只支持 CUDA/CPU；Mac 上 Trainer 会误用 MPS，训练中途显存泄漏 | macOS arm64 在 Web 界面里用 **MLX LoRA** 训练：训练 → 合并 → GGUF → llama-server；CUDA/CPU 仍走 PyTorch |
| 记忆 | 事实训练进 LoRA，加一条记忆就要重新训练 | 事实来自检索，LoRA 学风格；新增、修改、删除记忆都不用重新训练 |
| 上下文 | 每个请求 1024 token，检索到的内容放不进去 | 8192 token，参考记忆有单独的预算 |
| 对话模板 | ChatML，所有 token 都算损失 | 原生模板，只对 assistant 计算损失，Qwen3 默认关闭思考模式 |
| 检索阈值 | 写死 0.7（只适合 OpenAI 的 embedding，用 bge-m3 什么都检索不到） | 按 embedding 模型自动选择，可用 `L0_SIMILARITY_THRESHOLD` 覆盖 |
| llama.cpp | 打包的旧版本 | 转换器和运行时固定为同一版本；原生 Linux 自动编译 GPU 版（CUDA 或 Vulkan） |

### LoRA 架构跑通的证据

- **训练**：Qwen3-1.7B，在 Apple M1 Max 上用 MLX 做 LoRA 训练，183 条训练样本（另留 61 条验证），549 次迭代，验证集损失从 3.71 降到 2.24。
- **部署**：合并后的模型转成 GGUF，由 llama.cpp 提供推理（Mac 上用 Metal，Linux 上用 Vulkan 跑在 RTX 4070 Laptop 上），通过 Web 界面对话。
- **在真实运行中发现并修复的问题**：embedding 分批发送、大模型请求超时、GraphRAG 按退出码判断成败、MLX 缓存上限、graphrag 包安装，以及 Web 界面生产构建崩溃。

## 评测：Second-Me 2025 vs Second-Me 2026 vs twin

三个系统用同样的 10 份文档建档，回答关于同一个人的同一套 57 道题，由同一个裁判打分。

| 指标 | Second-Me 2025 | Second-Me 2026 | twin |
|---|---|---|---|
| 事实准确率 | 6.2% | 68.8% | **98.4%** |
| 记忆里没有的事不编造 | 10% | 80% | **100%** |
| 风格“像本人”（1–5） | 1.0 | 1.9 | **3.9** |
| 通用问题回答质量（1–5） | 2.2 | **3.0** | 2.0 |
| 回答耗时中位数（事实题） | 2.2 秒 | 4.3 秒 | 7.8 秒 |

题目构成：32 道文档里的事实题、10 道记忆里没有答案的题、10 道开放的风格题、5 道通用题。

**怎么解读**

- **2025 → 2026** 反映的是本分支的改动。两者都在本地用 llama.cpp 跑一个微调过的小模型（上游 Qwen2.5-0.5B，本分支 Qwen3-1.7B），所以一部分提升来自更大的基座模型。
- **twin** 用云端大模型（`gpt-6.1-sol`）作答，和本地 1.7B 模型不是同等条件下的比较。它的领先也来自设计：每个事实都挂着可核对的原文，置信度由系统计算而不是模型自评，证据不足时主动弃权。
- twin 只以本人身份、依据本人资料作答：没有资料的通用问题（比如“对比 Kafka 和 RabbitMQ”）会直接拒答，所以这一项得分最低。

**评测条件与局限**

- 裁判是 `claude-sonnet-5-5`，和所有作答模型都不是同一家族。题目由 `gpt-6.1-sol` 生成，和 twin 的作答模型同一家族，可能对 twin 略有利。
- 单次运行，57 道题，题目来自一个人的私人文档。题库和回答原文不公开，评测工具在 [eval/](eval/)。
- Second-Me：2025 版用 Qwen2.5-0.5B（在 RTX 4070 Laptop 上训练），2026 版用 Qwen3-1.7B（在 M1 Max 上用 MLX 训练）；embedding 都用 `@cf/baai/bge-m3`。twin 使用未修改的预发布代码和同样的 embedding。

## 为什么冻结

这个分支是一次架构上限的验证：凡是能替换的模块，都换成了 2026 年第三季度最新、最好的选择（基座模型、训练框架、检索式记忆、上下文长度、检索阈值、推理运行时），结果仍然达不到预期：事实准确率 68.8%，不编造 80%，风格 1.9 分。

剩下的差距不在模块，而在架构本身：本地小模型加 LoRA 负责作答，回答没有可核对的原话证据，不按日期作答，置信度靠模型自评，证据不足时也不会主动弃权。这些换模块补不上，所以转到 **[twin](https://github.com/seasonsolt/twin)** 从头设计：每个事实挂着逐字核对的原话和日期，置信度由系统计算，证据不足就弃权，评测带对照组和置信区间。后续开发都在 twin，不再在这套代码上继续。

本仓库会保留，作为一个能用的、本地优先的 Second-Me（Qwen3 + MLX 训练）。仍然欢迎针对当前版本的问题反馈和修复。

## 快速开始（Apple Silicon）

```bash
git clone https://github.com/seasonsolt/Second-Me.git
cd Second-Me
make setup   # 需要 arm64 的 Python 3.12
make start
```

- 训练页会自动选择 MLX。
- 检索阈值会按 embedding 模型自动选择；如果用的不是 OpenAI 系列或 bge-m3，请用 `L0_SIMILARITY_THRESHOLD` 自己校准。
- 如果 `python` 指向 x86 版本（报错 `Bad CPU type`），先把 arm64 的 Python 3.12 放到 PATH 最前面。

设计文档：[升级计划](docs/2026-upgrade-plan.md)、[记忆层与轻量 L2](docs/2026-memory-migration.md)、[验证记录](docs/2026-upgrade-validation.md)。其余用法与上游相同，见 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md)。

## 致谢与许可

- 原项目：[mindverse/Second-Me](https://github.com/mindverse/Second-Me)，论文 [AI-native Memory 2.0: Second Me](https://arxiv.org/abs/2503.08102)。
- 许可证：Apache License 2.0，与上游相同，见 [LICENSE](LICENSE)。本分支的修改说明见 [NOTICE](NOTICE)。
