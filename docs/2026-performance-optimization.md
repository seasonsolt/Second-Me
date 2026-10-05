# 2026 性能优化与实测

2026-10-05，M1 Max / 64 GB，本地 `feat/2026-upgrade`；未提交或推送。

## 已实施

- 合成并发统一读取 `CONCURRENCY_THREADS`，兼容旧小写配置；工作流开始即设置，使 GraphRAG 和三类 QA 合成都能使用。偏好合成取消原来每项仅提交一个 future 的内层线程池，改为真正跨项并发，结果顺序保持稳定。
- GraphRAG 输入包装改为稳定选择，缓存按 endpoint、模型、模板、语义配置和依赖版本分 namespace 持久保存。并发数不触发缓存失效；凭据只通过子进程环境传递。详见 [GraphRAG 说明](graphrag-performance.md)。
- 每次 Web 工作流自动生成 `data/performance/<run>/events.jsonl` 和 `summary.json`：步骤耗时、调用延迟、成功／失败／暂停状态、provider 返回的 token usage。GraphRAG 另写 `run/graphrag/*.json`，统计真实缓存 lookup 的 hit/miss。日志不保存 prompt、回答、API key 或 endpoint；指标写入失败不触发模型请求重试。SDK 内部重试包含在调用耗时内，不独立计数。
- MLX 支持按长度分组减少 padding、配置验证批次数，并记录输入／监督／padding token、训练步耗时和内存。Web 提供长度分组、验证批次及全量验证开关。默认 batch 1、梯度检查点开启、长度分组开启、验证 4 批；质量比较应选择全量验证。

## 合成并发实测

使用本地 OpenAI 兼容网关及其默认合成模型，8 个虚构事实抽取任务，SDK 重试关闭，同一批 prompt 按 2、4、6 路顺序运行；24 个返回全部满足 JSON 和事实检查。

| 并发 | 总耗时 | 样本／秒 |
| --- | ---: | ---: |
| 2 | 16.349 s | 0.489 |
| 4 | 8.432 s | 0.949 |
| 6 | 8.421 s | 0.950 |

本轮 4 路相对 2 路快约 1.94 倍，6 路没有明显额外收益。建议从 Web 的 4 路开始观察速率限制和实际耗时；代码默认仍为 2，已有用户设置保留。每个请求 4289 输入 token，其中 provider 报告 4096 缓存 token，因此这是小批量、已有 provider 缓存的测试，不能外推为完整工作流提速。原始数据见 [JSON](benchmarks/2026-10-05-synthesis-concurrency.json)。

## MLX 配置选择

以下 MLX 耗时覆盖训练、验证与 adapter 保存，不含加载、融合、GGUF 转换和推理服务。

短文本 fixture：batch 1／检查点开启 29.22 s、4.11 GB；batch 4／关闭 16.33 s、7.88 GB；batch 8／关闭 16.27 s、12.25 GB。短样本可尝试 batch 4，继续增大没有明显收益。

混合 512／1024／2048 token 的长文本 fixture：

| Batch | 检查点 | 训练阶段耗时 | 峰值内存 |
| --- | --- | ---: | ---: |
| 1 | 开启 | 30.83 s | 7.31 GB |
| 4 | 开启 | 33.58 s | 17.79 GB |
| 1 | 关闭 | 26.24 s | 21.01 GB |
| 2 | 关闭 | 23.54 s | 37.02 GB |
| 4 | 关闭 | 分配失败 | — |

保持 batch 1／检查点开启作为通用默认值。长文本 batch 2／关闭相对 batch 1／关闭只快约 10%，峰值内存却高约 76%。所有长文本测试使用相同的 28 GiB MLX allocator **指导值**，它不是硬性上限；batch 4 失败不能证明默认内存设置下 64 GB 机器一定无法运行。

不同 batch 使每轮优化器更新次数不同；速度测量不能证明模型质量相同。这里只测试 Qwen3-1.7B，未测 4B 性能；benchmark adapters 不作为个人正式模型交付。详细 fixture、指标与限制见 [MLX 实测报告](2026-mlx-performance.md) 和 [完整 JSON](2026-mlx-performance.json)。

## 验证与边界

本地临时后端及 3011 前端完成浏览器检查：MLX 控件可见，全量验证开关禁用／恢复验证批次，长度分组可切换。后端 70 项测试通过；前端类型检查与 `npm run build -- --no-lint` 通过。全项目 lint 的既有错误见升级验证记录。GraphRAG 使用真实 file cache／fnllm、模拟 HTTP 的离线集成验证，证明重开缓存能命中且变更输入能失效；尚未以真实用户笔记跑完整冷／热索引，不能给出实际 GraphRAG 提速百分比。

本轮未切换合成模型；此前的替代模型探测未显示整体速度优势。GraphRAG 子进程仍有原有取消登记限制；性能统计不代表这一限制已解决。
