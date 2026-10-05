# GraphRAG 缓存与并发

## 已核实的旧行为

- 此项目打包的 GraphRAG CLI 默认启用缓存；相对 `cache` 由 `--root` 解析，原本位于项目的 `lpm_kernel/L2/data_pipeline/graphrag_indexing/cache`。没有发现训练重跑时删除该缓存的逻辑。
- `fnllm` 缓存键包含实际 messages、history、model 和请求参数，因此相同笔记经过不同随机包装模板后会产生新键。原来的 GraphRAG 输入准备会随机选择模板，这是重复索引可能零命中的具体原因；不能据此断言每次真实运行都是零命中。
- 原缓存键未包含 API endpoint；同名 model 切换 provider 时，单一缓存可能复用错误来源的结果。
- 并发值原本固定为 2；Web 的 `CONCURRENCY_THREADS` 设置在 GraphRAG 之后的 preference 步骤，不能影响首轮索引。

## 当前行为

- Web 工作流开始即设置已保存的并发值；GraphRAG 的 chat 和 embedding 均使用共用 `get_synthesis_workers()`。优先大写变量，兼容小写变量，默认 2，非正整数明确失败。提高并发不改变缓存键。
- 笔记包装根据稳定内容哈希选取，同一输入重复准备保持一致。
- 持久缓存位于 `resources/L1/graphrag_cache/<namespace>`；namespace 包含 provider endpoint、model、语义配置、模板文本及 GraphRAG/fnllm 版本，不包含输出目录、并发数或 API key。
- 不以整批笔记内容作为 namespace，避免改一条笔记就使整批缓存失效。每个实际请求的内容仍由 fnllm 计算哈希；变更输入、历史或请求参数会产生 miss，未变请求可复用。
- 输出重建不清理缓存。首次升级使用新 namespace，旧缓存保留但不迁移；因此首轮仍可能是冷缓存。
- 语言替换只发生在 runtime prompt 副本，原始模板保留 `<lang>`，切换语言不会永久污染模板。
- runtime 设置放在 `resources/L1/graphrag_runtime`，只保存 credential 占位符；真实凭据仅传入子进程环境。索引程序不读取 `.env`，不输出配置、prompt 或 key。

## 如何测量

每次 GraphRAG 子进程写入 `run/graphrag/<source>-<time>.json`，包括：

- `chat_hits` / `chat_misses`
- `embedding_hits` / `embedding_misses`
- 其他 pipeline cache 的 hits/misses，以及写入操作数
- 耗时、并发数、cache namespace、成功状态和失败 workflow 名称

这是对真实 cache `get()` 的计数，非基于目录中文件数量推算。命中率按各类型 `hits / (hits + misses)` 计算；没有 lookup 时不应解读为 0% 命中率。计数不是 provider 请求数，cache miss 后可能因重试产生多个请求。

## 离线验证与限制

`tests/test_graphrag_runtime.py` 使用真正的 GraphRAG file cache、fnllm 和 OpenAI 客户端，所有 HTTP 请求被 `httpx.MockTransport` 截获，仅返回虚构数据。验证：

1. 相同请求在重新打开持久缓存后命中，未再次调用 mock provider。
2. 修改输入会 miss；embedding 重复请求同样命中。
3. 模型、endpoint、模板变化使 namespace 变化；并发、凭据及输出位置变化不会。
4. runtime 配置不保存测试凭据，两个模型收到同一并发，语言切换不修改原模板。

没有使用真实笔记、Cloudflare key 或云端调用。这些结果证明缓存机制可复用，不能代替生产吞吐测试。后续用同一批输入做冷／热两次索引，并比较 JSON 指标和总耗时，才可量化实际收益。发生模型同名换权重且 endpoint 不变时，应人工更换 model 版本名或清理对应 namespace。
