# 2026 记忆层与轻量 L2

## 聊天契约

使用 OpenAI `messages`，扩展设置集中于 `metadata`：

```json
{
  "messages": [{"role": "user", "content": "我下一次预约是什么时候？"}],
  "metadata": {"enable_l0_retrieval": true, "enable_l1_retrieval": false}
}
```

显式请求设置优先于角色设置；未设置时使用角色值。无角色的个人聊天默认启用 L0、关闭 L1。显式 `false` 始终生效。旧顶层检索字段可读取，但新客户端应使用 metadata。

角色 system 优先于请求 system；无角色时合并请求中的 system，生成前部单条 system。原请求和历史角色不被修改。参考材料使用 `<reference_memories>` 区域，有命中附来源，无命中明确说明没有相关记忆；资料内容转义，作为证据而非指令。

参考区预算为 4096（`DEFAULT_MAX_MEMORY_TOKENS`），按字符数近似 token（对 Qwen 的中英文均偏保守），可完整容纳 top-1 chunk 并给 8192 推理上下文留出身份、历史和回答空间。推理与训练合成（`bounded_memory_content`、`chat_data`）使用同一计数和预算，模型训练与推理看到的参考长度一致；实际本地推理还通过 llama-server 对完整 chat template 做 token 预算校验。当前问题不得静默截断。注意训练 `--max_length` 若小于含参考区的完整 prompt，样本会被过滤。

## 事实更新无需重训

- 上传文件：保存后直接生成文档／chunk embedding，不执行 GraphRAG 或 LoRA。
- `PUT /api/documents/<id>/content`，body `{"content":"新事实"}`：替换数据库文本、重建该文档 chunks 和 embedding。
- `POST /api/documents/<id>/index`：重试索引，或为新 embedding 模型重建该文档索引。
- `POST /api/documents/reindex`：逐个复用上面的单文档索引流程重建全部文档，返回 `total`／`succeeded`／`failed`／`failed_document_ids`；有失败时返回 500 及同样的计数，可重复调用。
- 原有删除文件入口：先从所有项目文档模型集合删除该文档的索引，再删除 SQL／文件记录；索引失败不会伪装成功。

内容更新的 canonical source 是数据库 `document.raw_content`，后续 L1／L2 数据准备读取该文本。此接口不会改写原 PDF／文本文件；重新扫描原文件属于重新导入。

索引失败返回失败，保留数据库内容并标记 FAILED，可通过 index 接口重试。上传请求需要已配置可用的 embedding 服务；同步处理大文件可能较慢。

## embedding 迁移

新 collection 名包含向量空间哈希和维度；空间哈希由 embedding 模型、规范化 embedding endpoint 和可选的 embedding provider 组成。URL／key 不写入 metadata；修改聊天供应商不影响记忆索引。切换 embedding 模型或 endpoint 不会清空旧索引。bge-m3 使用 1024 维。旧的无模型标记、1536 维 cosine 集合只允许默认 ada-002 兼容读取，其他模型应显式重建，避免相同维度但不同向量空间混用。

旧的非 cosine 集合不使用 `1-distance` 阈值；新索引采用 cosine。

相似度阈值按当前配置的 embedding 模型在每次检索时解析：模型名含 `bge-m3` 时为 0.5（Cloudflare `@cf/baai/bge-m3` 实测可回答问题 top-1 余弦 0.52–0.68，不可回答 0.34–0.47），其他（如 OpenAI）为 0.7。可用环境变量 `L0_SIMILARITY_THRESHOLD`（文档 chunk）和 `L1_SIMILARITY_THRESHOLD`（L1 shade）覆盖。阈值仍应在用户实际语料上校准。

## chunk 大小

默认 `DOCUMENT_CHUNK_SIZE` 由 4000 改为 1000（`DOCUMENT_CHUNK_OVERLAP` 仍为 200），使检索单元更聚焦。该设置只作用于新建或重建的索引；已有文档需调用 `POST /api/documents/reindex` 重新切分并生成 embedding。

任何事实新增／修改／删除都会将已有 L1 版本标为 stale，检索和训练数据准备不再使用 stale 的 global bio；status bio 不做物理删除，由下次 L1 生成替换。重新生成 L1 后恢复其检索；L0 新事实无需等待 L1。

## L2 数据迁移

保留 selfqa 和 preference 主体，给可变身份／bio／经历添加参考上下文；diversity 默认删除 infoRecall，global 权重 4→1，无上下文 unanswerable 权重降至 0.25。额外保留两个明确的检索无命中负例。

生成问答前先限制参考内容，与最终训练所见的上下文保持一致。新合成样本带 `memory_format_version: 1`、`training_type`、`context`。合并跳过缺版本标记的旧事实样本并提示重新生成；显式标为 style／values／preference 的人工样本仍可保留。旧事实数据不能通过简单重命名类型自动迁移。

这些契约降低事实背诵风险，但不构成模型已实现事实遗忘的证明；最终需固定适配器验证新增、修改、删除及无命中行为。

## 离线验证

`tests/test_memory_contracts.py` 使用人工数据、mock embedding、临时 SQLite 和 Flask test client，覆盖多条／空检索、距离度量、跨模型文档删除、system 顺序与幂等、显式关闭、L1 shade 缓存及 stale、更新替换 chunks 和失败重试、HTTP 失败状态与参考区边界／预算。不调用外部 embedding，不读取私人凭据。


`tests/test_chroma_memory_integration.py` 在独立子进程使用真实 Chroma 0.4.24 PersistentClient、临时 SQLite 和项目 StorageService／DocumentService／chunker／repository。固定本地 1024／1536 维向量并阻断网络，不加载 `.env`，验证上传三条命中、事实更新／删除即时生效、其他文档保留、模型／endpoint／维度切换、旧集合保留及聊天供应商不影响索引。2026-10-04：上述两个文件合计 12 tests passed，实际 Cloudflare embedding 与模型事实回答仍需单独验证。
