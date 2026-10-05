# Second-Me 2026 升级计划

记录日期：2026-10-04；修订：v2（结合同日 review，并对关键结论复核）。

状态：核心实现与本机验收已完成，未提交或推送。详细证据及未实测范围见 [验收记录](2026-upgrade-validation.md)。

基线：master，`d0e40251d9de61b3340b8d0d7d83150669f1885a`。

## 目标与约束

按优先级：

1. 默认基座升级为 `Qwen/Qwen3-1.7B`；较大档位采用用户已确认的 `Qwen/Qwen3-4B-Instruct-2507`。两个型号使用各自原生模板，不能假定模板相同。
2. Apple Silicon 的 Web 训练流程自动使用 MLX-LM LoRA，随后融合、转换 GGUF，并由 llama-server 推理。其他平台保持 PyTorch CUDA/CPU 路线，不采用 MPS。
3. 事实记忆交给检索层；LoRA 主要学习表达风格、偏好与价值判断，新增记忆无需重训。

遵循现有代码风格，最小必要修改。每阶段独立可运行、可验证。未经用户要求不 commit、不 push；若后续要求提交，使用英文说明修改原因。

## 已确定的实施决策

- [x] 较大档位采用 `Qwen3-4B-Instruct-2507`（用户明确确认）。
- [x] 按已批准计划，Mac 首版明确采用 SFT-only，DPO 作为独立后续能力。
- [x] 用户批准开始开发，并授权并行 subagent。
- [x] 三个 subagent 分别负责 Qwen3/构建、MLX 编排、检索/轻量 L2；主 agent 负责前端、推理服务、依赖隔离与集成实测。共享文件按所有权划分，修改冲突先协调。

## 工作区现状

- 主仓库：`<repo>`。
- 当前工作区及计划唯一维护位置：`<worktree>`（本地独立 worktree）。
- 复核时原分支已释放并删除；Claude 工作区现位于 `claude/goofy-sinoussi-93fbf4`。已依照用户最初要求在当前工作区执行 `git switch -c feat/2026-upgrade`，从上述 master 基线创建并切换分支。
- 不修改 Claude 工作区的 review 文档；所有实现与计划更新位于当前工作区。
- Review 来源：`<repo>/.claude/worktrees/goofy-sinoussi-93fbf4/docs/2026-upgrade-plan-review.md`。

## 已核实的代码事实

- `pyproject.toml` 固定 Transformers 4.47.1、Torch 2.5.1、PEFT 0.14.0、TRL 0.13.0。Qwen3 官方说明要求 Transformers 至少 4.51.0；最低版本不等于最终验证通过的版本组合。
- Web 入口 `L2/train_for_user.sh` 实际传入 rank=8、alpha=16、all-linear，覆盖 `train.py` 的 rank=64 等默认参数。
- Web 流程调用 PyTorch SFT；DPO 位于独立脚本，检查到的 Web 编排未自动调用它。
- 现有 MLX 脚本写死 Qwen2.5-7B；融合后启动 MLX server，没有完成目标 GGUF 部署链路。
- 项目内 `L2/convert_hf_to_gguf.py` 没有 Qwen3 架构注册。打包的 llama.cpp 也必须核验并与转换器同步固定版本。
- `chat()` 直接使用 `MultiTurnMessageBuilder`，未调用 `_get_strategy_chain`；后者顶层开关读取不一致是潜在问题，不是当前主路径检索失效的主因。
- `embedding_service.py` 遍历 Chroma 查询结果外层 `ids`，单查询只处理一个 chunk；L1 DTO 写死 `shades=[]`，检索器还调用 EmbeddingService 不存在的方法；消息构建器将新 system 追加到对话末尾并原地修改请求。
- llama-server 使用 ctx=2048、parallel=2；项目 chunk 配置为字符数，而上下文预算须按模型 token 计算。不能直接把字符长度等同于 token 长度。
- `local_llm_service.py` 将检测结果强制设为 `cuda_available=False` 并默认清空 CUDA_VISIBLE_DEVICES；GPU 参数分支不可达，Mac 也没有显式 Metal offload。
- `l2_generator.py` 当前合并 preference、diversity、selfqa；context 生成默认 `do_context=False`，当前调用未启用，不纳入旧 context 生成器改造。
- selfqa 使用 bio 和固定问题，不直接读笔记；preference 使用 L1 topic 串和 bio；diversity 直接使用笔记。前两者仍可能包含事实，不能简单视为全部风格数据。

## 阶段 0：环境与运行隔离

### 工作内容

1. 通过 `POETRY_VIRTUALENVS_IN_PROJECT=true` 并明确选择 Python，在当前工作区建立独立 `.venv`，验证 Poetry 确实选择它而非已缓存共享环境。不生成额外的本地 `poetry.toml`。
2. 明确使用 `/opt/homebrew/bin/python3.12`。必要时使用临时 PATH shim，让 `python` 指向 arm64 Python，绕过 `/usr/local/bin/python` 的 x86 Python 2.7；不安装 Rosetta。
3. 核实数据库、Chroma、模型输出、日志均为工作区独立路径，未链接到 master 的运行数据。
4. 后端暂定 8003、前端 3011，启动前检查可用性；llama-server 分配独立空闲端口。不得停止现有 8002、3010 或无关的 3000 服务。
5. 修复 setup 与 Docker CUDA 的 GraphRAG 包格式处理：当前 `.tar.gz` 实为 ZIP。统一按实际格式安装，不依赖手工补救。
6. 检查 `Dockerfile.backend.apple` 使用的 `graphrag-modified.tar.gz` 是否存在及其来源，避免保留失效 COPY。该 Dockerfile 运行在 Linux，不能因名字含 apple 就启用 Metal/MLX；MLX 训练仅在原生 macOS arm64。

### 涉及文件

- `scripts/setup.sh`
- `Dockerfile.backend.apple`、`Dockerfile.backend.cuda`
- 工作区本地 Poetry 配置及忽略规则（仅在必要时修改）
- `pyproject.toml`、锁文件（版本调整在下一阶段完成）

### 验收

- [x] Python 为 arm64，虚拟环境路径独立。
- [x] 安装脚本能正确识别、安装本地 GraphRAG 包。
- [ ] 两个 Docker 构建入口的包路径和安装格式正确；不可用的 Docker 验证明确记为未执行。
- [x] 服务端口、数据库、索引和输出目录不与 master 冲突。

## 阶段 1：Qwen3 完整链路

### 1.1 模型配置与下载

- 新任务默认 Qwen3-1.7B；确认后提供 Qwen3-4B-Instruct-2507（纯非思考模型）。
- 同步下载入口、模型校验、后端默认值、前端选择器、store、产物路径和文档。
- 前端保留原 Qwen2.5 四个旧档位，归入旧型号选项；已保存配置和产物不自动改名，使用各自模板。新默认值不会强制替换用户已保存的选择。
- `utils.py` 已使用 `Qwen/{model_name}`，无需为命名增加无必要映射；校验分片 safetensors 及 index 文件完整性。
- 下载完成后校验 config、tokenizer、权重完整性。
- 内存提示按后端区分，最终以实测更新，不沿用旧型号推荐值。

### 1.2 依赖组合

1. 明确首版选择 **Transformers 4.x，约束为 >=4.51,<5**，最终固定一组实际验证通过的精确版本。5.x 作为后续独立升级，不扩大本次迁移范围。
2. 核对 PEFT、TRL、Torch、datasets 的约束与现有调用，优先保留仍兼容的组件。
3. SFT 改为 prompt/completion 数据及 completion-only loss，不再依赖 `DataCollatorForCompletionOnlyLM` 和硬编码 assistant 响应前缀。选择支持该方式的配套 TRL，核对 `processing_class` 等 API；DPOTrainer 独立检查。
4. 在隔离环境完成解析、安装及导入验证，固定通过验证的版本并更新锁文件。
5. 验证 GraphRAG、embedding 和 Web 后端仍能导入启动。

重点风险：TRL API 变化、tokenizer 行为变化、新旧依赖约束冲突。Review 中列出的“最新版本号”未作为实现依据；本计划采用固定兼容组合，不追逐最新版本。

### 1.3 训练与推理模板

- 使用 Qwen3 原生 chat template，避免旧 ChatML 模板覆盖。
- 1.7B 设置 `enable_thinking=False`；2507 纯 instruct 不依赖该开关。检查 1.7B 模板注入的空 `<think>` 块，使训练前缀与推理生成前缀一致。
- 保留规范化 messages，在对应后端训练边界格式化。PyTorch prompt 为 `apply_chat_template(system + history + user, add_generation_prompt=True, enable_thinking=False)`（按型号传参）；completion 为目标 assistant 文本和原生结束标记。
- 校验 prompt token 是否是完整样本 token 的前缀，避免自行拼接的空 think/EOS 重复。多轮样本按目标 assistant 拆分，明确哪些 token 参与 loss。
- 验证 assistant-only loss mask，不学习复述 system、user 或检索材料。
- 覆盖多轮、EOS、截断和空回答。
- 现有 `<think>…</think><answer>…</answer>` CoT 数据明确过滤或转换，不直接混入默认样本。
- 对选定 llama.cpp 启用原生 Jinja 模板；按该 revision 支持的 `chat_template_kwargs`/reasoning 参数关闭 1.7B thinking，不假设所有版本均支持同一 CLI 标志。验证流式输出及旧 Qwen2.5 模板。

### 1.4 GGUF 与 llama.cpp

- 固定支持 Qwen3 的 llama.cpp revision，使转换器、GGUF 库、运行时配套。
- 避免项目内旧转换器与新运行时混用。
- setup 对已有构建做版本核验，不能仅凭可执行就跳过升级。
- Docker 两个构建入口及运行时 CUDA 重建脚本采用同一 revision；后者目前 clone 未固定 revision。检查 Makefile 的跳过构建/复用 volume 路径，不能绕过旧构建检测。
- 先跑通 F16 GGUF，再验证可选量化格式。
- 删除 GPU 检测后强制 False 的覆盖。原生 Mac Metal 构建显式 `--n-gpu-layers 999`；CUDA 按可用构建/设备启用，CPU 显式禁用 offload。保留用户 GPU 关闭选择，避免无条件清空 CUDA_VISIBLE_DEVICES 或覆盖已有设备选择。
- 从 llama-server 日志验证实际 offload，不能以命令含 GPU 参数作为成功证据。

### 主要涉及文件

- `pyproject.toml`、锁文件、`scripts/setup.sh`
- `Dockerfile.backend.apple`、`Dockerfile.backend.cuda`、`docker/app/rebuild_llama_cuda.sh`、`Makefile`
- `lpm_kernel/api/domains/trainprocess/training_params_manager.py`
- `lpm_kernel/api/domains/trainprocess/trainprocess_service.py` 及模型下载入口
- `lpm_kernel/L2/train.py`、`utils.py`、`train_for_user.sh`
- `lpm_kernel/L2/merge_lora_weights.py`、GGUF 转换入口及相关依赖
- `lpm_kernel/api/services/local_llm_service.py`
- `lpm_frontend/src/app/dashboard/train/training/page.tsx`
- `lpm_frontend/src/store/useTrainingStore.ts`
- 模型、安装与内存说明文档

### 验收

- [x] 目标模型配置和 tokenizer 可加载。
- [x] 1.7B 完成少量样本 SFT、保存、合并、GGUF 转换。
- [x] loss 有效，assistant token 未被全部屏蔽。
- [ ] llama-server 完成普通、流式和多轮聊天，无异常模板标签。
- [ ] 1.7B/2507 模板分别验证，Mac 日志证明 Metal offload；CUDA 未实机验证时注明。
- [ ] 构建、运行时重建及旧 volume 复用采用一致 revision；converter/gguf 导入不会被旧 `L2/gguf-py` 遮蔽。
- [x] 前后端默认值一致，旧产物不会被错误识别。

## 阶段 2：Apple Silicon Web 训练接入 MLX

### 2.1 后端选择

在编排入口选择一次并写入任务参数：

| 条件 | 后端 |
| --- | --- |
| macOS arm64 且 MLX 可用 | MLX |
| 其他平台，用户启用且 CUDA 可用 | PyTorch CUDA |
| 其他现有场景 | PyTorch CPU |

Apple Silicon 缺失 MLX 时明确报错和说明，不静默启动 CPU 训练。只在现有训练、融合步骤分派，不重写任务系统。

- 在 dev 训练依赖中加入带 `sys_platform == 'darwin' and platform_machine == 'arm64'` 标记的 mlx-lm，固定与 Transformers 4.x 兼容的版本。平台标记避免其他平台安装 MLX，但不能保证依赖安装/运行一定成功，因此保留预检错误处理。
- `training_backend` 等需要持久化的字段加入 `_default_training_params` 白名单，避免被忽略。记录解析后的后端；旧配置没有该字段时采用自动选择。
- 显式 PyTorch CPU 入口设置 Trainer `use_cpu=True`；仅清空 CUDA_VISIBLE_DEVICES 不足以防止 Trainer 自动选 MPS。Apple Silicon Web 默认仍为 MLX。

### 2.2 数据与参数

- 将规范化 messages 转为 train/valid JSONL。
- 从 `utils.create_chat_data` 的现有逻辑提取可复用的 messages 生成部分，其当前输出已套 tokenizer 模板，不能直接当作 messages 使用。PyTorch/MLX 共用同一原始语义和检索拼装格式，在后端分别 tokenization。
- 不复用 `mlx_training/data_transform.py`：它有错误资源路径、写死用户名、返回值类型不一致、语言前缀不一致，以及导入即执行的问题。
- MLX 采用结构化 chat/messages 和 prompt mask；不能把已经格式化的 PyTorch prompt 字符串直接交给会再次套 chat template 的 MLX prompt/completion reader。通过选定版本的模板参数或最小 dataset adapter 保持 token 前缀一致。
- 固定随机种子，避免同源改写样本跨训练/验证集泄漏。
- 无梯度累积时，iterations 初始按 `ceil(n_train * epochs / batch_size)` 计算；实际以选定 MLX 版本对 iteration、batch 和梯度累积的定义为准，有累积时重新推导，不能套用同一公式。n_train 为拆分后样本数；验证小数据集、余数批次及样本覆盖。
- 从实际 rank=8、alpha=16 出发，核实 MLX scale 定义和 target module 映射，不照抄同名参数。
- 配置 batch size、梯度累积、序列长度及梯度检查点。
- 暂不同时大范围调整 rank、量化和学习率。

### 2.3 导出链路

目标：`MLX LoRA → fuse → 必要的反量化/HF 格式导出 → GGUF → llama-server`。

首版使用与 PyTorch 共用的 HF 非量化基座 `resources/L2/base_models/<model>`，1.7B/4B 在此 Mac 的 64GB 内存上优先验证 bf16/fp16 LoRA。通过已固定版本的 MLX fuse 保存 HF 可读取权重，接现有 MERGE_WEIGHTS → CONVERT_MODEL 阶段。4-bit 基座及反量化仅作为可选后续优化，不阻塞首版。

接 UI 前先独立证明链路可用：

- 检查融合产物 tensor 名称、dtype、config、tokenizer。
- 量化基座明确反量化过程及峰值内存。
- 不假定 MLX 量化 safetensors 可直接被 HF 转换器读取。
- 失败保留适配器和日志，不标记完成。

### 2.4 生命周期

- 复用数据准备、训练、融合、转换状态。
- 写最小 Python wrapper，调用固定版本 MLX-LM 的训练 API/回调，不重写优化器或训练循环。保持现有监控契约：先输出字面量 `***** Running training *****`，再输出匹配 `(\d+)%\|[^|]+\| (\d+)/(\d+)` 的进度行（如 `10%|#         | 1/10`），按行 flush；预检失败不得输出假完成。
- 检查返回码和产物。`_start_training` 当前用 Popen 直接执行脚本，新 shell 入口需 shebang 和可执行权限。训练完成可沿用现有结束标记。
- 取消时终止对应子进程。
- 覆盖内存不足、缺失产物和重试，避免复用半成品。
- 内存统计区分 MLX 和 PyTorch，不能把 CUDA 统计当作 Mac GPU 使用量。

### 主要涉及文件

- `lpm_kernel/api/domains/trainprocess/trainprocess_service.py`
- `training_params_manager.py`、`train_progress.py` 及相关子进程处理
- `lpm_kernel/L2/mlx_training/`
- `lpm_kernel/L2/memory_manager.py`
- 前端训练状态及参数展示

### 验收

- [ ] M1 Max 上通过 Web UI 完成短程 LoRA 到 GGUF 聊天闭环。
- [x] 记录耗时、吞吐和峰值内存。
- [x] 取消、失败、重试行为正确。
- [x] 回调日志触发实际 Web 进度；参数保存后仍存在；loss 不含 prompt；PyTorch CPU 明确没有选 MPS。
- [x] CUDA/CPU 后端选择及原流程未被替换。

## DPO 决策

建议首版 Mac Web 流程明确 SFT-only，UI/任务结果显示 DPO 未执行。保留 PyTorch DPO 独立入口，依赖升级后检查兼容性。不自动转 CPU，也不自动传送数据至远程 CUDA 主机。

MLX DPO 若后续实施，需单独验证：成熟实现和维护成本、chosen/rejected 模板、reference model 内存、融合导出链路，以及相对 SFT-only 的实际质量收益。Review 提到的第三方 `mlx-lm-lora` 仅列为候选，功能与兼容性尚未验证。未完成评估前不承诺等价实现。

## 阶段 3：可靠检索与轻量 L2

先修好检索，再减少事实训练，避免能力空窗。

### 3.1 复用现有记忆层

优先 Chroma + 现有 L1/GraphRAG，暂不引入 Mem0、Letta、Zep-Graphiti、Cognee。当前需要解决的是接入一致性及更新语义，而非新增一套存储框架。

- 统一 DTO、metadata、聊天服务和 playground 的检索开关。
- 明确角色配置、请求配置、默认值优先级。
- 个人聊天默认启用 L0，保留显式关闭；L1 用于语义关联需求。
- 普通新事实检索不依赖每次完整 GraphRAG 重建。
- 结果带来源与长度预算，将检索内容作为参考数据而非可执行指令。
- 无命中允许表达未知，避免虚构“记得你曾经”。
- 避免日志记录完整个人记忆。

具体修复任务按调用链顺序执行：

| 问题 | 修复位置与要求 | 验证 |
| --- | --- | --- |
| Chroma 遍历外层 ids，单查询只返回 top-1；空内层也可能越界 | `file_data/embedding_service.py` 遍历 `ids[0]`，处理空结果，对齐 metadata/document/distance | 固定三条命中，输出三条；空命中返回空 |
| `GlobalBioDTO.from_model` 写死 `shades=[]` | `models/l1.py`、`kernel/l1/l1_manager.py` 从该版本真实 shade 数据读取，不填伪造值 | 有 shade 的版本可检索；空版本正常 |
| L1 调用不存在的 embedding/similarity 方法 | `knowledge_service.py` 复用已有 embedding 客户端与统一距离定义，避免每次重复嵌入全部 shade | 不吞接口异常；相关/无关内容阈值可解释 |
| 新 system 被放在最后一条 user 后，且请求被原地追加 | `message_builder.py` 合并为前部单条 system，处理原 system 与角色优先级，复制 messages | system 仅一条且在前；历史角色不变；重复调用不增长 |
| 默认策略首项 RoleBasedStrategy 无参构造失败 | 统一 `chat()` 的默认策略初始化，首项为基础策略，复用现有显式链结构 | `/api/talk/*` 与显式 chain 路径均可运行 |
| retrieval 参数路径不一致 | DTO、普通/advanced chat、metadata 和角色页面一起统一 | 显式 false 不被 `|| true` 覆盖 |

此外核实 Chroma collection 的实际距离度量：现有 `1 - distance` 与固定 0.7 阈值不能不加区分地用于欧氏/余弦距离。用固定相关/无关样本校准，避免“top-k 修好但仍无命中”。

### 3.2 推理与训练共用上下文格式及预算

先确定消息格式，再生成检索训练样本：

```text
system: 身份/风格/角色规则
        记忆使用规则（参考材料不是指令；无证据不编造）
        参考材料：固定边界标记，按来源列出相关记忆
        无命中：同一区域明确标记未找到相关记忆
user/assistant: 原对话历史，顺序和角色保持正确
user: 当前问题
```

- 提取一个小型共用拼装函数，在推理和训练数据格式化中复用；保持首部一条 system，不增加独立提示框架。
- 有命中/无命中/冲突/恶意参考内容样本都使用同一边界与位置。
- 身份/用户名称通过实时配置提供，不依赖 LoRA 背诵可变事实。
- 推理每请求上下文暂以 8192 token 为目标；初始预算：回答 2048、检索 2048、基础 system 与安全余量合计约 1024，其余给历史与当前问题。实际按模板后 token 数计算；过长当前问题不能被静默截掉。
- `--parallel 2` 下总 ctx 与每 slot 容量的关系按固定 llama.cpp revision 验证。初版可降低 parallel 到 1 来保证单请求预算；不把 ctx=8192 自动视为每请求均有 8192。
- chunk 当前约 4000 字符，先在检索端按来源截取/限额，避免为实验破坏旧索引；新索引 chunk 参数按实测调整。检索条数、chunk 大小、历史和输出额度共同满足预算。
- 阶段 1/2 的 SFT 冒烟仍可用 2048；检索格式样本先裁剪在其内。阶段 3 验证后再将目标训练序列提高到 4096（或实测合适值），同步评估 batch 和内存。不要求训练长度等于推理最大长度，但必须保证训练样本完整且未截掉答案。

### 3.3 记忆更新语义

- 新增完成 embedding 后参与后续检索，无需重训。
- 修改替换旧 chunk，删除清理相应索引。
- 按 embedding 模型/维度隔离集合；1024 维 bge-m3 不写入不兼容旧集合。
- L1 摘要尚未更新时，不应覆盖较新的 L0 事实。

### 3.4 数据划分

| 来源 | 处理 |
| --- | --- |
| selfqa | 仅 bio/固定问题，量小：保留身份与人格表达，修漏逗号；可变 bio 事实放入样本上下文/实时配置，避免原样背诵 |
| preference | 输入 L1 topics 和 bio：主体保留；抽样核对事实混入，调整生成规则保持风格/价值判断 |
| diversity | 主要笔记事实来源：默认禁用无上下文的 infoRecall，降低 global（当前权重 4）等事实复述；保留表达多样性和情境判断 |
| 旧 context 类 | 当前未启用，不改造旧生成器；用上述共用格式构造少量检索依赖样本 |
| 通用风格指令 | 将固定“优雅、诗意”等风格改为用户真实表达特征 |

- 合并前标记类型、统计数量，避免旧事实样本残留被再次合并。
- 使用同问题、不同上下文、不同答案的样本，训练依赖参考材料。
- 优先在生成阶段控制，不新增复杂分类系统。
- 临时偏好可通过记忆/配置覆盖；仅稳定风格变化需要考虑重训。
- 调整 `resources/L2/data_pipeline/data_prep/subjective/config/config.json` 及生成器：减少无上下文 `unanswerable` 合成，不删掉所需的“检索无命中”负例。
- 移除类型时同步修改 diversity 中 `q_dict.pop('unanswerable')`/`pop('global')` 等假设，避免配置删除后 KeyError。先统计各类型输出占比，再比较精简前后数量。
- `_prepare_l2_data` 当前硬编码 `lang='English'`；改用用户已配置的训练语言，缺省明确处理。验证中文表达训练不会无故变成英文。

### 3.5 其他入口一致性

- `mcp/mcp_local.py`、`mcp/mcp_public.py` 的 assistant 回复改为 assistant 角色；本地请求显式传递检索 metadata。公共客户端仅在现有协议允许时传递，不假定外部服务已支持。
- `/api/talk/*` 验证默认策略链、前置 system、检索和多轮历史。
- `standalone/role/[roleId]/page.tsx` 修复 `role.enable_l1_retrieval || true`，尊重显式 false。
- advanced chat 向普通 DTO 传顶层 retrieval 字段的路径一并迁移到统一契约。

### 主要涉及文件

- `lpm_kernel/L2/l2_generator.py`
- `lpm_kernel/L2/data_pipeline/data_prep/` 下生成器与 prompts
- `lpm_kernel/L2/utils.py`、训练数据合并/格式化入口
- `lpm_kernel/api/domains/kernel2/dto/chat_dto.py`
- `lpm_kernel/api/domains/kernel2/services/chat_service.py`
- `prompt_builder.py`、`knowledge_service.py`、advanced chat 相关策略
- `message_builder.py`、`lpm_kernel/models/l1.py`、`lpm_kernel/kernel/l1/l1_manager.py`
- `lpm_kernel/file_data/embedding_service.py`
- `resources/L2/data_pipeline/data_prep/subjective/config/config.json`
- `lpm_kernel/file_data/` 中记忆索引更新路径
- `lpm_frontend/src/components/playground/` 及请求参数
- `lpm_frontend/src/app/standalone/role/[roleId]/page.tsx`、`mcp/mcp_local.py`、`mcp/mcp_public.py`、talk 路由

### 验收

固定模型和适配器，验证：

- [ ] 新增事实后可回答。
- [ ] 修改事实后采用新内容。
- [ ] 删除事实后不再作为已知内容。
- [ ] 无命中不伪造记忆。
- [ ] API 与 playground 行为一致。
- [x] Chroma 多条/空命中、真实 L1 shade、默认策略链、显式关闭、重复消息构建均通过。
- [ ] 本地 MCP、talk、角色页、advanced chat 的历史角色与检索开关一致；公共接口未验证部分标注。
- [x] tokenizer 后满足每 slot 预算；训练和推理使用同一上下文边界、位置及无命中格式。
- [x] 中文训练语言有效，删减 diversity 类型不会导致 KeyError。
- [ ] 对比风格一致性、事实准确率和训练数据量。

## 阶段 4：回归与交付

### 固定冒烟数据与效果对照

- 准备约 48 条人工合成、无真实私人数据的 `merged.json` fixture，固定 seed、语言与样本 ID；包含中英文、身份/偏好、多轮、短/长答案和模板边界。按来源分组拆分（例如 32 train、8 valid、8 test），改写样本不得跨集合。
- 阶段 1/2 从 fixture 直接准备数据，绕过在线合成/GraphRAG 全流程，以少量训练 step 验证链路，不用几十步 loss 推断最终质量。
- 阶段 3 另补相同拼装格式的命中、无命中、事实冲突、记忆更新/删除、检索材料内指令样本；固定检索结果可脱离外部 embedding 做快速契约测试，真实 Chroma 再做端到端验证。
- 比较三组结果：未微调 Qwen3、相同数据的短程 LoRA、轻量 L2 LoRA+检索。固定输入、采样参数与检索材料，区分模型换代、训练后端、数据策略的影响。
- 记录耗时、step/token 吞吐、内存统计口径、GGUF 大小、首 token 延迟；Mac 共享内存不能与 CUDA 专用显存直接比较。

### 检查命令与证据

实现时使用仓库已有工具，按阶段执行：

```bash
# 仓库根目录，使用已经确认隔离的 Poetry 环境
poetry run ruff check <本阶段修改的Python文件>
poetry run pytest <本阶段相关测试>
bash -n scripts/setup.sh docker/app/rebuild_llama_cuda.sh
git diff --check

# lpm_frontend 目录（仓库没有 npm run lint，使用实际脚本）
npx tsc --noEmit
npm run eslint:error
npm run build
```

- Ruff 当前 dev 依赖为 0.1.15，先使用项目版本，不以升级 lint 工具扩大改动。
- 检查仓库既有失败基线；新增失败必须解决，既有无关失败注明，不把一次全仓 lint 失败等同于本阶段失败或全部通过。
- Docker 在本机具备条件时验证实际构建；GPU 容器运行需要可用 GPU runtime，CUDA 机器未安装 nvidia-container-toolkit 时不能声称通过。
- 训练/融合/转换检查保留命令、退出码、关键日志、产物与验证结果；构建通过不能替代模型端到端验证。

| 范围 | 验证 |
| --- | --- |
| 配置 | 新默认值、旧配置读取、后端选择 |
| 数据 | 模板、loss mask、拆分、事实过滤 |
| 训练 | 短程 SFT、保存、融合、错误传播 |
| 推理 | GGUF、多轮、流式、thinking 设置 |
| 记忆 | 新增、修改、删除、无命中 |
| 平台 | Mac 实机、CPU 冒烟；CUDA 条件具备时实机验证 |
| 前端 | 类型检查、构建、模型选择、状态展示 |

使用固定评估样本分别比较措辞/句长/语气、偏好判断、最新事实遵从度，以及耗时、内存、产物大小和延迟。阶段 1/2 先保持数据策略相对稳定，阶段 3 再改变样本组成，便于归因。

只为关键行为添加有意义测试，不为简单文案堆测试。未实机验证的路径明确标注，不能将配置检查当作训练成功。

交付：代码、锁定依赖、启动/训练文档、实测内存表、阶段验证记录。无用户指令不提交推送。

## 实施依赖与停止条件

| 顺序 | 可交付结果 | 进入下一阶段的条件 |
| --- | --- | --- |
| 0 | 隔离环境与可重复安装路径 | 不影响 master，依赖/数据/端口隔离已验证 |
| 1 | Qwen3 配置、模板及 GGUF 推理 | 固定依赖组合、转换和 Metal 推理通过；平台限制明确 |
| 2 | MLX Web 完整训练链路 | HF 非量化基座短程训练→fuse→GGUF 实测通过，状态与取消可用 |
| 3a | 可用检索与共用消息格式 | 多条命中/L1/入口一致性/上下文预算通过 |
| 3b | 精简 L2 数据 | 检索依赖与无命中样本验证后，才降低事实复述权重 |
| 4 | 回归报告与文档 | 关键回归通过，未执行项和限制有据可查 |

依赖无可用解、融合不能导出、上下文预算丢失答案或检索仍为空时，先解决该阶段问题，不能标记完成或继续删训练数据。GPU 未实机验证可以交付明确标注的结果，但不能宣称该平台训练已通过。

## Review 采纳记录

| Review 条目 | 本次处理 |
| --- | --- |
| 1 工作区过时 | 已复核分支释放，在当前工作区创建 feat/2026-upgrade，删除工作区确认项 |
| 2 检索五个问题 | 已列为具体修复与验证任务，补充距离阈值、parallel slot 预算 |
| 3 训练/推理格式 | 先定义前置单条 system 共用格式，再生成检索样本 |
| 4 依赖跨度 | 首版选择 Transformers 4.x，移除硬编码 collator 依赖；不采信未经复核的最新版本号 |
| 5 Docker 路径 | 加入两个 Dockerfile、运行时重建脚本和 Makefile，统一 revision/包格式 |
| 6 模型/模板 | 推荐 4B-Instruct-2507 待用户确认；区分模板、保留旧型号，按固定运行时验证 thinking 参数 |
| 7 GPU 参数 | 明确修复 forced False、Metal offload 与 CUDA 环境变量覆盖 |
| 8 MLX 简化/契约 | 非量化 HF 基座优先；保留导出实测；补 wrapper/进度/可执行权限/参数白名单；iterations 计入累积语义 |
| 9 依赖/MPS | 采用平台 marker 和显式 CPU，保留依赖缺失预检；marker 不保证安装或运行成功 |
| 10 数据结论 | 旧 context 不改；selfqa/preference 主体保留但处理 bio 事实；配置删减同步修 pop，保留检索无命中负例及中文语言 |
| 11 其他入口 | 加入 MCP/talk/角色页/advanced chat；外部公共服务支持不作未经验证的承诺 |
| 12 验证方式 | 固定 fixture、真实项目检查脚本、环境变量隔离及验证记录 |

上述采纳是计划决策；相应功能是否实现和通过验证由执行记录说明。

## 外部服务与边界

- 数据合成可用本地 OpenAI 兼容网关（地址与模型由用户配置，不写入仓库）。
- Embedding：Cloudflare Workers AI `@cf/baai/bge-m3`，1024 维；凭据由用户配置，不读取、打印或写入其 key。
- CUDA 机器：私有远程主机，RTX 4070 Laptop 8GB、30GB RAM；安装任何东西前询问。Docker 尚无 nvidia-container-toolkit，不假定容器可用 GPU。

## 官方参考

- [Qwen3-1.7B 模型说明](https://huggingface.co/Qwen/Qwen3-1.7B)：Transformers 要求、原生模板及 thinking 控制。
- [MLX-LM](https://github.com/ml-explore/mlx-lm)：Apple Silicon 推理、LoRA 及量化模型微调能力。
- [Qwen3-4B-Instruct-2507 模型说明](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)：纯非思考模式与模板用法。
- [MLX fuse 源码](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/fuse.py)、[LoRA 入口](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/lora.py)、[dataset 实现](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/datasets.py)：融合、回调、prompt mask 与模板应用方式；当前 main 仅作能力参考，实施锁定 revision 后重验。

## 执行记录

- 2026-10-04 初版：完成只读代码评估与官方能力核对；保存计划。
- 2026-10-04 v2：结合 review，复核关键代码结论；在当前 worktree 创建并切换 feat/2026-upgrade；更新计划、任务与验收。尚未修改实现、安装/升级依赖、训练、提交或推送。4B 型号、Mac SFT-only 与启动实施仍待确认。
- 2026-10-04 开发：用户批准开始开发、授权并行，并明确确认 4B-Instruct-2507。建立工作区独立 .venv，固定候选训练栈；小 Qwen3+LoRA 的实际 CPU 单步训练通过，原生 1.7B tokenizer 的 completion mask 验证通过，固定 b6500 Metal 构建完成。完整 MLX/GGUF 和全量回归正在进行，尚未提交/推送。

- 2026-10-04 最终本机验收：完整 1.7B MLX 两步训练、Web service 训练/融合/转换及 Metal 普通/流式推理通过；真实 Chroma/SQLite 生命周期、43 项回归、依赖锁与前端构建通过。UI 已检查默认模型/MLX提示/进度，云端合成未经浏览器全流程点击；4B 完整训练、CUDA/Docker 与真实 Cloudflare/公共网络入口未实测。未完成的效果对照与平台实测保留待验收，不能由短程 loss 推断效果。详细指标见 2026-upgrade-validation.md。未 commit、未 push。
