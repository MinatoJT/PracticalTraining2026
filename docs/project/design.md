# 软件设计与接口说明

文档编号：PT2026-SDD；版本 0.1；状态：待评审。下列结构来自固定基线源码，不代表已完成系统验证。

## 1. 模块与调用链

工程根为 `meta-comprehensive-rag-benchmark-starter-kit-main/`。

```text
Qt / WinUI3 / CLI
  -> UI/run_eval.py（数据集、检索器、Agent 选择、评测包装）
  -> newagents/agents.py（Task1KGAgent / Task2Agent / Task3Agent）
  -> BlackPearlPipeline
     -> 问题规划与历史指代 -> 图像 KG / 可选 Web 检索
     -> 证据短名单 -> 视觉重排 -> 证据门控
     -> DeepSeek 回答 -> 按条件复核 -> 答案门控 / 拒答
  -> CRAGEvaluator -> 逐轮 CSV / 汇总 JSON / JSONL trace
```

| 模块 | 责任与边界 |
|---|---|
| `newagents/agents.py` | 三个任务共用 `_CRAGMMAgent` 包装器，校验配置 task 与类匹配；三个任务类不互相继承 |
| `config.py` | `for_task()` 决定 Web、历史开关，并从环境取默认值和范围约束 |
| `pipeline.py` | 组合检索、规划、排序、门控与复核；图像缓存、会话状态和统计 |
| `retrieval/image_kg.py`、`retrieval/web.py` | 调用传入的搜索 pipeline，标准化证据；Web 查询与去重 |
| `providers/qwen.py`、`qwen_local.py` | 视觉规划、候选重排及部分答案核查；远程/本地后端 |
| `providers/deepseek.py` | 回答与条件复核、解析、降级 |
| `reasoning/evidence.py` | 词面/检索/重排分数融合及证据门控 |
| `conversation.py`、`schemas.py` | 历史处理、拒答识别及结构化记录 |
| `tracing.py` | 可选追加 JSONL；不保证日志写入一定成功 |

旧 `agents/` 仍提供 BaseAgent、基线和旧 Task1–3；`UI/run_eval.py` 的 `legacy_task*` 从这里导入。`Legacy失败/` 是另一套历史快照，不是这些选项的导入来源。

## 2. 主要接口

| 接口 | 输入 / 输出 | 约束与异常 |
|---|---|---|
| `Task*Agent(search_pipeline, config=None, vision_provider=None, answer_provider=None)` | 可注入检索与模型提供方 | config.task 不匹配抛 ValueError；用于离线替身测试 |
| `get_batch_size()` | 返回整数 1 | 表示当前代理期望的批量 |
| `batch_generate_response(queries, images, message_histories)` | 三组等长列表 → `List[str]` | 输入不等长抛 ValueError；按索引调用流水线 |
| `set_trace_contexts(contexts)` | 字典列表，evaluator 提供 session_id、interaction_id、turn_idx 等 | 与当前批次索引对应，不应串用旧上下文 |
| `inspect_retrieved_evidence(image)` | 图像 → 标准证据列表 | 用于自定义问答/检查入口 |
| 搜索对象 `search_pipeline(value, k=...)` | 图像或查询文本 → 搜索结果列表 | Task1 不用文本 Web 分支；检索是模拟索引，不是实时公网搜索 |

## 3. 内部数据

- `QueryPlan`：原始/独立问题、问题类型、视觉主体、OCR 文本、搜索词、是否 Web/重看图、置信度。
- `EvidenceItem`：eid、source、title、text、url、attributes、检索/词面/重排/最终分数及查询信息。
- `RerankDecision`：可回答性、主体、置信度、证据 ID 对应得分等。
- `AnswerDecision`：答案、可回答性、置信度、evidence_ids、knowledge_used、缺失信息等。

上述是 Python 内部结构，不是已版本化 HTTP 契约。提供方返回 JSON 必须经解析；JSON 可解析不能证明内容真实。

## 4. 任务差异、门控与状态

`AgentConfig.for_task()` 对 Task1 关闭 Web 和历史，对 Task2 开 Web、关历史，对 Task3 同时开启。默认最多 3 条搜索查询，但环境变量可配置至 5，不能把 README 的“3 条”当作固定上限。

证据门控检查空集合、最高证据分和特定高置信不可回答判断；最终答案门控检查答案/置信度、有效 evidence_ids。默认允许稳定知识的情况下，`knowledge_used` 可满足无有效 evidence_ids 的放行条件。视觉核查产生的证据也不等同于 KG/Web 原始事实。不能以“evidence-first”名称保证零幻觉。

图像证据缓存按 RGB 图像尺寸与像素哈希索引，容量 24；会话状态用于首轮锚点复用与追问处理。需要验证会话隔离、重新看图和上下文改变，不应只验证重复图片缓存命中。流水线异常可导致拒答，必须结合 trace 分清真正信息不足和运行故障。

## 5. 外部依赖与安全限制

需要 CRAG 检索库、Hugging Face 数据/索引缓存、Pillow，以及所选模型后端。API 模式涉及 Qwen/DashScope 和 DeepSeek；模型 ID 和服务可用性是运行时依赖，配置中存在某个名称不证明服务当前可用。

`TraceWriter` 仅过滤顶层名称包含 key/base64 或特定 image 名称的字段，且写入异常被忽略；它不是递归脱敏或可靠审计存储。问题、证据文本、嵌套对象可能仍含敏感信息；发布任何日志前人工检查，不写入真实密钥、图片 base64 或用户私有内容。
