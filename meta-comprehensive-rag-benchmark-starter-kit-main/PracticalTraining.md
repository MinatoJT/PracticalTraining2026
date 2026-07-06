# Practical Training Notes

## Task1KGAgent

新增 `agents/Task1KGAgent.py`，用于 Task1 单源增强：

1. 输入图像后调用官方图像检索模拟 API：`search_pipeline(image, k=5)`。
2. 读取返回的相似图像、实体名和 `entity_attributes`。
3. 清洗 HTML / Wiki 标记后形成结构化证据。
4. 有 `DEEPSEEK_API_KEY` 时调用 DeepSeek 文本模型基于 KG 证据回答；没有 key 或 API 失败时使用规则兜底。

## Qt UI

新增 `UI/` 文件夹，包含：

- `UI/app.py`：PySide6 前端。
- `UI/run_eval.py`：不修改项目骨架的评测包装器。
- `UI/run_ui.bat`：Windows 启动脚本。

UI 支持选择 Task1/Task2/Task3、Agent、评测数量、展示样例数量和评测模型。Task1 默认使用 `Task1KGAgent`；Task2/Task3 默认使用 `agents.user_config.UserAgent`。

## 自定义 Task1 问答模式

UI 新增 Custom Task1 question 模式。用户选择本地图片并输入问题，程序调用 UI/custom_task1.py，通过官方图像检索模拟 API 获取相似图片及 KG 结构化证据，再由 Task1KGAgent 生成答案。


## Dataset 缓存目录

为避免用户目录权限、中文路径或缓存损坏影响检索索引加载，UI 将 HF_HOME、HF_DATASETS_CACHE、HUGGINGFACE_HUB_CACHE、HF_XET_CACHE、TRANSFORMERS_CACHE、SENTENCE_TRANSFORMERS_HOME、CRAG_CACHE_DIR 和 CRAG_WEBSEARCH_CACHE_DIR 统一指向项目根目录下的 Dataset/ 子目录。


## ChromaDB metadata 分批读取补丁

本地 Anaconda 环境中的 `cragmm_search/image_search_mock_api/image_kg.py` 已做兼容补丁：原实现一次性读取全部 metadatas，在当前 ChromaDB/SQLite 组合下会触发 `too many SQL variables`。补丁改为按 1000 条分批读取 metadata，并按 Chroma 返回的 id 放回 `id2_data`，保持后续 `id2_data[image_id]` 访问逻辑不变。原文件备份为 `image_kg.py.bak_codex`。

## Gated Llama tokenizer 替代

官方 `local_evaluation.py` 会尝试下载 gated 的 `meta-llama/Llama-3.2-1B-Instruct` tokenizer 用于 75 token 截断。UI 包装器 `UI/run_eval.py` 已替换为本地简单 tokenizer，只用于 UI 实训评测，避免 HuggingFace 401 授权错误。

## Task1KGAgent 实体选择最小版本接口说明

当前 `Task1KGAgent` 的 Task1 流程为：`image + question -> image search KG -> 候选实体清洗 -> 规则重排 -> DeepSeek 选择实体 -> DeepSeek 基于选中实体回答`。

主要接口如下：

- `batch_generate_response(queries, images, message_histories)`：CRAG-MM 官方评测调用入口。对每条样本完成图像检索、实体选择和答案生成。
- `_image_search(image)`：调用官方 Task1 图像检索模拟 API，即 `self.search_pipeline(image, k=self.top_k)`，返回相似图像及 KG 实体。
- `_build_evidence(results)`：把图像检索原始结果压平成候选实体列表，保留 `score`、`entity_name`、`attributes` 和 `source_url`。
- `_rank_candidates_by_rules(query, evidence)`：规则重排入口。根据问题类型和实体属性对候选实体重新排序，避免盲信 image search 的 top-1。
- `_score_candidate_by_rules(query, candidate)`：候选实体规则打分。车辆、食物、建筑、动物、安全、颜色等问题会触发不同关键词加权。
- `_select_entity(query, ranked_evidence)`：实体选择总入口。优先让 DeepSeek 在规则重排后的前若干候选中选择实体；失败时回退到规则最高分实体。
- `_select_entity_with_llm(query, candidates)`：DeepSeek 实体选择接口。要求模型返回 JSON，如 `{"index": 1, "confidence": 0.7, "reason": "..."}`。
- `_parse_entity_selection(raw, candidates)`：解析 DeepSeek 返回的 JSON，并映射回候选实体。
- `_answer_with_llm(query, selected, candidates, history)`：DeepSeek 回答接口。只围绕选中的实体和其 KG 属性回答，减少多个候选导致的犹豫或全 IDK。
- `_build_answer_messages(query, selected, candidates, history)`：构造最终回答 prompt，包含选中实体、选中实体属性、少量备选实体和历史上下文。
- `_answer_with_rules(query, evidence)`：无 API key 或 API 调用失败时的规则兜底，尽量从 KG 字段直接抽取答案。
- `_find_attribute(evidence, attr_terms)`：按字段名模糊匹配 KG 属性，例如 `architect`、`floor_count`、`opening` 等。
- `_clean_text(text)` / `_clean_attributes(attrs)`：清洗 KG 中的 HTML、Wiki 链接和模板标记。
- `_debug(payload)`：写入调试日志，不记录 API key，只记录检索实体、实体选择和 LLM 调用状态。

该版本仍是最小实训版本，没有引入 YOLO 或额外视觉模型。其主要改进点是把“直接相信图像检索 top-1”改为“问题感知实体重排 + LLM 候选实体选择”。

## 实体选择与中文 Prompt 调整

针对前一版出现的两个问题：DeepSeek 实体选择 JSON 返回空、最终回答经常只复述实体名，`Task1KGAgent` 做了如下调整：

- `_build_entity_selection_messages(query, candidates)`：改为中文短提示，只要求 DeepSeek 返回 `INDEX: 数字`，不再要求 JSON，降低空回复概率。
- `_parse_entity_selection(raw, candidates)`：解析 `INDEX: 数字`、`index: 数字`、`索引: 数字` 等简单格式；解析失败时仍回退到规则最高分实体。
- `_build_answer_messages(query, selected, candidates, history)`：改为中文回答 prompt，明确要求回答“事实、判断、数值、日期、来源或安全建议”，不要只复述实体名。
- `_is_entity_echo(answer, selected)`：检测模型输出是否只是实体名。
- `_repair_entity_echo(query, selected, candidates, history)`：如果模型只输出实体名，则二次追问，强制其回答问题本身。

这些改动仍不引入额外视觉模型，保持 Task1 最小实现：官方图像检索 API + KG 实体候选 + DeepSeek 文本推理。

## DeepSeek 语义评测选项

默认 `Eval model=None` 时，官方 evaluator 只做 exact match：只有预测答案字符串与 ground truth 完全一致才算正确。因此简短但语义正确的答案也会被判 `INCORRECT`，并且表格中的 `API Response` 会显示 `None`。

UI 已新增 `deepseek-v4-flash - semantic judge` 选项。选择该选项后，`UI/run_eval.py` 会用 DeepSeek 作为语义评测器，判断 `Prediction` 是否覆盖 `Ground truth` 的关键信息。评测 prompt 要求 DeepSeek 返回 JSON：`{"accuracy": true/false, "reason": "..."}`。

注意：这只改变本地实训评测方式，不改变官方比赛评测逻辑。
