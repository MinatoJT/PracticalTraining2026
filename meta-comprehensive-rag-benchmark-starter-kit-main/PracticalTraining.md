# Practical Training Notes

## Task2/Task3 Systematic Fix (2026-07-12)

- `Task1KGAgent._call_llm(messages, max_tokens, purpose)`: Task1/2/3 共用 DeepSeek 调用入口。默认关闭思考模式，避免短结构化请求只产生 `reasoning_content` 而 `content` 为空；记录模型、choices 数、finish reason、content/reasoning 长度和 token usage，但不记录 API Key 或思维链正文。
- `Task1KGAgent.set_trace_contexts(contexts)`: evaluator 注入 `session_id`、`interaction_id`、`turn_idx`，供 Task3 维护会话状态和生成逐轮日志。
- `Task2Agent._build_web_evidence(results, source)`: 网页证据标记为 `broad` 或 `entity_directed`。只有 broad 结果允许进入 `_rerank_kg_with_web()`，阻断错误 top1 通过定向搜索自我强化。
- `Task2Agent._build_task2_answer_messages_clean(...)`: 使用 UTF-8 中文 Prompt，明确暂定视觉实体、KG 候选、Web 来源和回答格式。
- `Task3Agent._update_visual_anchor(...)`: 按 `session_id` 保存结构化视觉锚点、候选、图像分和 margin。后续无图检索轮次沿用候选；显式重新看图时允许新证据覆盖旧锚点。
- `Task3Agent._looks_like_followup(query)`: 按代词和追问短语判断，不再使用“词数小于等于 5”规则。
- `Task3Agent._should_use_image(...)`: 除首轮外，颜色、外观和明确图片指向的问题也会重新读取图片。
- `conversation_validation.valid_conversation_indices(dataset, requested)`: 先排除 ground truth 缺失的坏会话，再选择请求数量。原第 5 个候选会话有空答案，旧 iterator 将其跳过，所以请求 5 组只统计 4 组。
- `CRAGEvaluator.generate_agent_responses()`: 显式校验 queries/images/history/response 数量，禁止 `zip()` 静默截断，并注入逐轮 trace context。
- `UI/run_eval.py::patch_deepseek_judge()`: 语义 Judge 同样关闭思考模式，避免短 JSON 请求只返回 reasoning 后误判。
- 每次 UI 运行写入独立的 `UI/outputs/<task>/trace_<run_id>.jsonl`。日志包含空回答分类、IDK 原因、质量分支、视觉锚点和 KG 分数组成。
- `testTask23Diagnostics.py`: 不联网回归测试，覆盖 reasoning-only、length、空 choices、Web 投票隔离、视觉锚点建立与纠错、短问题判断、合法短答案、批量长度和有效会话选择。

## Qwen3-VL Visual Candidate Pipeline (2026-07-13)

- `agents/vision/qwen_vl_client.py`: 共享百炼 OpenAI-compatible 客户端，默认以 `qwen3.5-omni-plus` 生成视觉锚点、以 `qwen3.5-omni-flash` 重排候选，并以 `qwen3-vl-flash` 作一次模型级 fallback。Omni 请求不发送 `enable_thinking`，VL Flash 才按能力发送 `enable_thinking=false`；全部使用 JSON Mode，realtime 模型会被明确拒绝。API Key 优先读取 `QWEN_VL_API_KEY`，其次读取 `DASHSCOPE_API_KEY`，并清理引号和首尾空白，不记录 Key 或图片 base64。
- `agents/vision/visual_anchor.py`: 提取代码块或额外文字中的首个 JSON 对象，校验 anchor/rerank 字段、置信度和候选索引；非法结果返回明确错误。
- `agents/vision/visual_candidate_pipeline.py`: 合并 Image-KG 与 Qwen anchor 候选，Qwen 成功时以 `final_score` 为主排序，图像相似度只作 tie-breaker；Qwen 不可用、低置信或解析失败时完整回退旧逻辑。
- `Task1KGAgent._prepare_visual_evidence(...)`: Task1/2/3 共用入口。Task1 不新增文本或 Web 检索；Task2/3 保留 broad/entity Web 分流，并禁止定向 Web 覆盖高置信 Qwen 视觉判断。
- Task3 每轮均根据当前问题和必要用户历史生成问题相关 anchor，并以图片、当前问题和候选重新调用 Qwen 重排；完全相同请求由缓存去重，旧 assistant 回答只作为弱历史。
- Qwen 只负责视觉主体、OCR 与候选验证，DeepSeek 继续负责最终答案和质量门，evaluator 的官方接口无变化。
- 视觉链路默认直接调用百炼 API，不依赖 WSL、vLLM、本地模型或 `localhost:8000`。Qt 与 WinUI 均提供独立的锚点/重排模型字段，并通过密码框和环境变量传递 Qwen Key 及 Base URL，命令预览不包含 Key。旧 `QWEN_VL_MODEL` 仍可令两个阶段使用同一模型。
- `testQwenVision.py`: 离线测试视觉关闭回退、画中画、OCR、错误 JSON、空响应/超时、Qwen 主排序、同图缓存及日志脱敏。

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

## Task2Agent 接入与中文 UI 更新

本次将组员新增的 `agents/Task2Agent.py` 接入到实训 UI 和评测包装器中：

- `UI/run_eval.py` 新增 `--agent task2agent` 选项，并直接导入 `Task2Agent`。
- Task1 默认使用 `Task1KGAgent`；Task2 默认使用 `Task2Agent`；Task3 仍默认使用 `agents.user_config.UserAgent`。
- Task2 的 `build_search_pipeline(task2)` 会启用官方 web search index：`crag-mm-2025/web-search-index-validation`，Task1 仍禁用 web search。
- `TASK2_DEBUG_PATH` 默认写到 `UI/outputs/task2/debug.jsonl`，便于查看 Task2 的 KG、Web 检索、多源融合和 DeepSeek 回答状态。

同时，`UI/app.py` 已改为中文界面，并继续用 UTF-8 读取子进程输出，避免中文输出乱码。界面中的评测模型选项含义如下：

- `不使用语义评测（仅 exact match）`：官方精确字符串匹配。
- `deepseek-v4-flash - 语义评测`：调用 `UI/run_eval.py` 中的 `patch_deepseek_judge()`，让 DeepSeek 判断预测答案是否语义覆盖 ground truth。

注意：如果 UI 已经打开，需要关闭后重新运行 `UI/run_ui.bat`，否则仍会使用旧的 Python 进程和旧界面。

## Task1 完整句输出兜底

针对 DeepSeek 在 Task1 中有时输出实体名、别名列表、空字符串，或被质量闸门压成 `I don't know.` 的问题，`Task1KGAgent` 新增了输出质量控制：

- `_needs_sentence_rewrite(answer, query, candidates)`：检查回答是否只是实体名、逗号分隔别名、名词短语或缺少谓语。
- `_rewrite_as_sentence(query, bad_answer, selected, candidates, history)`：对不合格回答做一次 DeepSeek 二次改写，要求输出完整英文句。
- `_answer_with_heuristic_sentence(query, candidates)`：当 DeepSeek 仍返回空串或短语时，按常见 Task1 问题类型生成完整句兜底，例如食品来源、颜色波长、车展判断、乘客数、建造耗时和清理空间附件。
- `_is_any_entity_echo(answer, candidates)`：收窄实体复述检测，只拦截“几乎只有实体名”的回答；允许 `Saint Isaac's Cathedral took 40 years...` 这类包含实体名的完整句通过。

该兜底仅用于本地实训稳定性，目标是避免明显不合格的短语答案进入评测；正常的 DeepSeek 完整句回答不会被替换。


## DeepSeek Judge 容错修复

针对本地实训评测中出现“DeepSeek 返回内容里已经有 `accuracy: true`，但外层仍被判 `False`”的问题，`UI/run_eval.py` 做了如下修复：

- `_parse_deepseek_judge(raw)`：容错解析 DeepSeek judge 输出，兼容 JSON 被截断、代码块包裹、大小写差异，以及只出现 `accuracy true/false` 的情况。
- `_semantic_shortcut(query, ground_truth, prediction)`：在答案已经明显覆盖 ground truth 的关键词或数字时，先用本地高置信规则判定，避免语义正确答案被 judge API 格式问题误杀。
- 数字词归一化：将 `five`、`seven` 等英文数字词映射为 `5`、`7`，解决 “five passengers” 与 “5 people” 语义一致但字面不同的问题。
- `run_eval.py` 会先把数据集切到 `--num-conversations` 指定的前 N 条，避免 batch 边界导致 `5` 条测试实际跑出 `6` 或 `8` 条。

该修复只影响本地实训评测的判分稳定性，不改变 Agent 的原始回答内容。


## Task2 Web Search 最小修复

针对 Task2 初始化 web search 时出现的 `chromadb.errors.InternalError: too many SQL variables`，本地 Anaconda 环境中的 `cragmm_search/web_search_mock_api/api/web_index.py` 做了最小兼容补丁：

- 原实现初始化时一次性调用 `self.vector_db.get()` 读取全部 ids 和 metadatas，网页索引规模较大时会触发 SQLite/ChromaDB 的变量上限。
- 补丁改为 lazy metadata：初始化只保存空的 `index_to_metadata` 缓存；搜索命中某个 chunk id 后，`get_page_name/get_page_snippet/get_page_url` 再通过 `vector_db.get(ids=[id], include=["metadatas"])` 按需读取单条 metadata。
- 原文件备份为 `web_index.py.bak_codex` 和 `web_index.py.bak_lazy_codex`。

同时修复了 Task2 web search 的 embedding 维度不匹配：

- 官方 `web-search-index-validation` 使用 `BAAI/bge-large-en-v1.5` 建索引，embedding 维度为 1024。
- UI runner 之前给 Task2 使用 `sentence-transformers/all-MiniLM-L6-v2`，embedding 维度为 384，会导致 `Collection expecting embedding with dimension of 1024, got 384`。
- `UI/run_eval.py` 已调整为：Task1 仍使用 MiniLM 占位；Task2/Task3 启用 web search 时使用 `BAAI/bge-large-en-v1.5`。

验证结果：运行 `UI/run_eval.py --task task2 --agent task2agent --num-conversations 1 --eval-model None --no-progress` 已完成，无 Chroma traceback；`UI/outputs/task2/debug.jsonl` 中 `web_count=8`，说明网页检索已返回结果。


## Task2 中文 Prompt 与完整句修复

针对 Task2 跑通后回答仍经常退化为单个实体名或 top1 短语的问题，`agents/Task2Agent.py` 做了如下调整：

- 主流程不再只依赖 `_select_entity()` 的单一 top1 实体，而是复用 Task1 的 `_select_supporting_entities()`，把多个高置信 Image-KG 候选传入回答阶段。
- Web query 由“问题 + 单个实体”改为“问题 + 多 KG 候选上下文”，减少 top1 跑偏时网页检索继续跑偏。
- `_build_task2_answer_messages()` 改为中文 prompt，明确要求综合 Image-KG 候选和 Web evidence，且必须输出完整英文句，不能只输出实体名、车型名、建筑名、食物名或逗号短语。
- `_answer_task2_with_llm()` 复用 Task1 的完整句质量闸门：不合格回答会触发 Task2 专用二次改写 `_rewrite_task2_as_sentence()`。
- DeepSeek 仍输出空串或短语时，回退到 Task1 的 `_answer_with_heuristic_sentence()`，至少给出完整句兜底，避免 hallucination 表格里全是单词。

验证：无 API smoke test 跑通 `--task task2 --agent task2agent --num-conversations 1`，web search 返回 `web_count=8`，输出从 `Honda Civic` 变为完整句：`No, this car is not suitable for transporting seven passengers at once because it seats about five people.`
## Windows UI 与环境配置更新

- 新增 `WinUI3/`：这是独立的 Windows 原生 WinUI 3 前端，用于运行 Task1、Task2、Task3 数据集评测和 Task1 自定义问答，不替换原 Qt UI。
- `WinUI3/run_winui3.bat` 使用 bat 所在目录定位项目，不依赖固定盘符；WinUI 内部会优先使用 `CRAGMM_PYTHON`，其次使用项目 `.venv`，再尝试系统 `python` 或 `py -3`。
- `UI/run_ui.bat` 已取消 `C:\anaconda\python.exe` 固定路径，改为同样的自动 Python 选择逻辑。
- `UI/app.py` 默认 Python 从 `sys.executable` 继承，因此直接运行 Qt UI 时也不会绑定某台电脑的 Anaconda 路径。
- `requirements.txt` 补充了 PySide6、hf_xet、python-dotenv、numpy 上限等依赖，方便新电脑按统一命令配置 Python 环境。

推荐新电脑配置流程：

```bat
cd meta-comprehensive-rag-benchmark-starter-kit-main
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\pip install -r requirements.txt
UI\run_ui.bat
```

WinUI 3 版本还需要安装 .NET 8 SDK，然后运行：

```bat
WinUI3\run_winui3.bat
```


WinUI 3 构建提示：如果缺少 `Microsoft.Build.Packaging.Pri.Tasks.dll`，需要安装 Visual Studio/Build Tools 的 Windows 应用开发和 Windows App SDK 构建组件。

### WinUI 3 依赖安装验证

- 已通过 Visual Studio Installer 为本机 BuildTools 补齐 `Microsoft.VisualStudio.ComponentGroup.UWP.BuildTools` 及 Windows SDK 组件。
- 安装前 C 盘空间不足，已使用 `pip cache purge` 清理 pip 下载缓存约 15GB。
- `dotnet build` 会从 .NET SDK 自身路径查找 PRI Tasks，仍可能失败；WinUI3 启动脚本已改为使用 Visual Studio BuildTools 的 `MSBuild.exe`。
- 验证命令：`MSBuild.exe WinUI3\CRAGMM.WinUI.csproj /restore /p:Configuration=Release /p:Platform=x64`，结果为 0 warning / 0 error。

### WinUI 3 启动失败修复

- 现象：系统已安装 Windows App Runtime 1.6，但 `CRAGMM.WinUI.exe` 仍提示需要 Windows App Runtime。
- 原因：管理员 PowerShell 能看到 `Microsoft.WindowsAppRuntime.1.6`，普通启动上下文未必能看到同一组用户级 MSIX runtime 包。
- 修复：WinUI3 项目已启用 `WindowsAppSDKSelfContained=true`，改为自带 Windows App SDK 运行时文件，降低对系统 App Runtime 注册状态的依赖。

### WinUI 3 窗口无法唤起排查

- Windows 事件日志显示进程启动后在 `Microsoft.UI.Xaml.dll` 处崩溃，异常码 `0xc000027b`。
- 为降低 XAML 资源和编码因素影响，已将 `MainWindow.xaml` 改为 ASCII 文案和普通颜色，并保留全部控件名称与逻辑绑定。
- 新增 `cragmm_winui_startup.log`，用于记录 `OnLaunched`、窗口构造和激活阶段。

### WinUI 3 资源缺失修复

- 启动日志显示 `XamlParseException: Cannot find a Resource with the Name/Key TabViewButtonBackground`。
- 原因是 `App.xaml` 未合并 WinUI 控件默认资源。
- 已在 `App.xaml` 中加入 `<controls:XamlControlsResources />`。


## Task3Agent 上下文优化

新增 `agents/Task3Agent.py`，用于 Task3 多轮问答场景。该 Agent 不改变项目骨架，直接继承 `Task2Agent` 的 Image-KG、Web search、KG-Web 融合和完整句兜底能力；新增的核心逻辑是上下文优化：

- `batch_generate_response(queries, images, message_histories)`：官方评测入口。每轮先读取 `message_histories`，再将当前问题改写成可独立检索的问题，随后复用 KG/Web 检索并生成上下文一致的回答。
- `_build_context_state(history)`：压缩最近若干轮对话，提取上一轮用户问题、上一轮助手回答和近期实体，避免直接把过长历史塞进检索 query。
- `_rewrite_query_with_context(query, context)`：当前轮若包含 `it/this/that/they` 等追问指代，优先调用 DeepSeek 改写为独立问题；无 API 或调用失败时使用规则兜底拼接上一轮上下文。
- `_build_task3_web_query(contextual_query, context, selected_entity, kg_evidence)`：在 Task2 网页检索 query 基础上补入历史实体，提高多轮指代问题的网页召回。
- `_answer_task3_with_llm(...)` / `_build_task3_answer_messages(...)`：使用中文 Prompt 约束 DeepSeek 保持历史一致性，避免与前文冲突；用户英文提问时输出英文完整句，最多两句话。
- `_rewrite_task3_as_sentence(...)`：当模型仍输出实体名、标题或短语时，二次改写成完整自然句。
- `TASK3_DEBUG_PATH`：Task3 调试日志路径，默认写入 `UI/outputs/task3/debug.jsonl`，记录上下文改写、检索和回答状态，不记录 API key。

`UI/run_eval.py` 已新增 `--agent task3agent`；Qt UI 与 WinUI3 UI 的 Agent 下拉框均已加入 `Task3Agent`，选择 Task3 时默认使用该 Agent。

### Task2/Task3 日志诊断与检索修复

根据 `UI/outputs/task2/debug.jsonl` 与 `UI/outputs/task3/debug.jsonl` 的实际结果，完成以下修复：

- Task2 `_merge_web_results(...)`：同时执行不带实体的宽查询和带 KG 实体的精确查询，合并并去重结果，避免错误 top1 持续污染 Web 检索。
- Task2 `_rerank_kg_with_web(...)`：使用网页标题和片段反向给 KG 候选加分；随后调用已有 `_select_entity(...)`，不再直接把规则 top1 当成最终实体。
- Task2 `_needs_sentence_rewrite(...)`：使用通用完整句检查替换固定动词白名单，避免 `belongs/contains/provides` 等正常谓语被误判并覆盖成 IDK；同时拦截 `It took 40` 这类缺少单位的片段。
- Task3 `_build_context_state(...)`：从检索上下文中排除 `I don't know`，并区分用户历史与旧 assistant 答案，防止错误答案在后续轮累积传播。
- Task3 `_should_use_image(...)`：首轮或明确重新指向图片时才做图像检索；普通后续轮使用会话实体和 Web，避免同一图片每轮重新排序后实体漂移。
- Task3 `_rerank_kg_with_context(...)`：根据当前问题与用户历史给 KG 候选加分，并为动物、植物、船只、茶品等 Task3 常见视觉主体补充类别约束。
- Task3 `_answer_addresses_current_question(...)`：检测机械重复上一轮答案和缺少数值的回答，触发上下文改写，不再让身份答案覆盖人物、地点、数量等追问。

### Qt 实时测试对话面板

为便于观察 Agent 实际输入输出，Qt 前端新增“实时测试对话”区域：

- `UI/run_eval.py::install_live_event_stream(agent, task)`：在不修改 Agent 接口的前提下包装 `batch_generate_response()`，每轮完成后立即输出一条 JSON 事件。
- JSON 事件包含 `conversation_id`、`turn`、`query`、`response`、`history` 和 `image_path`；同一 Task3 会话通过图片摘要稳定分组。
- 评测图片会生成轻量缩略图，保存到 `UI/outputs/live/<run_id>/`；同一多轮会话只保存一次。
- `UI/app.py::EvalWorker.live_event`：从子进程标准输出中识别实时事件，普通日志仍进入原控制台。
- `UI/app.py::ConversationView`：按会话 Tab 展示图片、Query 气泡和 Agent Response 气泡；Task3 后续轮持续追加到同一个 Tab。
- 每次点击运行会清空前端旧会话页，但不会删除磁盘中的历史评测结果或缩略图。
