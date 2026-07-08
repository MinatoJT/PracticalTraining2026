# Task2Agent.py 说明文档

## 1. 文件定位

`Task2Agent.py` 是在你已有的 `Task1KGAgent.py` 基础上扩展出来的 **Task2 多源增强 Agent**。

核心原则是：

```text
不修改 Task1 代码
不重写 Task1 的主体逻辑
不复制 Task1 的完整实现
只通过继承 Task1KGAgent，在外层新增 Task2 所需能力
```

继承关系如下：

```python
from agents.Task1KGAgent import Task1KGAgent

class Task2Agent(Task1KGAgent):
    ...
```

因此，Task2 的基础能力仍然来自你自己的 Task1：

```text
图片输入
↓
图像检索 image search
↓
mock KG 实体解析
↓
KG 候选实体重排
↓
DeepSeek 实体选择
↓
基于 KG 的回答生成
```

Task2 在此基础上新增：

```text
网页检索 Web Search
↓
网页证据解析
↓
网页噪声过滤
↓
KG + Web 多源证据融合
↓
多源增强 Prompt
↓
DeepSeek 最终回答
```

---

## 2. 与 Task1 的关系

### 2.1 Task1 保持不动

`Task2Agent.py` 没有修改 `Task1KGAgent.py` 中的任何函数。

Task2 直接复用 Task1 中已有的这些方法：

```python
self._image_search(image)
self._build_evidence(raw_image_results)
self._rank_candidates_by_rules(query, kg_evidence)
self._select_entity(query, ranked_kg)
self._answer_with_rules(query, ranked_kg)
self._finalize_answer(answer)
self._format_history(history)
self._clean_text(text)
```

这些能力仍然由 Task1 负责。

Task2 不改变 Task1 的图像检索、KG 解析、实体选择和规则兜底逻辑。

---

### 2.2 Task2 是 Task1 的外层扩展

Task2 的主流程是：

```python
raw_image_results = self._image_search(image)
kg_evidence = self._build_evidence(raw_image_results)
ranked_kg = self._rank_candidates_by_rules(query, kg_evidence)
selected_entity = self._select_entity(query, ranked_kg)

web_query = self._build_web_query(query, selected_entity, ranked_kg)
raw_web_results = self._web_search(web_query)
web_evidence = self._build_web_evidence(raw_web_results)

ranked_web = self._rank_web_evidence(
    query=query,
    web_query=web_query,
    web_evidence=web_evidence,
    selected_entity=selected_entity,
    kg_evidence=ranked_kg,
)

fused_context = self._fuse_multisource_evidence(
    query=query,
    selected_entity=selected_entity,
    kg_evidence=ranked_kg,
    web_evidence=ranked_web,
)
```

也就是说：

```text
Task1 负责“图片 → KG”
Task2 负责“KG → Web 检索 → 多源融合 → 回答”
```

---

## 3. Task2Agent.py 的整体功能

`Task2Agent.py` 对应赛题中的 **Multi-source Augmentation，多源增强任务**。

题目要求是：

```text
在 Task1 单源增强基础上，增加网页检索模拟 API。
网页内容可能包含答案，也可能包含噪声。
模型需要筛选信息、融合多源证据，并生成准确回答。
```

所以这个文件实现了以下功能：

| 功能模块 | 对应方法 | 作用 |
|---|---|---|
| 复用 Task1 图像检索 | `_image_search()` | 从图片检索相似图像和 KG 实体 |
| 复用 Task1 KG 解析 | `_build_evidence()` | 将 KG 原始结果整理成候选实体 |
| 复用 Task1 实体选择 | `_select_entity()` | 选出最可能对应图片主体的实体 |
| 新增 Web Query 构造 | `_build_web_query()` | 根据用户问题和 KG 实体构造网页检索 query |
| 新增网页检索 | `_web_search()` | 调用允许的 web search API |
| 新增网页解析 | `_build_web_evidence()` | 提取网页标题、URL、snippet |
| 新增网页证据过滤 | `_rank_web_evidence()` | 对网页结果排序并去噪 |
| 新增网页评分 | `_score_web_evidence()` | 根据问题、实体、KG 属性匹配程度评分 |
| 新增多源融合 | `_fuse_multisource_evidence()` | 整合 KG 证据和 Web 证据 |
| 新增 Task2 回答 | `_answer_task2_with_llm()` | 调用 DeepSeek 生成多源增强答案 |
| 新增 Task2 Prompt | `_build_task2_answer_messages()` | 明确 KG 和 Web 的证据优先级 |
| 新增调试日志 | `_debug_task2()` | 记录 Task2 的检索、过滤和回答状态 |

---

## 4. 主要修改 / 新增内容说明

这里的“修改”指的是相对于 Task1 的新增扩展，不是修改 Task1 源码。

---

### 4.1 新增 `Task2Agent` 类

```python
class Task2Agent(Task1KGAgent):
```

含义：

```text
Task2Agent 继承 Task1KGAgent。
Task2Agent 自动拥有 Task1 的全部方法。
Task2Agent 只新增 Task2 需要的 Web 检索和多源融合功能。
```

---

### 4.2 新增 Task2 初始化参数

```python
web_top_k: int = 8
web_keep_top_n: int = 4
min_web_score: float = 0.08
```

含义：

| 参数 | 含义 |
|---|---|
| `web_top_k` | 网页检索时最多取多少条原始结果 |
| `web_keep_top_n` | 过滤后最多保留多少条网页证据 |
| `min_web_score` | 网页证据最低相关性分数 |

初始化后保存为：

```python
self.web_top_k = web_top_k
self.web_keep_top_n = web_keep_top_n
self.min_web_score = min_web_score
```

---

### 4.3 将 `get_batch_size()` 改为 1

```python
def get_batch_size(self) -> int:
    return 1
```

原因：

Task2 每条数据会比 Task1 多做：

```text
1. 网页检索
2. 网页证据过滤
3. 多源增强 Prompt
4. DeepSeek 多源回答
```

如果 batch 太大，API 调用可能等待过久。  
所以 Task2 先设置为 `1`，优先保证稳定运行。

这不是修改 Task1 的 batch size，而是 Task2 自己的运行策略。

---

### 4.4 重写 `batch_generate_response()`

Task2 必须重写主流程，因为 Task1 的主流程只做单源 KG 增强。

Task2 的主流程为：

```text
1. 调用 Task1 的图像检索和 KG 解析；
2. 调用 Task1 的候选实体重排和实体选择；
3. 根据问题 + KG 实体构造 Web Query；
4. 调用 Web Search；
5. 解析网页结果；
6. 过滤网页噪声；
7. 融合 Image-KG 和 Web Evidence；
8. 调用 DeepSeek 生成最终回答；
9. 如果 DeepSeek 不可用，回退到 Task1 的规则回答。
```

这部分是 Task2 的核心。

---

### 4.5 新增 `_build_web_query()`

作用：构造网页检索 query。

如果只搜索用户原问题，很多问题会缺少图片主体信息。

例如：

```text
用户问题：What is its manufacturer?
```

如果直接搜：

```text
What is its manufacturer?
```

网页检索会完全不知道 `its` 指什么。

所以 Task2 会把 Task1 选中的实体拼进去：

```text
What is its manufacturer? Ferrari 458
```

核心逻辑：

```python
parts = [self._clean_query_text(query)]

if selected_entity and selected_entity.get("entity_name"):
    parts.append(str(selected_entity["entity_name"]))
else:
    names = [
        item.get("entity_name", "")
        for item in kg_evidence[:2]
        if item.get("entity_name")
    ]
    parts.extend(names)
```

这样 Web 检索会更贴近图片主体。

---

### 4.6 新增 `_web_search()`

作用：调用网页检索 API。

```python
results = self.search_pipeline(web_query, k=self.web_top_k)
```

它和 Task1 的 `_image_search()` 有明显区别：

| 方法 | 输入 | 作用 |
|---|---|---|
| `_image_search(image)` | PIL Image | 图像检索 / mock KG |
| `_web_search(web_query)` | 文本 query | 网页检索 |

这正好对应 Task2 的新增要求：允许网页检索。

---

### 4.7 新增 `_build_web_evidence()`

作用：将网页原始结果整理成统一格式。

兼容字段包括：

```text
标题字段：
page_name / title / name

链接字段：
page_url / url / source_url

摘要字段：
page_snippet / snippet / description / summary / text / content / page_content
```

最终整理成：

```python
{
    "source": "web",
    "rank": idx + 1,
    "score": 0.1234,
    "title": title,
    "url": url,
    "snippet": snippet[:1200],
}
```

这样后面过滤、融合、Prompt 构造都可以统一处理。

---

### 4.8 新增 `_rank_web_evidence()`

作用：过滤网页噪声，只保留相关网页。

它会调用：

```python
_score_web_evidence()
```

然后只保留：

```python
candidate["web_rule_score"] >= self.min_web_score
```

最后按分数排序，只取前 `web_keep_top_n` 条。

---

### 4.9 新增 `_score_web_evidence()`

作用：给网页证据打分。

评分依据包括：

```text
1. 网页内容是否命中用户问题关键词；
2. 网页内容是否命中 web_query 关键词；
3. 网页内容是否命中 Task1 选中实体；
4. 网页内容是否命中 KG 候选实体；
5. 网页内容是否命中 KG 属性；
6. snippet 是否过短；
7. 是否包含广告、登录、cookie 等噪声词。
```

加分逻辑：

```python
score += 0.06 * len(query_tokens & text_tokens)
score += 0.03 * len(web_query_tokens & text_tokens)
score += 0.08 * len(entity_tokens & text_tokens)
```

降分逻辑：

```python
if len(str(web_item.get("snippet", ""))) < 40:
    score -= 0.08
```

以及噪声词：

```python
noise_terms = [
    "advertisement", "subscribe", "cookie", "privacy policy",
    "login", "sign up", "cart", "buy now", "sponsored"
]
```

这个模块是 Task2 区别于“简单网页拼接”的关键。

---

### 4.10 新增 `_fuse_multisource_evidence()`

作用：整合 KG 和 Web 证据。

融合后的结构包括：

```python
{
    "query": query,
    "selected_entity": {...},
    "kg_candidates": [...],
    "web_evidence": [...],
    "policy": "..."
}
```

证据优先级写得很明确：

```text
Image-KG evidence is primary.
Web evidence is auxiliary.
Use web evidence only when it is relevant and consistent with image-KG evidence.
```

也就是说：

```text
KG 是强证据
Web 是辅助证据
Web 不应该覆盖 KG
Web 主要用于补充背景知识
```

---

### 4.11 新增 `_answer_task2_with_llm()`

作用：调用 DeepSeek 完成 Task2 的最终回答。

它和 Task1 的回答不同，因为 Task2 会把两类证据都传给模型：

```text
1. Selected image-KG entity
2. Selected entity attributes
3. Other image-KG candidates
4. Filtered web evidence
5. Conversation history
```

如果 DeepSeek 调用失败，则回退：

```python
return self._answer_with_rules(query, kg_candidates)
```

这保证了系统不会因为 API 暂时失败而完全崩掉。

---

### 4.12 新增 `_build_task2_answer_messages()`

这是 Task2 的 Prompt 核心。

System Prompt 明确告诉模型：

```text
You are a visual question answering assistant for a multi-source augmented task.
Image-KG evidence is directly retrieved from visually similar images and is the primary evidence.
Web evidence is auxiliary and may contain noise.
Use web evidence only when it is relevant to the question and consistent with the image-KG evidence.
Do not invent unsupported facts.
If the answer cannot be determined from the provided evidence, answer 'I don't know'.
Answer in the same language as the user's question.
```

这对应 Task2 的关键要求：

```text
多源信息筛选
跨源融合
噪声鲁棒性
证据不足不编造
```

User Prompt 中进一步规定：

```text
1. 优先使用 selected image-KG entity 和 attributes；
2. Web evidence 只用于补充或验证；
3. 忽略无关或噪声网页；
4. Web 和 KG 冲突时，优先 KG；
5. 直接回答问题，不输出推理过程。
```

---

## 5. Task2Agent.py 的完整运行流程

整体流程可以概括为：

```text
输入：
    queries
    images
    message_histories

对每个样本：

1. 使用 Task1 的 _image_search(image)
   获取相似图片和 mock KG。

2. 使用 Task1 的 _build_evidence()
   将 KG 原始结果整理为实体候选。

3. 使用 Task1 的 _rank_candidates_by_rules()
   根据用户问题对 KG 候选实体重排。

4. 使用 Task1 的 _select_entity()
   选出最可能的图片主体实体。

5. 使用 Task2 的 _build_web_query()
   将用户问题和 KG 实体拼成网页检索 query。

6. 使用 Task2 的 _web_search()
   调用网页检索 API。

7. 使用 Task2 的 _build_web_evidence()
   抽取网页标题、URL 和 snippet。

8. 使用 Task2 的 _rank_web_evidence()
   过滤无关网页和噪声网页。

9. 使用 Task2 的 _fuse_multisource_evidence()
   组织 KG + Web 多源证据。

10. 使用 Task2 的 _answer_task2_with_llm()
    调用 DeepSeek 生成最终答案。

11. 使用 Task1 的 _finalize_answer()
    清洗最终答案。
```

简化图：

```text
图片 + 问题
    ↓
Task1 图像检索 / mock KG
    ↓
KG 实体候选
    ↓
Task1 实体选择
    ↓
构造 Web Query
    ↓
Task2 网页检索
    ↓
网页证据过滤
    ↓
KG + Web 证据融合
    ↓
DeepSeek 多源回答
    ↓
最终答案
```

---

## 6. 如何使用 Task2Agent.py

### 6.1 文件放置位置

将 `Task2Agent.py` 放到项目的 `agents/` 目录下。

推荐目录结构：

```text
agents/
├── base_agent.py
├── Task1KGAgent.py
├── Task2Agent.py
├── user_config.py
└── ...
```

注意：

```text
Task1KGAgent.py 必须保留原样。
Task2Agent.py 会从 agents.Task1KGAgent 导入 Task1KGAgent。
```

---

### 6.2 修改 `user_config.py`

把 `user_config.py` 改成：

```python
from agents.Task2Agent import Task2Agent

UserAgent = Task2Agent
```

如果之前是：

```python
UserAgent = RandomAgent
```

或者：

```python
UserAgent = Task1KGAgent
```

需要替换为 Task2Agent。

---

### 6.3 配置 DeepSeek API Key

不要把 API Key 写进代码。

Linux / Mac：

```bash
export DEEPSEEK_API_KEY="你的新 key"
```

Windows PowerShell：

```powershell
$env:DEEPSEEK_API_KEY="你的新 key"
```

如果需要指定模型：

```bash
export DEEPSEEK_MODEL="你的模型名"
```

Windows PowerShell：

```powershell
$env:DEEPSEEK_MODEL="你的模型名"
```

默认会沿用 Task1 中的逻辑：

```python
self.model_name = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
```

---

### 6.4 开启 Task2 调试日志

建议先开启调试日志，检查网页检索是否正常。

Linux / Mac：

```bash
export TASK2_DEBUG_PATH="debug/task2_debug.jsonl"
```

Windows PowerShell：

```powershell
$env:TASK2_DEBUG_PATH="debug/task2_debug.jsonl"
```

日志中会记录：

```text
1. 当前 query；
2. Task1 选中的 KG 实体；
3. KG 证据数量；
4. Web Query；
5. Web 原始结果数量；
6. 过滤后保留的网页标题；
7. DeepSeek 是否可用；
8. 最终回答片段。
```

日志不会记录 API Key。

---

### 6.5 本地运行评测

根据项目 README 的评测方式，可以运行类似命令：

```bash
python local_evaluation.py     --dataset-type single-turn     --split validation     --num-conversations 20     --display-conversations 3     --eval-model None
```

Task2 需要网页检索，所以不要加：

```bash
--suppress-web-search-api
```

这个参数是 Task1 单源增强关闭 Web Search 时用的。  
Task2 必须允许 Web Search。

---

## 7. Task2 与 Task1 的区别总结

| 对比项 | Task1KGAgent | Task2Agent |
|---|---|---|
| 图像检索 | 有 | 复用 Task1 |
| mock KG 解析 | 有 | 复用 Task1 |
| KG 实体重排 | 有 | 复用 Task1 |
| DeepSeek 实体选择 | 有 | 复用 Task1 |
| 网页检索 | 无 | 新增 |
| 网页证据解析 | 无 | 新增 |
| 网页去噪 | 无 | 新增 |
| KG-Web 融合 | 无 | 新增 |
| 多源 Prompt | 无 | 新增 |
| Web 与 KG 冲突处理 | 无 | 新增 |
| 证据不足回答 I don't know | 有一定体现 | 明确强化 |
| batch size | Task1 自己决定 | Task2 保守设为 1 |

---

## 8. 适合写进报告的表述

可以在报告中写：

```text
在 Task1 单源增强 Agent 的基础上，我们实现了 Task2 多源增强 Agent。系统首先复用 Task1 的图像检索与 mock KG 实体解析能力，从相似图像中获得与当前图片相关的结构化实体及属性；随后将用户问题与 KG 中选出的主实体拼接为网页检索 query，调用网页检索接口获取外部网页证据。由于网页结果可能包含噪声，系统进一步设计了基于问题关键词、实体关键词和 KG 属性关键词的证据评分机制，对网页片段进行过滤和排序。最终，系统将 image-KG 证据作为强证据、web evidence 作为辅助证据输入 DeepSeek，并在 prompt 中显式规定证据优先级和冲突处理规则，从而实现多源信息筛选、跨源融合与噪声鲁棒性。
```

---

## 9. 答辩时可以这样讲

```text
Task2 不是重新写一个 Agent，而是在 Task1KGAgent 上继承扩展。Task1 解决的是“图片到 KG 结构化证据”的问题，Task2 在此基础上新增网页检索。

我们的处理方式不是把网页结果直接塞给模型，而是先根据用户问题、Task1 选出的 KG 实体和 KG 属性对网页片段进行相关性打分，只保留高相关网页。最后在 Prompt 中明确规定：KG 是直接与图像相关的强证据，Web 是辅助证据；如果两者冲突，优先 KG；如果证据不足，回答 I don't know。

因此，Task2 的核心改进是多源证据融合和网页噪声过滤。
```

---

## 10. 注意事项

### 10.1 Task1 的问题会传递到 Task2

因为 Task2 继承并复用 Task1，所以如果 Task1 的图像检索返回为空，Task2 的 KG 证据也会为空。

这时 Task2 仍然会尝试网页检索，但网页检索 query 可能不够准确。

所以建议先确认 Task1 能正常返回：

```text
entity_name
entity_attributes
score
```

---

### 10.2 Task2 不适合关闭 Web Search

Task2 的核心就是多源增强，因此运行 Task2 时不要使用：

```bash
--suppress-web-search-api
```

否则 Task2 的网页检索会失败，只能退化成 Task1。

---

### 10.3 DeepSeek API 不可用时会退化成 Task1

如果没有设置 `DEEPSEEK_API_KEY`，或者 API 调用失败，Task2 会调用：

```python
self._answer_with_rules(query, ranked_kg)
```

这意味着：

```text
不会崩溃
但也不会真正完成多源增强回答
```

所以正式测试时必须确保 DeepSeek API 正常。

---

### 10.4 需要查看 Debug 日志

如果结果不好，优先看：

```text
debug/task2_debug.jsonl
```

重点检查：

```text
1. selected_entity 是否正确；
2. web_query 是否合理；
3. web_count 是否大于 0；
4. kept_web_titles 是否相关；
5. answer 是否只是在复述实体名。
```

---

## 11. 最终结论

`Task2Agent.py` 的作用可以概括为：

```text
在不修改 Task1KGAgent.py 的前提下，
继承 Task1 的 image-KG 单源增强能力，
新增网页检索、网页证据过滤和 KG-Web 多源融合，
从而满足 Task2 Multi-source Augmentation 的要求。
```

它的核心优势是：

```text
1. 保持 Task1 稳定，不破坏已有代码；
2. 显式加入 Web Search，符合 Task2 要求；
3. 不直接盲信网页结果，而是先过滤噪声；
4. 明确 KG 强证据、Web 辅助证据的优先级；
5. Prompt 中约束模型不要编造，证据不足回答 I don't know；
6. 保留规则兜底，提升运行稳定性。
```
