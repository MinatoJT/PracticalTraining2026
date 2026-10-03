# 测试与验证说明 / 本轮记录

文档编号：PT2026-VV；版本 0.1；检查日期：2026-10-03；被检查代码：`870ec0ae954e1035ba34ab30f90e07b1cf158496`。

## 1. 结果摘要

| 检查 | 本轮结果 | 证据 / 限制 |
|---|---|---|
| 仓库与入口静态核对 | 已执行 | 根目录、newagents、agents、UI runner、配置与测试源码；不等于运行验证 |
| newagents 单元测试 | 阻塞，退出码 1 | 缺少 `cragmm_search`，导入测试模块失败；18 个业务用例未执行 |
| Task1–3 真实数据及 API 评测 | 未执行 | 无本轮在线指标、模型可用性或费用证据 |
| Qt / WinUI3 / Docker | 未执行 | 无目标环境构建与交互结果；静态发现 WinUI3 默认 agent 标签与 runner 不兼容 |
| 性能、费用、隐私与稳健性 | 未执行 | 不发布通过或合规结论 |

单测使用命令：在工程根执行 `python -m unittest newagents.tests.test_pipeline -v`，检查进程移除了 Qwen / DashScope / DeepSeek API Key 环境变量。原始导入失败输出见 [证据记录](evidence/2026-10-03-unittest.txt)。测试框架显示的 `Ran 1 test` 是失败的加载占位项，不能当作执行了一个业务测试。

## 2. 已有用例与适用范围

`newagents/tests/test_pipeline.py` 有 18 个 `test_*` 方法，采用 FakeSearchPipeline、FakeVision、FakeAnswerer 与 mock。覆盖范围包括：

- JSON 提取；Task1 不调用 Web；Task2 多查询与证据去重
- Task3 历史传递、同图检索缓存、首轮锚点复用、视觉追问重新重排
- 地址关系查询、同名实体第二跳与 planner 已解析的第二跳、裸指代定义
- 高精度核查选择：简单高置信身份、数字/历史、奖项/类别列表
- 无证据拒答且不调用回答器；批长不一致；缺凭据时保持离线
- Qwen 请求错误不自动切模型；低置信视觉主体审查

替身测试验证程序分支与接口，不证明真实图片识别、模型答案质量、索引可用性、API 限流行为或线上性能。`testTask1.py`、`testTask23Diagnostics.py`、`testQwenVision.py` 主要对应旧 agents 体系，不能替代新实现验证；本轮未执行。

## 3. 待执行验证计划

| 编号 | 操作与预期 | 需要保留 |
|---|---|---|
| VAL-01 | 安装真实依赖后运行 18 个 newagents 单测；全部业务用例被发现且结果明确 | 命令、Python/依赖版本、完整日志、退出码 |
| VAL-02 | 固定 Task1 样本；检查 Web 调用为零，回答与 KG/视觉依据及拒答原因 | trace、样本 ID、回答、人工核对 |
| VAL-03 | 固定 Task2 样本与噪声/重复证据；检查去重、多源依据和错误传播 | 检索词、证据 ID、模型响应摘要、评分 |
| VAL-04 | Task3 2–6 轮案例覆盖新话题、代词、重新看图、空/无效会话和会话隔离 | 逐轮输入/输出、session/turn ID、锚点变更 |
| VAL-05 | 缺凭据、超时、限流、无效 JSON、无日志权限分别验证 | 故障注入方法、降级原因、错误记录；不可误报质量拒答 |
| VAL-06 | 同一批样本比较 exact match 与本地语义 judge；按口径分别汇总 | judge 模型/版本、提示词或代码 SHA、原始判定、分歧分析 |
| VAL-07 | 目标 Windows 环境验证 Qt/WinUI3；选择任务/agent、重复运行、结果展示 | 构建日志、运行环境、操作记录 |
| VAL-08 | 日志递归敏感内容抽查、文件覆盖与归档恢复演练 | 脱敏记录、清单/校验值、完整性检查 |

尚未批准数值验收阈值。验证结果记录至少包括：编号、日期、执行人、环境、代码 SHA、配置、样本、预期、实际、结果（通过/失败/阻塞）、证据位置、缺陷及复测。

## 4. 指标解释与比较约束

以当前 `local_evaluation.py::calculate_scores` 为代码口径：accuracy 为 correct/total，missing 为 miss/total，hallucination_rate 为 hallucination/total；通常 total>1 时 truthfulness_score 为 `(2*correct + miss)/total - 1`。代码在 total<=1 的该项分支返回 0，其他除法对空样本也需单独验证。

missing 是当前 evaluator 的拒答/缺失分类比例，不能直接解释成“正确识别所有不可回答问题”的能力。多轮计分有连续两次不正确后影响后续轮次的处理，并计算会话均值；应同时保留原始逐轮结果和处理后的口径。`--eval-model None` 是 exact match，UI DeepSeek judge 含本地规则捷径和语义判断，既非独立人工真值，也不是官方线上评分。

历史笔记中任何数字在缺少对应代码版本、数据样本、模型/配置与原始日志时，只能引用为历史观察；不得填入本轮结果或宣称提升。
