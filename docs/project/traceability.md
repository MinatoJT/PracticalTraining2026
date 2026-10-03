# 需求—设计—验证追踪矩阵

文档编号：PT2026-RTM；版本 0.1；状态：待评审。
工程内源码路径以下省略 `meta-comprehensive-rag-benchmark-starter-kit-main/` 前缀。需求见 [SRS](requirements.md)，验证项见 [VV](verification.md)。

| 需求 | 设计 / 源码依据 | 现有用例或待验证项 | 当前证据与缺口 |
|---|---|---|---|
| REQ-01 单源增强 | `config.py::for_task`、`retrieval/image_kg.py`、Task1KGAgent | `test_task1_never_calls_web_search`；VAL-02 | 静态存在；单测导入阻塞，未证明真实 KG 回答效果 |
| REQ-02 多源增强 | `retrieval/web.py`、`reasoning/evidence.py`、Task2Agent | `test_task2_uses_multi_query_and_deduplicates_web_evidence`；VAL-03 | 替身用例存在；真实噪声鲁棒性待测 |
| REQ-03 多轮问答 | `conversation.py`、`pipeline.py` 会话状态和查询修正 | history/cache/anchor/visual followup/namesake/bare definition 用例；VAL-04 | 用例未执行；跨会话及真实 2–6 轮覆盖待补 |
| REQ-04 批接口 | `agents.py::batch_generate_response`、`agents/base_agent.py`、evaluator 长度检查 | `test_batch_length_validation`；VAL-01 | 不等长检查静态存在；完整契约运行待测 |
| REQ-05 拒答 | `evidence_gate`、`pipeline.py::_finalize`、providers | `test_no_evidence_returns_unknown_without_answer_call`、missing keys / request error 用例；VAL-05 | 稳定知识例外与故障拒答需人工审查 |
| REQ-06 可追溯证据 | `tracing.py`、`set_trace_contexts`、`UI/run_eval.py` 输出 | VAL-08 | 有日志机制，不保证递归脱敏或成功写入；完整归档待落实 |
| REQ-07 指标与复现 | `local_evaluation.py`、`UI/run_eval.py`、`evaluation_utils.py` | VAL-06、真实 Task1–3 评测 | 本轮无数值结果；口径与阈值待确认 |
| REQ-08 课程交付 | `_course_extract.txt`、本项目文档导航 | 维护者对源码/报告/周报核对 | 未取得最终交付件及教师验收记录 |

## 反向定位与维护

新增或修改代码先找到受影响需求行，再更新设计、测试和结果；新增测试应标识覆盖哪项需求，不用“有测试文件”替代测试结果。没有需求来源的实现可标注为实验性，待评审是否纳入范围；没有验证的需求保持缺口可见。

本轮证据入口：[导入失败原始记录](evidence/2026-10-03-unittest.txt)。真实实验后补充：运行编号 → 代码 SHA → 数据/配置 → 逐轮结果 → 指标 → 失败原因 → 复测结果。不要把未执行的 VAL 项填成完成。
