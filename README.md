# PracticalTraining2026 · 多模态 RAG 实训

基于 CRAG-MM starter kit，围绕图像知识图谱单源增强、网页多源增强和多轮问答开展课程实训。实际 Python 工程位于 [`meta-comprehensive-rag-benchmark-starter-kit-main/`](meta-comprehensive-rag-benchmark-starter-kit-main/)，仓库根目录不是运行目录。

> 本轮为文档整理，不改变代码、默认配置或既有实验结果。资料基于提交 `870ec0ae954e1035ba34ab30f90e07b1cf158496`（2026-10-03 检查）。文档参照 GJB 438C 文档结构及 GJB 5000B 过程证据思路，按课程项目裁剪；不构成标准符合性、成熟度等级或认证结论。

## 从这里开始

- [课程任务与验收口径](docs/project/requirements.md)：Task1–3、工程约束、待确认事项
- [设计与接口](docs/project/design.md)：实际模块、输入输出、状态和外部依赖
- [运行与配置](docs/project/operations.md)：运行目录、入口、参数和结果归档
- [测试与验证](docs/project/verification.md)：现有测试、已执行检查、未验证范围
- [配置与变更管理](docs/project/change-management.md)：版本、评审、基线和变更影响
- [需求—实现—验证追踪](docs/project/traceability.md)
- [文档控制与标准裁剪](docs/project/README.md)

## 三条实现路径不要混淆

| 路径 | 当前用途 | 阅读建议 |
|---|---|---|
| `newagents/` | 当前 UI runner 的 Task1–3 实现；组合式证据流水线 | 优先阅读 [newagents 说明](meta-comprehensive-rag-benchmark-starter-kit-main/newagents/README.md) |
| `agents/` | 原有基线、旧 Task1–3 和 BaseAgent 接口；仍被保留 | [代理接口说明](meta-comprehensive-rag-benchmark-starter-kit-main/agents/README.md)；UI 的 `legacy_*` 选项实际指向这里 |
| `Legacy失败/` | 历史失败版本快照 | 仅作问题复盘，不作为当前运行入口 |

[`PracticalTraining.md`](meta-comprehensive-rag-benchmark-starter-kit-main/PracticalTraining.md) 保留了历次修复、运行环境和实验笔记；其中历史状态、个人机器路径、依赖本地补丁和旧默认值不等同于当前可复现状态。原有 starter kit [README](meta-comprehensive-rag-benchmark-starter-kit-main/README.md) 和 [docs](meta-comprehensive-rag-benchmark-starter-kit-main/docs/) 继续保留。

## 最短运行路线

在自行管理的环境中，从仓库根目录进入工程目录：

```sh
cd meta-comprehensive-rag-benchmark-starter-kit-main
python -m venv .venv
# 激活 .venv 后执行；依赖体量较大，安装前确认空间和所选 PyTorch 平台。
python -m pip install -r requirements.txt
python -m unittest newagents.tests.test_pipeline -v
```

运行 API 模式前，在进程环境设置 `QWEN_VL_API_KEY`（或 `DASHSCOPE_API_KEY`）与 `DEEPSEEK_API_KEY`。不要提交真实凭据。模型调用会产生费用并向相应服务发送问题、图像或证据，先使用可公开的测试样本。仅复制 `.env.example` 不代表所有入口都会加载该文件。

```sh
python UI/run_eval.py --task task1 --agent task1kg --num-conversations 5 --eval-model None
python UI/run_eval.py --task task2 --agent task2agent --num-conversations 5 --eval-model None
python UI/run_eval.py --task task3 --agent task3agent --num-conversations 5 --eval-model None
```

这些是按当前参数定义整理的复现命令，本次未运行联网评测。`--eval-model None` 使用 exact match；本地 DeepSeek judge 属于另一评测口径，不能混报为官方成绩。CLI 的默认 agent 固定为 `task1kg`，所以切换 task 时应像示例一样同时显式传入 agent。

直接执行 `python local_evaluation.py` 走 `agents/user_config.py`，该文件当前 `UserAgent = RandomAgent`，不会自动切到 `newagents`。WinUI3 默认 agent 标签与 runner 参数存在不匹配，见 [运行说明](docs/project/operations.md)。

## 当前验证状态

已静态核对源码、配置和文档，并尝试执行 newagents 单元测试。当前检查环境缺少 `cragmm_search`，测试在模块导入阶段被阻塞，18 个业务用例未执行。没有本轮模型 API、真实数据集、性能、Qt/WinUI3 或 Docker 通过结论。详见 [验证记录](docs/project/verification.md)。

根目录 `requirements.txt` 与工程目录同名文件内容不同；上述命令采用工程目录版本。项目授权及上游材料使用边界尚需维护者确认，本次不新增许可证或假定授权。
