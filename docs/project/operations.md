# 运行、配置与结果归档

文档编号：PT2026-OPS；版本 0.1；状态：操作说明，命令未作本轮端到端运行验证。

## 1. 运行目录与环境

所有以下命令都从 `meta-comprehensive-rag-benchmark-starter-kit-main/` 执行。使用该目录 `requirements.txt`，不要误用仓库根同名文件。该文件建议 Python 3.12，而旧 Dockerfile 使用 Python 3.10 并安装 vLLM，属于不同路径；不能把 Docker 作为已验证的 newagents API 快速部署方案。

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# POSIX: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest newagents.tests.test_pipeline -v
```

不要为通过测试伪造缺失依赖模块。安装条件不足时保留导入失败记录。联网运行可能下载较大数据/模型并产生 API 费用，预先检查磁盘、网络、账号授权和预算。历史笔记中的本地 Anaconda 第三方库补丁没有自动随仓库安装，遇到相应问题须记录依赖版本并复核，不应声称已具备补丁。

## 2. 配置关键项

| 环境变量 | 当前作用 |
|---|---|
| `QWEN_VL_API_KEY` / `DASHSCOPE_API_KEY` | Qwen API 凭据，优先前者；仅在受控环境设置 |
| `DEEPSEEK_API_KEY` | 回答/所选语义评测服务凭据 |
| `QWEN_VL_BASE_URL`、`DEEPSEEK_BASE_URL` | 提供方端点；谨慎更改，避免向错误服务传输数据 |
| `QWEN_VL_ANCHOR_MODEL` / `QWEN_VL_RERANK_MODEL` | 视觉规划与重排模型；`QWEN_VL_MODEL` 是兼容覆盖值 |
| `DEEPSEEK_MODEL` | 回答模型；配置默认值不是在线可用性验证 |
| `NEWAGENTS_QWEN_BACKEND` | 默认 api；local 分支需另外验证本地模型和硬件条件 |
| `NEWAGENTS_QWEN_LOCAL_MODEL` | 本地模式默认 `Qwen/Qwen3-VL-4B-Instruct` |
| `NEWAGENTS_MAX_SEARCH_QUERIES` | 默认 3，配置范围 1–5 |
| `NEWAGENTS_MIN_EVIDENCE_SCORE` / `NEWAGENTS_MIN_ANSWER_CONFIDENCE` | 默认为 0.08 / 0.24；调参时作为实验配置归档 |
| `NEWAGENTS_ALLOW_STABLE_KNOWLEDGE` | 默认 1；严格证据对照实验可评审后设 0 |
| `NEWAGENTS_DEBUG_PATH` | 可选 JSONL 日志路径；UI runner 设置独立 trace 路径 |

完整取值以 `newagents/config.py::AgentConfig.for_task` 和 providers 为准。`.env.example` 包含旧 agents 参数，不能整体当作 newagents 的配置规范；某些入口加载 `.env` 的时机不同，使用显式进程环境较清楚。不要把密钥写入命令日志、截图、文档或提交。

## 3. 入口选择

```sh
python UI/run_eval.py --task task1 --agent task1kg --num-conversations 5 --display-conversations 5 --eval-model None --revision v0.1.2 --sample-seed 42
python UI/run_eval.py --task task2 --agent task2agent --num-conversations 5 --eval-model None --revision v0.1.2 --sample-seed 42
python UI/run_eval.py --task task3 --agent task3agent --num-conversations 5 --eval-model None --revision v0.1.2 --sample-seed 42
```

同时明确 task 与 agent；CLI 不因 `--task` 自动改变默认 agent。Task3 的有效会话筛选与多轮计分使“请求会话数”和“最终计分轮数”不同，报告须分别记录。

- Qt：`python UI/app.py`；Windows 也可查看 `UI/run_ui.bat`。现有启动器可能含机器相关假设，先检查再用。
- WinUI3：已静态发现默认选项 `blackpearl_task1/2/3` 与 runner 的 `--agent` choices 不匹配（XAML 设置标签，C# 原样传参）。这些默认选项会被参数解析拒绝，当前不要据旧 README 认定可直接运行；先使用上述 CLI，代码修复另行评审。参阅 `WinUI3/README.md`、`.csproj` 与启动脚本，需要 Windows、.NET 8 和相应 Windows SDK/构建组件；本轮未构建。
- 原生 `local_evaluation.py` 走 `agents.user_config.UserAgent`，当前为 RandomAgent，不能与 UI 新实现混为一个基线。
- `legacy_task1kg` / `legacy_task2agent` / `legacy_task3agent` 对应当前 `agents/`，不是 `Legacy失败/`。

## 4. 输出与归档

UI runner 写到 `UI/outputs/<task>/`：

- `trace_<时间戳>_<进程号>.jsonl`：本次运行 trace
- `turn_evaluation_results_all.csv`、`turn_evaluation_results_ego.csv`：逐轮结果
- `scores_dictionary.json`：汇总
- `vision_stats.json`：视觉调用统计

后三类固定文件名会被后续运行覆盖。每次运行结束，在下一次启动前复制到独立实验目录，带上实际命令、代码 SHA、环境清单、数据 revision/样本顺序、task/agent、脱敏配置和控制台日志。建议目录命名 `YYYYMMDD-HHMM-task-agent-sha`；这是人工归档约定，当前程序不会自动生成完整清单。

现有 `.gitignore` 忽略 UI/outputs、Dataset、JSONL/CSV/JSON 等。保留大文件/原始实验材料在受控存储中，文档只登记可访问位置与摘要；不要强行提交含凭据或受限数据的日志。无输出或空日志不代表通过。
