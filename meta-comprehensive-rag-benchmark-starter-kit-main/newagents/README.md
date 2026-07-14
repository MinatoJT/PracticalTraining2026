# CRAG-MM evidence-first agents

The public classes keep the original project-facing names so existing integrations can switch imports without changing call sites:

```python
from newagents import Task1KGAgent, Task2Agent, Task3Agent
```

They use one composition-based, evidence-first pipeline:

1. Build a standalone question (Task 3 treats earlier assistant text as unverified context).
2. Retrieve image-KG candidates; Task 2/3 also issue up to three web queries.
3. Ask Qwen-VL to verify the visual subject and rerank the compact evidence set.
4. Apply an evidence gate. Unsupported questions return `I don't know.`
5. Ask DeepSeek for a short JSON answer constrained to the selected evidence.

Task classes do not inherit from each other. The existing `agents/` implementation is untouched.

## Runtime

API mode is the default and reads `QWEN_VL_API_KEY` (or `DASHSCOPE_API_KEY`) and
`DEEPSEEK_API_KEY` from the process environment. Secrets are never written to traces.

The only local vision option is Qwen3-VL-4B BF16, batch size 1:

```powershell
$env:NEWAGENTS_QWEN_BACKEND = "local"
$env:NEWAGENTS_QWEN_LOCAL_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
```

No vLLM dependency or 48 GB assumption is introduced.
