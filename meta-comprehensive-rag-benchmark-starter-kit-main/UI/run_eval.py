import argparse
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
from functools import wraps
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from conversation_validation import valid_conversation_indices
from evaluation_utils import build_deepseek_judge_prompt, semantic_shortcut
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
DATASET_DIR = ROOT_DIR / "Dataset"
DATASET_DIR.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(DATASET_DIR / "hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", str(DATASET_DIR / "hf_datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DATASET_DIR / "hf_hub"))
os.environ.setdefault("HF_XET_CACHE", str(DATASET_DIR / "hf_xet"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(DATASET_DIR / "transformers"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(DATASET_DIR / "sentence_transformers"))
os.environ.setdefault("CRAG_CACHE_DIR", str(DATASET_DIR / "crag_images"))
os.environ.setdefault("CRAG_WEBSEARCH_CACHE_DIR", str(DATASET_DIR / "crag_web_search"))
os.environ.setdefault("TASK1_DEBUG_PATH", str(ROOT_DIR / "UI" / "outputs" / "task1" / "debug.jsonl"))
os.environ.setdefault("TASK2_DEBUG_PATH", str(ROOT_DIR / "UI" / "outputs" / "task2" / "debug.jsonl"))
os.environ.setdefault("TASK3_DEBUG_PATH", str(ROOT_DIR / "UI" / "outputs" / "task3" / "debug.jsonl"))
os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")

from datasets import load_dataset
from rich.console import Console

from agents.Task1KGAgent import Task1KGAgent
from agents.Task2Agent import Task2Agent
from agents.Task3Agent import Task3Agent
import types
_fake_user_config = types.ModuleType("agents.user_config")
_fake_user_config.UserAgent = Task1KGAgent
sys.modules["agents.user_config"] = _fake_user_config
from cragmm_search.search import UnifiedSearchPipeline
import local_evaluation
from utils import display_results, ensure_crag_cache_dir_is_configured


class _SimpleEncoding:
    def __init__(self, ids):
        self.ids = ids


class _SimpleTokenizer:
    def __init__(self):
        self.max_length = 75

    def enable_truncation(self, max_length):
        self.max_length = max_length

    def encode_batch(self, texts):
        return [_SimpleEncoding(str(text).split()[: self.max_length]) for text in texts]

    def decode(self, ids):
        return " ".join(ids)


class _TokenizerFactory:
    @staticmethod
    def from_pretrained(_model_name):
        return _SimpleTokenizer()


local_evaluation.Tokenizer = _TokenizerFactory
CRAGEvaluator = local_evaluation.CRAGEvaluator
console = Console()
LIVE_EVENT_PREFIX = "__CRAGMM_LIVE_EVENT__"


def install_live_event_stream(agent, task: str) -> None:
    """包装 Agent 批量接口，把每轮问答和图片作为 JSON 行实时发送给 Qt。"""
    if os.environ.get("CRAGMM_LIVE_EVENTS", "0") != "1":
        return

    original = agent.batch_generate_response
    emit_lock = threading.Lock()
    run_id = f"{int(time.time())}_{os.getpid()}"
    image_dir = ROOT_DIR / "UI" / "outputs" / "live" / run_id
    image_dir.mkdir(parents=True, exist_ok=True)
    saved_images = {}

    def prepare_image(image):
        """生成稳定会话标识并保存轻量缩略图；同一多轮图片只保存一次。"""
        try:
            preview = image.convert("RGB").copy()
            preview.thumbnail((720, 540))
            buffer = io.BytesIO()
            preview.save(buffer, format="JPEG", quality=88)
            data = buffer.getvalue()
            conversation_id = hashlib.sha1(data).hexdigest()[:12]
            if conversation_id not in saved_images:
                path = image_dir / f"{conversation_id}.jpg"
                path.write_bytes(data)
                saved_images[conversation_id] = str(path)
            return conversation_id, saved_images[conversation_id]
        except Exception:
            fallback_id = f"unknown-{len(saved_images) + 1}"
            return fallback_id, ""

    @wraps(original)
    def wrapped(queries, images, message_histories):
        responses = original(queries, images, message_histories)
        with emit_lock:
            for query, image, history, response in zip(queries, images, message_histories, responses):
                conversation_id, image_path = prepare_image(image)
                event = {
                    "task": task,
                    "conversation_id": conversation_id,
                    "turn": len(history or []) // 2,
                    "query": str(query),
                    "response": str(response),
                    "history": history or [],
                    "image_path": image_path,
                }
                print(LIVE_EVENT_PREFIX + json.dumps(event, ensure_ascii=False), flush=True)
        return responses

    agent.batch_generate_response = wrapped


def _semantic_shortcut(query: str, ground_truth: str, prediction: str) -> bool:
    return semantic_shortcut(query, ground_truth, prediction)
    """本地高置信语义捷径：答案已经明显覆盖关键词/数字时，避免 LLM judge JSON 截断误判。"""
    pred_l = prediction.lower()
    gt_l = ground_truth.lower()
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    }
    for word, digit in number_words.items():
        pred_l = re.sub(rf"\b{word}\b", digit, pred_l)
        gt_l = re.sub(rf"\b{word}\b", digit, gt_l)
    if "i don't know" in pred_l or "i don’t know" in pred_l:
        return False

    # 数字类答案要求 ground truth 中的关键数字大多出现在 prediction 中。
    gt_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", gt_l))
    pred_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", pred_l))
    if gt_numbers:
        # 区间/里程类答案允许覆盖大多数数字即可，避免 40-45 与 40/45 表达差异。
        if len(gt_numbers & pred_numbers) >= max(1, min(len(gt_numbers), 3)):
            return True

    stop = {
        "the", "and", "for", "with", "that", "this", "from", "what", "when", "where",
        "which", "who", "why", "how", "does", "did", "was", "were", "are", "is",
        "its", "into", "about", "there", "their", "have", "has", "had", "can",
        "could", "would", "should", "will", "shall", "than", "then", "them", "they",
        "you", "your", "because", "between", "not", "also", "been", "being", "answer",
    }
    gt_tokens = {t for t in re.findall(r"[a-z0-9]+", gt_l) if len(t) >= 4 and t not in stop}
    pred_tokens = {t for t in re.findall(r"[a-z0-9]+", pred_l) if len(t) >= 4 and t not in stop}
    if not gt_tokens or not pred_tokens:
        return False
    overlap = len(gt_tokens & pred_tokens)
    return overlap >= 3 and overlap / max(1, min(len(gt_tokens), len(pred_tokens))) >= 0.55


def _parse_deepseek_judge(raw: str) -> dict:
    """容错解析 DeepSeek judge 输出，兼容 JSON 截断、代码块和大小写差异。"""
    raw = raw or ""
    match = re.search(r"\{.*?\}", raw, re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return {"accuracy": bool(parsed.get("accuracy")), "reason": str(parsed.get("reason", ""))[:120]}
        except Exception:
            pass
    raw_l = raw.lower()
    if re.search(r'"?accuracy"?\s*[:：]\s*true', raw_l) or re.search(r"\btrue\b", raw_l):
        return {"accuracy": True, "reason": raw[:120]}
    if re.search(r'"?accuracy"?\s*[:：]\s*false', raw_l) or re.search(r"\bfalse\b", raw_l):
        return {"accuracy": False, "reason": raw[:120]}
    return {"accuracy": False, "reason": raw[:120]}


def patch_deepseek_judge(eval_model_name):
    """当评测模型选择 deepseek-* 时，用 DeepSeek 替换官方语义评测逻辑。"""
    if not eval_model_name or not str(eval_model_name).startswith("deepseek"):
        return eval_model_name

    def evaluate_response_with_deepseek(self, crag_turn_data):
        agent_response = str(crag_turn_data["agent_response"])
        ground_truth = str(crag_turn_data["ground_truth"])
        query = str(crag_turn_data["query"])

        is_idk = "i don't know" in agent_response.lower() or "i don’t know" in agent_response.lower()
        is_exact_match = agent_response.strip().lower() == ground_truth.strip().lower()
        is_correct = is_exact_match
        is_semantically_correct = is_exact_match
        api_response = None

        if not is_idk and not is_exact_match:
            if os.environ.get("CRAGMM_SEMANTIC_SHORTCUT", "0") == "1" and _semantic_shortcut(query, ground_truth, agent_response):
                api_response = {"accuracy": True, "reason": "local_semantic_shortcut"}
                is_correct = True
                is_semantically_correct = True
            else:
                try:
                    from openai import OpenAI

                    client = OpenAI(
                        api_key=os.environ.get("DEEPSEEK_API_KEY"),
                        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                    )
                    prompt = (
                        "你是问答系统评估器。判断 Prediction 是否正确回答 Question，且是否覆盖 Ground truth 的关键信息。\n"
                        "允许同义改写、简短回答、顺序不同；只给实体名且没有回答属性时判 false。\n"
                        "只返回一行 JSON，不要解释：{\"accuracy\": true 或 false}\n\n"
                        f"Question: {query}\n"
                        f"Ground truth: {ground_truth}\n"
                        f"Prediction: {agent_response}\n"
                    )
                    prompt = build_deepseek_judge_prompt(query, ground_truth, agent_response)
                    response = client.chat.completions.create(
                        model=eval_model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=40,
                        extra_body={"thinking": {"type": "disabled"}},
                    )
                    raw = response.choices[0].message.content or ""
                    parsed = _parse_deepseek_judge(raw)
                    is_semantically_correct = bool(parsed.get("accuracy"))
                    is_correct = is_semantically_correct
                    api_response = parsed
                except Exception as exc:
                    api_response = {"accuracy": False, "reason": f"deepseek_judge_error: {repr(exc)}"}
                    is_correct = False
                    is_semantically_correct = False

        return {
            **crag_turn_data,
            "is_exact_match": is_exact_match,
            "is_correct": is_correct,
            "is_miss": is_idk,
            "is_semantically_correct": is_semantically_correct,
            "api_response": api_response,
        }

    CRAGEvaluator.evaluate_response = evaluate_response_with_deepseek
    return eval_model_name


def build_search_pipeline(task: str):
    # Task2/Task3 的 web-search-index-validation 是用 BAAI/bge-large-en-v1.5
    # 建的 1024 维索引；如果用 MiniLM(384 维) 查询会触发 dimension mismatch。
    text_model_name = "BAAI/bge-large-en-v1.5" if task != "task1" else "sentence-transformers/all-MiniLM-L6-v2"
    image_model_name = "openai/clip-vit-large-patch14-336"
    image_index = "crag-mm-2025/image-search-index-validation"
    web_index = None if task == "task1" else "crag-mm-2025/web-search-index-validation"
    return UnifiedSearchPipeline(
        text_model_name=text_model_name,
        image_model_name=image_model_name,
        web_hf_dataset_id=web_index,
        image_hf_dataset_id=image_index,
    )


def _has_complete_answers(conversation: dict) -> tuple[bool, str]:
    """检查多轮会话每个 turn 是否都有非空 ground truth。"""
    turns = conversation.get("turns", []) or []
    answers = conversation.get("answers", {}) or {}
    if isinstance(answers, dict):
        ids = list(answers.get("interaction_id", []) or [])
        values = list(answers.get("ans_full", []) or [])
        lookup = {str(key): value for key, value in zip(ids, values)}
    else:
        lookup = {
            str(item.get("interaction_id")): item.get("ans_full")
            for item in answers
            if isinstance(item, dict)
        }
    for turn in turns:
        interaction_id = str(turn.get("interaction_id", ""))
        if not str(lookup.get(interaction_id, "") or "").strip():
            return False, f"missing_answer:{interaction_id}"
    return True, ""


def _select_valid_conversations(dataset, requested: int):
    """先过滤坏会话再截取数量，保证请求 5 组时尽量实际评测 5 组。"""
    valid_indices, excluded = valid_conversation_indices(dataset, requested)
    if excluded:
        console.print(f"[yellow]已跳过 {len(excluded)} 个答案不完整会话：{excluded}[/yellow]")
    return dataset.select(valid_indices)


def main():
    parser = argparse.ArgumentParser(description="CRAG-MM 本地评测 UI 后端")
    parser.add_argument("--task", choices=["task1", "task2", "task3"], default="task1")
    parser.add_argument("--agent", choices=["task1kg", "task2agent", "task3agent", "user_config"], default="task1kg")
    parser.add_argument("--num-conversations", type=int, default=20)
    parser.add_argument("--display-conversations", type=int, default=5)
    parser.add_argument("--eval-model", default="None")
    parser.add_argument("--revision", default="v0.1.2")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    ensure_crag_cache_dir_is_configured()

    dataset_type = "multi-turn" if args.task == "task3" else "single-turn"
    repo_name = f"crag-mm-2025/crag-mm-{dataset_type}-public"
    eval_model = None if args.eval_model.lower() == "none" else args.eval_model
    eval_model = patch_deepseek_judge(eval_model)

    console.print(f"[bold blue]任务:[/bold blue] {args.task}")
    console.print(f"[bold blue]Agent:[/bold blue] {args.agent}")
    console.print(f"[bold blue]数据集:[/bold blue] {repo_name} ({args.revision})")

    dataset = load_dataset(repo_name, revision=args.revision)
    split = "validation" if "validation" in dataset else list(dataset.keys())[0]
    selected = dataset[split]
    num_conversations = min(args.num_conversations, len(selected))
    if num_conversations >= 0:
        if args.task == "task3":
            selected = _select_valid_conversations(selected, num_conversations)
        else:
            selected = selected.select(range(num_conversations))
        num_conversations = len(selected)

    # 每次运行使用独立日志文件，避免旧 run 追加后被误当成同一批结果。
    run_id = f"{int(time.time())}_{os.getpid()}"
    trace_dir = ROOT_DIR / "UI" / "outputs" / args.task
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"trace_{run_id}.jsonl"
    os.environ[f"{args.task.upper()}_DEBUG_PATH"] = str(trace_path)
    if args.task in {"task2", "task3"}:
        os.environ["TASK1_DEBUG_PATH"] = str(trace_path)

    search_pipeline = build_search_pipeline(args.task)
    if args.agent == "task1kg":
        agent_cls = Task1KGAgent
    elif args.agent == "task2agent":
        agent_cls = Task2Agent
    elif args.agent == "task3agent":
        agent_cls = Task3Agent
    else:
        import importlib
        sys.modules.pop("agents.user_config", None)
        ProjectUserAgent = importlib.import_module("agents.user_config").UserAgent
        agent_cls = ProjectUserAgent
    agent = agent_cls(search_pipeline=search_pipeline)
    install_live_event_stream(agent, args.task)

    evaluator = CRAGEvaluator(
        dataset=selected,
        agent=agent,
        eval_model_name=eval_model,
        num_conversations=num_conversations,
        show_progress=not args.no_progress,
        num_workers=4,
    )
    turn_results, score_dicts = evaluator.evaluate_agent()

    display_results(
        console,
        turn_results["all"],
        score_dicts["all"],
        display_conversations=args.display_conversations,
        is_ego=False,
        is_multi_turn=(dataset_type == "multi-turn"),
    )

    output_dir = ROOT_DIR / "UI" / "outputs" / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    turn_results["all"].to_csv(output_dir / "turn_evaluation_results_all.csv", index=False)
    turn_results["ego"].to_csv(output_dir / "turn_evaluation_results_ego.csv", index=False)
    with open(output_dir / "scores_dictionary.json", "w", encoding="utf-8") as f:
        json.dump(score_dicts, f, indent=2)
    console.print(f"[bold green]结果已保存到:[/bold green] {output_dir}")


if __name__ == "__main__":
    main()
