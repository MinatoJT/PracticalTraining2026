import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
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
os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")

from datasets import load_dataset
from rich.console import Console

from agents.Task1KGAgent import Task1KGAgent
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
            try:
                from openai import OpenAI

                client = OpenAI(
                    api_key=os.environ.get("DEEPSEEK_API_KEY"),
                    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                )
                prompt = (
                    "你是问答系统评估器。请判断 Prediction 是否正确回答了 Question，且是否覆盖 Ground truth 的关键信息。\n"
                    "允许同义改写、简短回答、顺序不同；如果 Prediction 只给实体名但没有回答问题所问属性，判 false。\n"
                    "只返回 JSON：{\"accuracy\": true 或 false, \"reason\": \"简短原因\"}\n\n"
                    f"Question: {query}\n"
                    f"Ground truth: {ground_truth}\n"
                    f"Prediction: {agent_response}\n"
                )
                response = client.chat.completions.create(
                    model=eval_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=120,
                )
                raw = response.choices[0].message.content or ""
                match = re.search(r"\{.*\}", raw, re.S)
                parsed = json.loads(match.group(0)) if match else {"accuracy": False, "reason": raw[:120]}
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
    text_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    image_model_name = "openai/clip-vit-large-patch14-336"
    image_index = "crag-mm-2025/image-search-index-validation"
    web_index = None if task == "task1" else "crag-mm-2025/web-search-index-validation"
    return UnifiedSearchPipeline(
        text_model_name=text_model_name,
        image_model_name=image_model_name,
        web_hf_dataset_id=web_index,
        image_hf_dataset_id=image_index,
    )


def main():
    parser = argparse.ArgumentParser(description="CRAG-MM 本地评测 UI 后端")
    parser.add_argument("--task", choices=["task1", "task2", "task3"], default="task1")
    parser.add_argument("--agent", choices=["task1kg", "user_config"], default="task1kg")
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

    search_pipeline = build_search_pipeline(args.task)
    if args.agent == "task1kg":
        agent_cls = Task1KGAgent
    else:
        import importlib
        sys.modules.pop("agents.user_config", None)
        ProjectUserAgent = importlib.import_module("agents.user_config").UserAgent
        agent_cls = ProjectUserAgent
    agent = agent_cls(search_pipeline=search_pipeline)

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
