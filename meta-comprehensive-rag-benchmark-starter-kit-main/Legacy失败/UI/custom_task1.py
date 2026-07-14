import argparse
import os
import sys
from pathlib import Path

from PIL import Image

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
from agents.Task1KGAgent import Task1KGAgent
from cragmm_search.search import UnifiedSearchPipeline


def build_task1_search_pipeline():
    return UnifiedSearchPipeline(
        image_model_name="openai/clip-vit-large-patch14-336",
        image_hf_dataset_id="crag-mm-2025/image-search-index-validation",
        web_hf_dataset_id=None,
    )


def main():
    parser = argparse.ArgumentParser(description="Run one custom Task1 image+question example")
    parser.add_argument("--image", required=True, help="Local image path")
    parser.add_argument("--question", required=True, help="Question to ask about the image")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("Mode: custom Task1 question")
    print(f"Image: {image_path}")
    print(f"Question: {args.question}")
    print("Initializing image-search mock API. First run may download the CLIP model/index.")

    image = Image.open(image_path).convert("RGB")
    search_pipeline = build_task1_search_pipeline()
    agent = Task1KGAgent(search_pipeline=search_pipeline)
    answer = agent.batch_generate_response([args.question], [image], [[]])[0]

    print("\nAnswer:")
    print(answer)

    raw_results = agent._image_search(image)
    evidence = agent._build_evidence(raw_results)
    print("\nTop retrieved KG evidence:")
    for idx, item in enumerate(evidence[:3], start=1):
        attrs = item.get("attributes", {})
        preview = "; ".join(f"{k}: {v}" for k, v in list(attrs.items())[:8])
        print(f"[{idx}] score={item.get('score')} entity={item.get('entity_name')} {preview}")


if __name__ == "__main__":
    main()



