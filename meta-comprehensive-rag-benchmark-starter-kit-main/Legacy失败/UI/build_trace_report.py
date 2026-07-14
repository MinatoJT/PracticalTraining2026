import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path


def build_report(trace_path: Path, output_path: Path) -> int:
    """把 JSONL 事件流合并成一行一个 turn 的诊断表。"""
    events = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))

    answer_events = defaultdict(deque)
    rewrite_events = defaultdict(deque)
    for event in events:
        if event.get("event") == "task3_llm_success":
            answer_events[str(event.get("query", ""))].append(event)
        elif event.get("event") == "task3_sentence_rewrite":
            rewrite_events[str(event.get("query", ""))].append(event)

    rows = []
    for event in events:
        if event.get("event") != "task3_query":
            continue
        query = str(event.get("query", ""))
        answer_event = answer_events[query].popleft() if answer_events[query] else {}
        rewrite_event = rewrite_events[query].popleft() if rewrite_events[query] else {}
        score_components = event.get("kg_score_components", []) or []
        image_top1 = score_components[0].get("entity", "") if score_components else ""
        rows.append({
            "conversation": event.get("session_id", ""),
            "turn": event.get("turn_idx", ""),
            "query": query,
            "contextual_query": event.get("contextual_query", ""),
            "use_image": event.get("use_image", False),
            "image_top1": image_top1,
            "selected_entity": event.get("selected_entity", ""),
            "visual_anchor": event.get("anchor_entity", ""),
            "web_query": event.get("web_query", ""),
            "raw_answer": answer_event.get("raw_answer", ""),
            "gate_reason": answer_event.get("idk_reason", ""),
            "quality_branches": "+".join(answer_event.get("quality_branches", []) or []),
            "rewrite_answer": rewrite_event.get("rewritten", ""),
            "final_answer": event.get("answer", ""),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Task3 逐轮诊断 CSV")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.trace.with_name(args.trace.stem + "_diagnostic.csv")
    count = build_report(args.trace, output)
    print(f"Generated {count} rows: {output}")


if __name__ == "__main__":
    main()
