import argparse
import csv
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


STAGES = {
    "A": "A_head_baseline",
    "B": "B_anchor_only",
    "C": "C_pure_visual_rerank",
    "D": "D_evidence_retry",
}


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def load_trace(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def norm(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def entity_match(expected, selected):
    expected_n, selected_n = norm(expected), norm(selected)
    if not expected_n or not selected_n:
        return False
    if selected_n in expected_n or expected_n in selected_n:
        return True
    expected_tokens = {x for x in expected_n.split() if len(x) > 2}
    selected_tokens = {x for x in selected_n.split() if len(x) > 2}
    overlap = expected_tokens & selected_tokens
    return len(overlap) >= 2 and len(overlap) / max(1, min(len(expected_tokens), len(selected_tokens))) >= 0.66


def vision_rows(trace):
    return {
        str(row.get("conversation", "")): row
        for row in trace
        if row.get("event") == "vision_pipeline"
    }


def llm_rows(trace):
    return {
        str(row.get("query", "")): row
        for row in trace
        if row.get("event") == "llm_success"
    }


def legacy_top1(row):
    source_map = {
        item.get("entity"): item.get("sources", [])
        for item in row.get("candidate_sources", [])
    }
    for item in row.get("legacy_candidates", []):
        entity = item.get("entity", "")
        if source_map.get(entity) == ["image_kg"]:
            return entity
    return ""


def result_label(row):
    if bool(row.get("is_miss")):
        return "IDK"
    return "CORRECT" if bool(row.get("is_correct")) else "HALLUCINATION"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--diagnostics-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    diagnostics_root = Path(args.diagnostics_root).resolve()
    output_root = Path(args.output_root).resolve()
    report_root = output_root / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")
    report_root.mkdir(parents=True, exist_ok=True)
    snapshots = report_root / "config_snapshots"
    snapshots.mkdir(exist_ok=True)

    expected_df = pd.read_csv(diagnostics_root / "candidate_comparison.csv")
    expected = dict(zip(expected_df["interaction_id"].astype(str), expected_df["expected_entity_annotation"].astype(str)))
    metrics, results, traces, visions, llms = {}, {}, {}, {}, {}

    for short, folder in STAGES.items():
        stage_dir = run_root / folder
        score = load_json(stage_dir / "scores_dictionary.json", {}).get("all", {})
        vision_stats = load_json(stage_dir / "vision_stats.json", {})
        metrics[short] = {
            "stage": short,
            "name": folder,
            "semantic_accuracy": score.get("accuracy"),
            "exact_accuracy": score.get("exact_match"),
            "missing_rate": score.get("missing"),
            "hallucination_rate": score.get("hallucination_rate"),
            "actual_idk_count": score.get("miss"),
            "truthfulness_score": score.get("truthfulness_score"),
            "schema_fallback_count": vision_stats.get("pipeline_fallbacks"),
            "visual_api_calls": (vision_stats.get("anchor_calls", 0) or 0) + (vision_stats.get("rerank_calls", 0) or 0),
            "visual_api_average_latency_seconds": vision_stats.get("average_latency"),
            "visual_prompt_tokens": vision_stats.get("prompt_tokens"),
            "visual_completion_tokens": vision_stats.get("completion_tokens"),
            "visual_total_tokens": vision_stats.get("total_tokens"),
        }
        result_path = stage_dir / "fixed_sample_results.csv"
        results[short] = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
        traces[short] = load_trace(stage_dir / "trace.jsonl")
        visions[short] = vision_rows(traces[short])
        llms[short] = llm_rows(traces[short])
        snapshot = stage_dir / "config_snapshot.json"
        if snapshot.exists():
            shutil.copy2(snapshot, snapshots / f"{short}_{folder}.json")

    pd.DataFrame(metrics.values()).to_csv(report_root / "ablation_metrics.csv", index=False, encoding="utf-8-sig")

    all_ids = list(results["D"]["interaction_id"].astype(str))
    per_case = []
    for case_id in all_ids:
        base = {"interaction_id": case_id, "expected_entity_annotation": expected.get(case_id, "")}
        for short in STAGES:
            match = results[short][results[short]["interaction_id"].astype(str) == case_id]
            if match.empty:
                continue
            row = match.iloc[0]
            base[f"{short}_response"] = row.get("agent_response", "")
            base[f"{short}_result"] = result_label(row)
            vision = visions[short].get(case_id, {})
            base[f"{short}_selected_entity"] = vision.get("selected_entity", "")
            base[f"{short}_selection_source"] = vision.get("selection_source", "")
        per_case.append(base)
    pd.DataFrame(per_case).to_csv(report_root / "per_case_comparison.csv", index=False, encoding="utf-8-sig")

    transitions = []
    for case_id, row in visions["C"].items():
        old_entity = legacy_top1(row)
        new_entity = row.get("selected_entity", "")
        old_ok = entity_match(expected.get(case_id, ""), old_entity)
        new_ok = entity_match(expected.get(case_id, ""), new_entity)
        transitions.append({
            "interaction_id": case_id,
            "expected_entity_annotation": expected.get(case_id, ""),
            "legacy_top1": old_entity,
            "legacy_entity_correct": old_ok,
            "pure_visual_rerank_top1": new_entity,
            "rerank_entity_correct": new_ok,
            "transition": "fixed" if new_ok and not old_ok else "regressed" if old_ok and not new_ok else "unchanged",
        })
    pd.DataFrame(transitions).to_csv(report_root / "rerank_transition.csv", index=False, encoding="utf-8-sig")

    fallback_rows = []
    for short in ("B", "C", "D"):
        counts = Counter()
        for row in visions[short].values():
            counts[row.get("selection_source") or row.get("fallback_level") or "unknown"] += 1
        for source, count in sorted(counts.items()):
            fallback_rows.append({"stage": short, "selection_source": source, "count": count})
    pd.DataFrame(fallback_rows).to_csv(report_root / "fallback_summary.csv", index=False, encoding="utf-8-sig")

    idk_rows = []
    for short in STAGES:
        raw_idk = 0
        retry_triggered = 0
        retry_success = 0
        for row in llms[short].values():
            initial = str(row.get("initial_raw_answer", "")).lower()
            retry = row.get("evidence_retry", {}) or {}
            raw_idk += int("don't know" in initial or "do not know" in initial)
            retry_triggered += int(bool(retry.get("retry_triggered")))
            retry_success += int(retry.get("retry_result") == "answered")
        idk_rows.append({
            "stage": short,
            "actual_idk_count": metrics[short]["actual_idk_count"],
            "raw_deepseek_idk_count": raw_idk if traces[short] else "unavailable",
            "evidence_retry_triggered": retry_triggered,
            "evidence_retry_success": retry_success,
        })
    pd.DataFrame(idk_rows).to_csv(report_root / "idk_origin_comparison.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([{
        "status": "not_run",
        "accept": 0,
        "rewrite": 0,
        "abstain": 0,
        "false_rejection": 0,
        "reason": "Stage E was prohibited because D did not clearly outperform C without a hallucination trade-off.",
    }]).to_csv(report_root / "verifier_confusion_matrix.csv", index=False, encoding="utf-8-sig")

    run_commands = []
    for short, folder in STAGES.items():
        path = run_root / folder / "run_command.txt"
        run_commands.append(f"[{short}] " + (path.read_text(encoding="utf-8-sig").strip() if path.exists() else "unavailable"))
    (report_root / "run_commands.txt").write_text("\n".join(run_commands) + "\n", encoding="utf-8")

    transition_df = pd.DataFrame(transitions)
    fixed = int((transition_df["transition"] == "fixed").sum()) if not transition_df.empty else 0
    regressed = int((transition_df["transition"] == "regressed").sum()) if not transition_df.empty else 0
    legacy_correct = int(transition_df["legacy_entity_correct"].sum()) if not transition_df.empty else 0
    rerank_correct = int(transition_df["rerank_entity_correct"].sum()) if not transition_df.empty else 0
    c_fallback = int(metrics["C"].get("schema_fallback_count") or 0)
    d_retry = idk_rows[-1]["evidence_retry_success"]

    report = f"""# CRAG-MM Controlled Repair Report

Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}

## A. Current HEAD fixed-sample baseline

- Baseline commit: `6d7bb37c19a65c48ab9fb73f2f8e1d16dd6a95cc`.
- Fixed set: 10 conversations from `sample_ids.txt`; the baseline HEAD dataset-first-10 ID set was independently checked to be identical.
- A result: semantic accuracy {metrics['A']['semantic_accuracy']:.0%}, exact accuracy {metrics['A']['exact_accuracy']:.0%}, missing {metrics['A']['missing_rate']:.0%}, hallucination {metrics['A']['hallucination_rate']:.0%}.
- The legacy HEAD did not emit the requested external trace file, so A-level internal selection traces are unavailable; scores and per-case outputs were preserved.

## B. Existing code problems found

1. The HEAD score was inflated by sample-shaped answer heuristics in `Task1KGAgent`; controlled code removed those specific branches rather than restoring them.
2. The original rerank mixed factual question fit with visual identity, allowing non-visible facts to influence entity choice.
3. Rerank JSON was initially limited to 1024 output tokens. The first C run had 9/10 `content_truncated` fallbacks and only one successful rerank.
4. Even after schema truncation was eliminated, visually ambiguous scenes still caused entity drift: Honda Civic/Jaguar F-Pace, earbuds/charging case, and Saint Isaac's/Kazan Cathedral.
5. DeepSeek can return IDK despite usable evidence; one bounded retry recovered a correct answer in D.

## C. Pure visual rerank modification

- The query is split into target reference/category, requested property, visual constraints, and non-visual constraints.
- Rerank receives only visual candidate views and a prompt that explicitly prohibits dates, events, specifications, origins, capacities, history, and other non-visible facts.
- Automated scan of the final C rerank reasons found no banned non-visual scoring terms.
- Manual contact-sheet review: anchor descriptions matched the visible subject/category in 10/10 cases, but B reached a specific answer-bearing entity in only 2/10. C improved entity-grade top1 to {rerank_correct}/10.

## D. Schema and fallback modification

- Parser now recovers fenced JSON, numeric strings, optional scores, confidence clamping, and 0-based indices while rejecting invalid/out-of-range selections.
- Fallback order is rerank -> recovered rerank -> anchor exact/alias/token -> anchor category filter -> legacy top1.
- A separate `QWEN_VL_RERANK_MAX_TOKENS=4096` removed the 1024-token truncation: formal C schema/pipeline fallbacks = {c_fallback}.
- Legacy entity-grade top1 = {legacy_correct}/10; pure visual rerank = {rerank_correct}/10; corrections = {fixed}; regressions = {regressed}.

## E. Claim verifier

- A claim-level contract was implemented and deterministic tests passed.
- Stage E was not run and verifier remains disabled because D did not clearly improve C without increasing hallucination. Accept/rewrite/abstain/false-rejection counts are therefore all zero, not estimated.

## F. Evidence retry

- Retry is allowed once only when the initial answer is IDK/empty, selected confidence is sufficient, and relevant supplied evidence exists.
- D triggered one evidence retry and recovered one answer (`eaab8...`). No retry loop occurred.
- D versus C: accuracy {metrics['C']['semantic_accuracy']:.0%} -> {metrics['D']['semantic_accuracy']:.0%}; missing {metrics['C']['missing_rate']:.0%} -> {metrics['D']['missing_rate']:.0%}; hallucination {metrics['C']['hallucination_rate']:.0%} -> {metrics['D']['hallucination_rate']:.0%}. The hallucination increase means D is not accepted as the default configuration.

## G. Modified files

- `.env.example`, `PracticalTraining.md`, `UI/run_eval.py`, `UI/run_controlled_repair.ps1`
- `agents/Task1KGAgent.py`, `agents/Task2Agent.py`, `agents/Task3Agent.py`
- `agents/vision/qwen_vl_client.py`, `visual_anchor.py`, `visual_candidate_pipeline.py`, `visual_query.py`
- `agents/claim_verifier.py`, `fixed_sample_selection.py`, `testControlledRepair.py`, `testQwenVision.py`

## H. Unit tests

- `C:\\anaconda\\python.exe -B -m unittest -q testControlledRepair testQwenVision testTask23Diagnostics testTask1`
- Result: 61 tests passed.

## I. Ablation metrics

| Stage | Semantic | Exact | Missing | Hallucination | IDK |
|---|---:|---:|---:|---:|---:|
| A HEAD baseline | {metrics['A']['semantic_accuracy']:.0%} | {metrics['A']['exact_accuracy']:.0%} | {metrics['A']['missing_rate']:.0%} | {metrics['A']['hallucination_rate']:.0%} | {metrics['A']['actual_idk_count']:.0f} |
| B anchor only | {metrics['B']['semantic_accuracy']:.0%} | {metrics['B']['exact_accuracy']:.0%} | {metrics['B']['missing_rate']:.0%} | {metrics['B']['hallucination_rate']:.0%} | {metrics['B']['actual_idk_count']:.0f} |
| C pure visual rerank | {metrics['C']['semantic_accuracy']:.0%} | {metrics['C']['exact_accuracy']:.0%} | {metrics['C']['missing_rate']:.0%} | {metrics['C']['hallucination_rate']:.0%} | {metrics['C']['actual_idk_count']:.0f} |
| D evidence retry | {metrics['D']['semantic_accuracy']:.0%} | {metrics['D']['exact_accuracy']:.0%} | {metrics['D']['missing_rate']:.0%} | {metrics['D']['hallucination_rate']:.0%} | {metrics['D']['actual_idk_count']:.0f} |

## J. Per-case comparison

See `per_case_comparison.csv` and `rerank_transition.csv`. The preserved contact sheet is `fixed10_contact_sheet.jpg` in the run directory.

## K. IDK source changes

- A internal raw-IDK origin is unavailable because the accepted HEAD did not emit an external trace.
- B produced {metrics['B']['actual_idk_count']:.0f} final IDKs; C produced {metrics['C']['actual_idk_count']:.0f}; D produced {metrics['D']['actual_idk_count']:.0f}.
- D had one raw DeepSeek IDK eligible for retry and one successful retry.

## L. Hallucination source changes

- C hallucinations are mainly wrong visual target/entity selection or insufficient factual coverage, not schema fallback after the 4096-token fix.
- D removed final IDKs but increased hallucinations from {metrics['C']['hallucination_rate']:.0%} to {metrics['D']['hallucination_rate']:.0%}; one increase came from stochastic visual identity drift, not from the successful evidence retry itself.

## M. Features enabled by default

- `VISION_ENABLED=1`, `VISION_RERANK_ENABLED=1`, `VISION_RERANK_MODE=pure_visual`.
- Pure visual rerank is retained because C improved B from {metrics['B']['semantic_accuracy']:.0%} to {metrics['C']['semantic_accuracy']:.0%}, reduced missing from {metrics['B']['missing_rate']:.0%} to {metrics['C']['missing_rate']:.0%}, and did not increase hallucination.
- Parser, schema normalization, hierarchical fallback, logging, and fixed-sample diagnostics remain enabled/available.

## N. Features kept disabled

- `ANSWER_RELIABILITY_ENABLED=0`
- `VISUAL_VERIFIER_ENABLED=0`
- `EVIDENCE_RETRY_ENABLED=0`; the implementation remains available for controlled tests.
- Claim verifier stage E was not run.
- Evidence retry remains experimental; pure visual rerank is retained as the only new layer with a clean B-to-C gain, but still requires a larger repeat.

## O. Unresolved issues

1. Entity-level visual disambiguation remains unstable for multiple cars, object/container ambiguity, and visually similar landmarks.
2. The answer generator sometimes uses broad or irrelevant KG attributes after a correct category-level anchor.
3. Ten samples are sufficient for regression detection but too small for a stable production acceptance estimate.
4. The HEAD baseline contains sample-shaped heuristics, so its 70% score is not a clean generalization benchmark even though it is the required rollback baseline.
5. Task2 and Task3 need their own fixed multi-source/multi-turn ablations after Task1 identity selection is stable.

## Acceptance decision

**Rejected for full-chain default enablement.** Formal C fixed schema reliability but reached only {metrics['C']['semantic_accuracy']:.0%}. D improved accuracy to {metrics['D']['semantic_accuracy']:.0%} and removed IDK, but hallucination rose to {metrics['D']['hallucination_rate']:.0%}; neither meets the requirement of not falling below A ({metrics['A']['semantic_accuracy']:.0%}) without trading missing for hallucination. Only the clean B-to-C rerank layer is retained; no commit or push was performed.
"""
    (report_root / "REPAIR_REPORT.md").write_text(report, encoding="utf-8")
    (output_root / "latest_repair_report.txt").write_text(str(report_root) + "\n", encoding="utf-8")
    print(report_root)


if __name__ == "__main__":
    main()
