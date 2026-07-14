from typing import Any, Dict, Tuple


_IDENTITY = {"supported", "contradicted", "uncertain"}
_KNOWLEDGE = {"supported", "contradicted", "insufficient"}
_COVERAGE = {"complete", "partial", "off_target"}
_DECISIONS = {"accept", "rewrite", "abstain"}


def normalize_claim_verdict(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Validate claim-level verifier output without treating uncertainty as rejection."""
    if not isinstance(raw, dict):
        return {}, "verdict_not_object"
    identity = str(raw.get("identity_status", "uncertain")).strip().lower()
    knowledge = str(raw.get("knowledge_status", "insufficient")).strip().lower()
    coverage = str(raw.get("coverage_status", "partial")).strip().lower()
    decision = str(raw.get("decision", "accept")).strip().lower()
    if identity not in _IDENTITY:
        return {}, "invalid_identity_status"
    if knowledge not in _KNOWLEDGE:
        return {}, "invalid_knowledge_status"
    if coverage not in _COVERAGE:
        return {}, "invalid_coverage_status"
    if decision not in _DECISIONS:
        return {}, "invalid_decision"

    # Insufficient knowledge or verifier uncertainty alone cannot force IDK.
    if decision == "abstain" and identity != "contradicted" and knowledge != "contradicted":
        decision = "rewrite" if coverage in {"partial", "off_target"} else "accept"
    return {
        "identity_status": identity,
        "knowledge_status": knowledge,
        "coverage_status": coverage,
        "decision": decision,
        "reasons": [str(item)[:240] for item in raw.get("reasons", []) if str(item).strip()][:8],
        "supported_claims": [str(item)[:240] for item in raw.get("supported_claims", []) if str(item).strip()][:12],
        "unsupported_claims": [str(item)[:240] for item in raw.get("unsupported_claims", []) if str(item).strip()][:12],
        "contradicted_claims": [str(item)[:240] for item in raw.get("contradicted_claims", []) if str(item).strip()][:12],
        "corrected_answer": str(raw.get("corrected_answer", "")).strip()[:1200],
    }, ""


def apply_claim_verdict(candidate_answer: str, verdict: Dict[str, Any]) -> Tuple[str, str]:
    """Apply only contract-safe decisions; caller controls whether verifier is enabled."""
    decision = verdict.get("decision", "accept")
    corrected = str(verdict.get("corrected_answer", "")).strip()
    if decision == "accept":
        return candidate_answer, "verifier_accept"
    if decision == "rewrite" and corrected:
        return corrected, "verifier_rewrite"
    if decision == "abstain" and (
        verdict.get("identity_status") == "contradicted"
        or verdict.get("knowledge_status") == "contradicted"
    ):
        return "I don't know.", "verifier_hard_contradiction"
    return candidate_answer, "verifier_no_destructive_change"
