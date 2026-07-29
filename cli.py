"""
CLI for the offline N-ATLaS grammar checker.

Examples:
    python cli.py --lang yo --text "Mo je nse oko"
    python cli.py --lang ha --file draft.txt
    python cli.py --lang ig --text "..." --tier1-only   # no model weights needed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from natlas_grammar.lightweight_classifier import LightweightGate
from natlas_grammar.llm_corrector import NAtlasCorrector
from natlas_grammar.orchestrator import GrammarChecker


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_semantic_matcher(config: dict):
    """
    Builds one shared SemanticMatcher instance (or None). Build this ONCE and
    pass it into build_checker() for every language — constructing it per
    language would load the ~1.8GB LaBSE model multiple times over.
    """
    sem_cfg = config.get("semantic", {})
    if not sem_cfg.get("enabled"):
        return None
    from natlas_grammar.semantic import SemanticMatcher  # lazy import

    try:
        return SemanticMatcher(cache_dir=sem_cfg.get("cache_dir", "resources/embedding_cache"))
    except RuntimeError as e:
        print(
            f"[warn] {e}\nContinuing without semantic matching "
            "(closest-meaning lookup and meaning-drift guard disabled).",
            file=sys.stderr,
        )
        return None


def build_checker(
    lang_code: str, config: dict, tier1_only: bool, semantic_matcher=None
) -> GrammarChecker:
    lang_cfg = config["languages"][lang_code]
    tier1_cfg = config["tier1"]
    sem_cfg = config.get("semantic", {})

    gate = LightweightGate(
        lang_code=lang_code,
        lexicon_path=lang_cfg.get("lexicon_path"),
        oov_ratio_flag_threshold=tier1_cfg["oov_ratio_flag_threshold"],
        kenlm_path=lang_cfg.get("kenlm_path"),
        perplexity_zscore_flag_threshold=tier1_cfg["perplexity_zscore_flag_threshold"],
        semantic_matcher=semantic_matcher,
        semantic_oov_accept_similarity=sem_cfg.get("oov_accept_similarity", 0.82),
        fuzzy_match_enabled=tier1_cfg.get("fuzzy_match_enabled", True),
        fuzzy_match_threshold=tier1_cfg.get("fuzzy_match_threshold", 0.82),
    )

    corrector = None
    if not tier1_only:
        llm_cfg = config["llm"]
        try:
            corrector = NAtlasCorrector(
                model_path=llm_cfg["model_path"],
                n_ctx=llm_cfg.get("n_ctx", 8092),
                n_threads=llm_cfg.get("n_threads"),
                n_gpu_layers=llm_cfg.get("n_gpu_layers", 0),
                temperature=llm_cfg["temperature"],
                repeat_penalty=llm_cfg["repeat_penalty"],
                max_new_tokens=llm_cfg["max_new_tokens"],
            )
        except RuntimeError as e:
            print(
                f"[warn] {e}\n"
                "Running Tier-1-only (rule/lexicon flags, no corrections) instead.",
                file=sys.stderr,
            )
            corrector = None

    return GrammarChecker(
        lang_code=lang_code,
        gate=gate,
        corrector=corrector,
        auto_apply_threshold=lang_cfg.get("auto_apply_threshold"),
        semantic_matcher=semantic_matcher,
        drift_guard_similarity=sem_cfg.get("drift_guard_similarity") if semantic_matcher else None,
    )


def main():
    parser = argparse.ArgumentParser(description="Offline N-ATLaS grammar checker")
    parser.add_argument("--lang", required=True, choices=["yo", "ig", "ha"])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Text to check")
    src.add_argument("--file", help="Path to a text file to check")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--tier1-only",
        action="store_true",
        help="Skip the LLM tier entirely — just show what Tier 1 would flag",
    )
    parser.add_argument(
        "--apply-all",
        action="store_true",
        help="Also print the full text with every Tier-2 suggestion accepted "
        "(like clicking 'accept all' in Grammarly). Review it — this is not "
        "the same as the auto-apply threshold, which is more conservative.",
    )
    args = parser.parse_args()

    text = args.text if args.text else Path(args.file).read_text(encoding="utf-8")
    config = load_config(args.config)
    semantic_matcher = None if args.tier1_only else build_semantic_matcher(config)
    checker = build_checker(args.lang, config, args.tier1_only, semantic_matcher=semantic_matcher)

    results = checker.check(text)

    for r in results:
        if not r.escalated:
            continue
        print(f"\n> {r.sentence}")
        if not r.has_error:
            if args.tier1_only or "not configured" in r.explanation:
                reasons = ", ".join(r.gate_reasons) if r.gate_reasons else "n/a"
                print(f"  (flagged by Tier 1 — reasons: {reasons})")
                print("   Tier 2/N-ATLaS was not run. Use without --tier1-only and provide a model to get a correction.")
            else:
                print("  (flagged by Tier 1, but Tier 2 reviewed it and found no correction needed)")
            continue
        print(f"  Suggested: {r.corrected_sentence}")
        for e in r.edits:
            if e.op == "replace":
                print(f"    ~ '{e.original}' -> '{e.suggestion}'")
            elif e.op == "delete":
                print(f"    - remove '{e.original}'")
            elif e.op == "insert":
                print(f"    + insert '{e.suggestion}'")
        status = "auto-applied" if r.auto_applied else "suggested (needs review)"
        print(f"  [{status}] {r.explanation}")
        if r.semantic_similarity is not None:
            print(f"  (meaning similarity to original: {r.semantic_similarity:.2f})")

    flagged_none = all(not r.escalated for r in results)
    if flagged_none:
        print("No issues flagged.")

    if args.apply_all:
        rebuilt = " ".join(
            r.corrected_sentence if (r.has_error and r.corrected_sentence) else r.sentence
            for r in results
        )
        print("\n--- Full corrected text (all suggestions accepted) ---")
        print(rebuilt)


if __name__ == "__main__":
    main()