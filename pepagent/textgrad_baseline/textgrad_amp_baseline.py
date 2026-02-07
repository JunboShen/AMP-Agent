#!/usr/bin/env python3
"""
TextGrad baseline for AMP candidate selection + sequence optimization.

Usage example:
  python textgrad_amp_baseline.py \
    --fasta ./data/smorfs_30_100aa_part.faa \
    --engine gpt-5.2 \
    --candidate-count 40 \
    --optimize-top 8 \
    --steps 2
    # optionally:
    # --api-key "$OPENAI_API_KEY" --base-url "https://api.openai.com/v1"

Notes:
- You must set provider credentials in environment variables expected by TextGrad/LiteLLM.
- This script ranks candidates via an LLM scoring prompt and optionally refines sequences
  with TextGrad (textual gradients) for improved antimicrobial activity.
- If a model name is not natively supported by TextGrad, the script will route it through
  LiteLLM by prefixing it with "experimental:".
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    import textgrad as tg
    from textgrad.engine import get_engine
except Exception as exc:  # pragma: no cover - import guard for runtime only
    raise SystemExit(
        "textgrad is required. Install with: pip install textgrad\n"
        f"Import error: {exc}"
    ) from exc

# Provider configuration (placeholders)
# Prefer setting these in your environment. You can also edit the placeholders
# directly or pass --api-key/--base-url at runtime.
API_KEY_PLACEHOLDER = "YOUR_API_KEY_HERE"
BASE_URL_PLACEHOLDER = "https://api.example.com/v1"
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", API_KEY_PLACEHOLDER)
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", BASE_URL_PLACEHOLDER)

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMFWVYC")  # common hydrophobic residues for AMP heuristics


@dataclass
class Candidate:
    seq_id: str
    sequence: str
    method: str
    parent_id: str
    length: int
    net_charge: float
    hydrophobic_frac: float
    heuristic_score: float
    llm_score: float
    llm_rationale: str


def normalize_sequence(seq: str) -> str:
    seq = re.sub(r"\s+", "", seq.upper())
    return "".join(aa for aa in seq if aa in AA20)


def parse_fasta(path: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header = None
    seq_parts: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    seq = normalize_sequence("".join(seq_parts))
                    records.append((header, seq))
                header = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            seq = normalize_sequence("".join(seq_parts))
            records.append((header, seq))
    return records


def heuristic_features(seq: str) -> Tuple[int, float, float, float]:
    length = len(seq)
    if length == 0:
        return 0, 0.0, 0.0, 0.0
    pos = seq.count("K") + seq.count("R") + 0.1 * seq.count("H")
    neg = seq.count("D") + seq.count("E")
    net_charge = pos - neg
    hydrophobic = sum(seq.count(aa) for aa in HYDROPHOBIC)
    hydrophobic_frac = hydrophobic / length

    # Simple heuristic: prefer ~40 aa, cationic, and moderate hydrophobicity
    length_score = 1.0 - min(abs(length - 40), 40) / 40.0
    charge_score = max(min((net_charge + 1.0) / 10.0, 1.0), 0.0)
    hydro_score = 1.0 - min(abs(hydrophobic_frac - 0.45), 0.45) / 0.45
    heuristic_score = 0.35 * charge_score + 0.35 * hydro_score + 0.30 * length_score
    return length, net_charge, hydrophobic_frac, heuristic_score


def parse_llm_score(text: str) -> Tuple[float, str]:
    text = text.strip()
    rationale = ""
    score = None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            score = float(data.get("score"))
            rationale = str(data.get("rationale", "")).strip()
        except Exception:
            score = None
    if score is None:
        num = re.findall(r"(\d+(?:\.\d+)?)", text)
        if num:
            score = float(num[0])
    if score is None:
        score = 0.0
    score = max(0.0, min(100.0, score))
    return score, rationale


def llm_score_sequence(seq: str, scorer: tg.BlackboxLLM) -> Tuple[float, str]:
    prompt = (
        "Score the following peptide for antimicrobial activity (0-100, higher is better). "
        "Consider cationic charge, amphipathicity, hydrophobicity balance, stability, "
        "and likely toxicity. Respond with JSON only: "
        "{\"score\": <number>, \"rationale\": \"<one sentence>\", \"risks\": \"<short>\"}.\n\n"
        f"Sequence:\n{seq}"
    )
    out = scorer(tg.Variable(prompt, requires_grad=False, role_description="scoring prompt"))
    return parse_llm_score(out.get_value())


def optimize_sequence(
    seq: str,
    engine,
    steps: int,
    constraints: List[str],
    role_description: str,
) -> str:
    seq_var = tg.Variable(seq, requires_grad=True, role_description=role_description)
    eval_instruction = tg.Variable(
        "You are evaluating an antimicrobial peptide sequence. Provide concise, actionable "
        "feedback to improve antimicrobial potency and spectrum while maintaining stability "
        "and minimizing toxicity. Focus on cationic charge and amphipathic structure. "
        "Do NOT rewrite the sequence; only provide feedback.",
        requires_grad=False,
        role_description="evaluation criteria for AMP optimization",
    )
    loss_fn = tg.TextLoss(eval_instruction, engine=engine)
    optimizer = tg.TextualGradientDescent(
        parameters=[seq_var],
        engine=engine,
        constraints=constraints,
        verbose=0,
    )

    target_len = len(seq)
    for _ in range(steps):
        loss = loss_fn(seq_var)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        cleaned = normalize_sequence(seq_var.get_value())
        if len(cleaned) != target_len:
            cleaned = seq  # keep original if optimizer violates length/charset
        seq_var.set_value(cleaned)

    return seq_var.get_value()


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_fasta(path: str, rows: List[Dict[str, object]], top_k: int) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows[:top_k]:
            header = f">{row['rank']}|{row['method']}|{row['seq_id']}"
            handle.write(f"{header}\n{row['sequence']}\n")


def resolve_engine_name(engine_name: str) -> str:
    native = (
        engine_name.startswith("experimental:")
        or engine_name.startswith("azure")
        or engine_name.startswith("ollama")
        or engine_name.startswith("vllm")
        or engine_name.startswith("groq")
        or engine_name.startswith("together-")
        or "gpt-4" in engine_name
        or "gpt-3.5" in engine_name
        or "gpt-35" in engine_name
        or "claude" in engine_name
        or "gemini" in engine_name
        or engine_name in {"command-r-plus", "command-r", "command", "command-light"}
    )
    if native:
        return engine_name
    return f"experimental:{engine_name}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default="./data/smorfs_30_100aa_part.faa")
    ap.add_argument("--engine", default=os.environ.get("TEXTGRAD_ENGINE", "gpt-5.2"))
    ap.add_argument("--api-key", default=DEFAULT_API_KEY)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--candidate-count", type=int, default=40)
    ap.add_argument("--optimize-top", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--out-csv", default="textgrad_amp_baseline.csv")
    ap.add_argument("--out-fasta", default="textgrad_amp_baseline_top10.fasta")
    args = ap.parse_args()

    if not args.api_key or args.api_key == API_KEY_PLACEHOLDER:
        raise SystemExit(
            "Missing API key. Set OPENAI_API_KEY, edit API_KEY_PLACEHOLDER, "
            "or pass --api-key."
        )
    os.environ["OPENAI_API_KEY"] = args.api_key
    if args.base_url and args.base_url != BASE_URL_PLACEHOLDER:
        os.environ["OPENAI_BASE_URL"] = args.base_url

    records = parse_fasta(args.fasta)
    filtered = []
    for seq_id, seq in records:
        length, net_charge, hydrophobic_frac, h_score = heuristic_features(seq)
        if not (args.min_len <= length <= args.max_len):
            continue
        filtered.append(
            (seq_id, seq, length, net_charge, hydrophobic_frac, h_score)
        )
    if not filtered:
        raise SystemExit("No sequences remain after length filtering.")

    filtered.sort(key=lambda x: x[5], reverse=True)
    candidates = filtered[: args.candidate_count]

    engine_name = resolve_engine_name(args.engine)
    if engine_name.startswith("experimental:"):
        engine = get_engine(engine_name, cache=False)
    else:
        engine = get_engine(engine_name)
    tg.set_backward_engine(engine)

    score_system_prompt = tg.Variable(
        "You are a strict antimicrobial peptide evaluator. "
        "Output only JSON with numeric score and a one-sentence rationale.",
        requires_grad=False,
        role_description="system prompt for AMP scoring",
    )
    scorer = tg.BlackboxLLM(engine=engine, system_prompt=score_system_prompt)

    scored_candidates: List[Candidate] = []
    for seq_id, seq, length, net_charge, hydrophobic_frac, h_score in candidates:
        llm_score, llm_rationale = llm_score_sequence(seq, scorer)
        scored_candidates.append(
            Candidate(
                seq_id=seq_id,
                sequence=seq,
                method="textgrad_original",
                parent_id=seq_id,
                length=length,
                net_charge=net_charge,
                hydrophobic_frac=hydrophobic_frac,
                heuristic_score=h_score,
                llm_score=llm_score,
                llm_rationale=llm_rationale,
            )
        )

    scored_candidates.sort(key=lambda c: c.llm_score, reverse=True)
    to_optimize = scored_candidates[: min(args.optimize_top, len(scored_candidates))]

    optimized_candidates: List[Candidate] = []
    for cand in to_optimize:
        constraints = [
            f"Output only uppercase amino-acid letters (ACDEFGHIKLMNPQRSTVWY) with length exactly {cand.length}.",
            "No spaces, punctuation, or extra text.",
        ]
        role_desc = (
            f"Antimicrobial peptide sequence (length {cand.length}) "
            "using only canonical amino acids."
        )
        improved = optimize_sequence(
            cand.sequence, engine=engine, steps=args.steps, constraints=constraints, role_description=role_desc
        )
        length, net_charge, hydrophobic_frac, h_score = heuristic_features(improved)
        llm_score, llm_rationale = llm_score_sequence(improved, scorer)
        optimized_candidates.append(
            Candidate(
                seq_id=f"{cand.seq_id}_opt",
                sequence=improved,
                method="textgrad_optimized",
                parent_id=cand.seq_id,
                length=length,
                net_charge=net_charge,
                hydrophobic_frac=hydrophobic_frac,
                heuristic_score=h_score,
                llm_score=llm_score,
                llm_rationale=llm_rationale,
            )
        )

    all_candidates = scored_candidates + optimized_candidates
    all_candidates.sort(key=lambda c: c.llm_score, reverse=True)

    rows: List[Dict[str, object]] = []
    for idx, cand in enumerate(all_candidates, start=1):
        rows.append(
            {
                "rank": idx,
                "seq_id": cand.seq_id,
                "parent_id": cand.parent_id,
                "method": cand.method,
                "sequence": cand.sequence,
                "length": cand.length,
                "net_charge": round(cand.net_charge, 3),
                "hydrophobic_frac": round(cand.hydrophobic_frac, 4),
                "heuristic_score": round(cand.heuristic_score, 4),
                "llm_score": round(cand.llm_score, 2),
                "llm_rationale": cand.llm_rationale,
            }
        )

    write_csv(args.out_csv, rows)
    write_fasta(args.out_fasta, rows, top_k=10)

    print(f"[OK] wrote {args.out_csv}")
    print(f"[OK] wrote {args.out_fasta}")
    print("\nTop 10 candidates:")
    for row in rows[:10]:
        print(
            f"{row['rank']:>2} | {row['method']:<18} | {row['seq_id']:<20} "
            f"| score={row['llm_score']:>6} | len={row['length']} | seq={row['sequence']}"
        )


if __name__ == "__main__":
    main()
