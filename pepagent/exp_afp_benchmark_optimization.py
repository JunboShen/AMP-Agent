#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load .env from repo root (alongside this script) so CLI runs work regardless of cwd.
_DOTENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_DOTENV_PATH, override=True)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CACHE = _SCRIPT_DIR / "workspace" / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_DEFAULT_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_DEFAULT_CACHE))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import autogen
except ModuleNotFoundError as exc:
    raise SystemExit(
        "autogen is not installed. Install it with `pip install pyautogen` before running this script."
    ) from exc

from pepagent_v2.llm_config import llm_config
from pepagent_v2.agents import strategic_planner, optimizer_agent, assistant


def build_user_proxy(auto_mode: bool):
    return autogen.UserProxyAgent(
        name="user_proxy",
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
        human_input_mode="NEVER" if auto_mode else "ALWAYS",
        system_message=(
            "user_proxy. Approve plan execution automatically when in auto mode."
            if auto_mode
            else "user_proxy. Plan execution needs to be approved by user_proxy."
        ),
        max_consecutive_auto_reply=50 if auto_mode else None,
        code_execution_config=False,
    )


def _manager_llm_config(base_config: dict | None) -> dict | None:
    """GroupChatManager cannot accept tool/function configs; strip them if present."""
    if not base_config:
        return base_config
    manager_cfg = dict(base_config)
    manager_cfg.pop("functions", None)
    manager_cfg.pop("tools", None)
    return manager_cfg


def _require_openai_key() -> None:
    # Normalize base URL env var names if provided in .env.
    base_url = (
        os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASEURL")
    )
    if base_url:
        os.environ.setdefault("OPENAI_API_BASE", base_url)
        os.environ.setdefault("OPENAI_BASE_URL", base_url)

    key = os.environ.get("OPENAI_API_KEY")
    if key and key.strip().startswith("${") and key.strip().endswith("}"):
        ref = key.strip()[2:-1].strip()
        resolved = os.environ.get(ref)
        if resolved:
            os.environ["OPENAI_API_KEY"] = resolved
            return
    if not key:
        alt = os.environ.get("AGENT_OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY")
        if alt:
            os.environ["OPENAI_API_KEY"] = alt
            return
    key = os.environ.get("OPENAI_API_KEY")
    if not key or key.strip().startswith("${"):
        raise SystemExit(
            "OPENAI_API_KEY is not set (or is a placeholder like ${AGENT_OPENAI_API_KEY}). "
            "Export a real key as OPENAI_API_KEY before running."
        )


def _advanced_args_provided(argv: list[str]) -> bool:
    advanced_flags = ("--top-k", "--optimize-top", "--max-generations")
    for arg in argv:
        for flag in advanced_flags:
            if arg == flag or arg.startswith(f"{flag}="):
                return True
    return False


def _speaker_label(msg: dict) -> str:
    role = msg.get("role", "unknown")
    name = msg.get("name")
    if name:
        return f"function:{name}" if role == "function" else name
    return role


def _format_message(msg: dict) -> str:
    lines = []
    speaker = _speaker_label(msg)
    content = msg.get("content")
    if isinstance(content, (dict, list)):
        content = json.dumps(content, ensure_ascii=False)
    if isinstance(content, str):
        cleaned = content.strip()
        if cleaned and cleaned.lower() != "none":
            lines.append(cleaned)
    function_call = msg.get("function_call")
    if isinstance(function_call, dict):
        fn = function_call.get("name")
        args = function_call.get("arguments")
        lines.append(f'```json\n{{"function_call": "{fn}", "arguments": {args}}}\n```')
    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        lines.append(f'```json\n{{"tool_calls": {tool_calls}}}\n```')
    if not lines:
        lines.append("(empty message)")
    return speaker, "\n\n".join(lines)


def _save_chat_markdown(
    messages: list[dict],
    output_path: Path,
    title: str,
    run_metadata: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    function_calls: list[str] = []
    artifacts: list[tuple[str, str]] = []
    for msg in messages:
        speaker = _speaker_label(msg)
        counts[speaker] = counts.get(speaker, 0) + 1
        if isinstance(msg.get("function_call"), dict):
            fn = msg["function_call"].get("name")
            if fn:
                function_calls.append(fn)
        if msg.get("role") == "function" and msg.get("content"):
            try:
                payload = json.loads(msg["content"])
            except Exception:
                payload = None
            if isinstance(payload, dict):
                for key in ("output_file", "output_csv", "output_fasta", "saved_csv"):
                    value = payload.get(key)
                    if isinstance(value, str):
                        artifacts.append((key, value))

    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n")
        fh.write("## Run Metadata\n\n")
        for key, value in run_metadata.items():
            fh.write(f"- {key}: {value}\n")
        fh.write("\n")
        fh.write("## Conversation Summary\n\n")
        fh.write(f"- Total messages: {len(messages)}\n")
        for speaker, count in sorted(counts.items()):
            fh.write(f"- {speaker}: {count}\n")
        if function_calls:
            unique = sorted(set(function_calls))
            fh.write(f"- Function calls: {len(function_calls)}\n")
            fh.write(f"- Functions used: {', '.join(unique)}\n")
        if artifacts:
            fh.write("\n## Artifacts\n\n")
            for key, value in artifacts:
                fh.write(f"- {key}: {value}\n")
            fh.write("\n")
        fh.write("\n## Detailed Log\n\n")
        for idx, msg in enumerate(messages, start=1):
            speaker, body = _format_message(msg)
            fh.write(f"### Message {idx} ({speaker})\n\n")
            fh.write(body + "\n\n")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_work_dir = script_dir / "workspace"
    default_input_csv = default_work_dir / "afp_screening_top800.csv"

    parser = argparse.ArgumentParser(description="Benchmark AFP optimization from top-800 candidates.")
    parser.add_argument("--input-csv", type=Path, default=default_input_csv)
    parser.add_argument("--work-dir", type=Path, default=default_work_dir)
    parser.add_argument("--min-amp-prob", type=float, default=0.50)
    parser.add_argument("--top-k", type=int, default=800)
    parser.add_argument("--optimize-top", type=int, default=80)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-generations", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument(
        "--chat-log",
        type=Path,
        default=None,
        help="Optional path to save the multi-agent discussion as markdown.",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="Require manual approval at each user_proxy turn (disable auto mode).",
    )
    args = parser.parse_args()
    advanced_provided = _advanced_args_provided(sys.argv[1:])

    _require_openai_key()

    input_csv = args.input_csv.resolve()
    work_dir = args.work_dir.resolve()
    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    user_proxy = build_user_proxy(auto_mode=not args.human)

    groupchat = autogen.GroupChat(
        agents=[user_proxy, strategic_planner, optimizer_agent, assistant],
        messages=[],
        max_round=args.max_rounds,
        speaker_selection_method="round_robin",
        allow_repeat_speaker=False,
    )
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=_manager_llm_config(llm_config),
    )

    top_k_value = args.top_k if args.top_k is not None else "omit"
    try:
        with open(input_csv, "r", encoding="utf-8") as handle:
            total_rows = max(0, sum(1 for _ in handle) - 1)
    except Exception:
        total_rows = None

    simple_message = f"""
Hi! I have a CSV of AFP candidates from screening (10-50 aa ORFs from human skin/oral/lung).
Please optimize these AFP candidates (prefer TextGrad) for a SINGLE generation only,
then report the final CSV path and top 5 sequences.
Novel AFP discovery is the goal.
Input size: {total_rows if total_rows is not None else "unknown"} rows.
Requested selection sizes: top_k={args.top_k}, optimize_top={args.optimize_top}.
Use only this local CSV: {input_csv}
Work dir: {work_dir}
Do not run remote ORF prediction or download tools.
If no candidates meet the AMP threshold, report that and stop.
Reply TERMINATE when done.
""".strip()

    detailed_message = f"""
INITIATING AFP OPTIMIZATION PROTOCOL (BENCHMARK, TEXTGRAD PREFERRED).

Context:
- Input CSV contains the top 800 AFP candidates from screening.
- Sequences are 10-50 aa ORFs from human skin/oral/lung metagenomes.
 - Input size: {total_rows if total_rows is not None else "unknown"} rows.
 - Requested selection sizes: top_k={args.top_k}, optimize_top={args.optimize_top}.

Constraints:
- Use ONLY the local CSV: {input_csv}
- Work directory: {work_dir}
- Do NOT run download_top_fastq_and_build_smorf or any remote ORF prediction tools.
- Keep sequence lengths within 10-50 aa.
- If no candidates meet AMP prob >= {args.min_amp_prob}, report that and stop.
Goal:
- Optimize the candidates using TextGrad if available (fallback genetic if needed).
- Run exactly ONE generation (no iterative loops), then rescore and finalize.
- Emphasize novelty alongside potency and safety when ranking.
Notes:
- If the input is large, prefer scaling candidate selection to match the requested sizes
  rather than using very small defaults, and briefly note the choice.

When you finish, print the final CSV path and the top 5 candidates, then reply TERMINATE.
""".strip()

    initial_message = detailed_message if advanced_provided else simple_message

    user_proxy.initiate_chat(manager, message=initial_message)

    if args.chat_log is not None:
        log_path = args.chat_log
    else:
        log_path = work_dir / "exp_afp_benchmark_optimization_chat_log.md"

    run_metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "script": Path(__file__).name,
        "input_csv": str(input_csv),
        "work_dir": str(work_dir),
        "min_amp_prob": args.min_amp_prob,
        "top_k": args.top_k,
        "optimize_top": args.optimize_top,
        "steps": args.steps,
        "max_generations": args.max_generations,
        "max_rounds": args.max_rounds,
        "model": os.environ.get("OPENAI_MODEL") or os.environ.get("OPENAI_MODEL_LIST") or "unset",
        "base_url": os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL") or "default",
    }
    _save_chat_markdown(
        messages=groupchat.messages,
        output_path=log_path,
        title="AFP Benchmark Optimization Discussion",
        run_metadata=run_metadata,
    )
    print(f"Saved chat log to: {log_path}")


if __name__ == "__main__":
    main()
