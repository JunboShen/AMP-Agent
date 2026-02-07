#!/usr/bin/env python3
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pepagent_v2 import agents
import autogen


def _reset_agents():
    agents.user_proxy.reset()
    agents.planner.reset()
    agents.critic.reset()
    agents.assistant.reset()
    if agents.codex_coder is not None:
        agents.codex_coder.reset()


def main() -> None:
    os.environ.setdefault("AUTOGEN_USE_DOCKER", "0")

    if agents.codex_coder is None:
        raise SystemExit("Codex_Coder is not available. Check ./coding_agent.py and MCP setup.")

    _reset_agents()

    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        system_message="user_proxy. Auto-approve plan execution.",
        max_consecutive_auto_reply=40,
        code_execution_config=False,
    )

    manager_llm_config = {
        "config_list": agents.llm_config.get("config_list", []),
        "seed": 45,
    }

    groupchat = autogen.GroupChat(
        agents=[
            user_proxy,
            agents.planner,
            agents.critic,
            agents.assistant,
            agents.codex_coder,
        ],
        messages=[],
        max_round=120,
    )
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=manager_llm_config,
        system_message=(
            "This manager selects a speaker, collects responses, and broadcasts them."
        ),
    )

    user_proxy.initiate_chat(
        recipient=manager,
        message=(
            "Goal: test the coding workflow using general agents + Codex_Coder only.\n"
            "Use Codex_Coder with codex_mcp_run for all code edits and shell commands.\n"
            "Do NOT embed code blocks in codex_mcp_run prompts; describe the task and file path instead.\n"
            "Task:\n"
            "1) Create ./workspace/train_amp_classifier.py with full training code for ESM2 AMP classification.\n"
            "   It must accept CLI args: --positive_fasta, --negative_fasta, --output_dir, --epochs.\n"
            "   Include FASTA loading, tokenizer/model setup (Transformers), training loop, basic validation,\n"
            "   and save model/tokenizer to output_dir.\n"
            "2) Create ./workspace/infer_amp_classifier.py with full inference code.\n"
            "   It must accept CLI args: --model_dir, --input_fasta, --output_csv.\n"
            "   Load model/tokenizer, score sequences, and write CSV with seq_id, sequence, amp_probability, pred_label.\n"
            "3) Run: python -m py_compile ./workspace/train_amp_classifier.py ./workspace/infer_amp_classifier.py\n"
            "4) Do NOT run training or inference (compilation only).\n"
            "Return the compile output and reply TERMINATE."
        ),
    )


if __name__ == "__main__":
    main()
