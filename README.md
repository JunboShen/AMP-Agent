# AMP-Agent

Multi-agent workflows for antimicrobial peptide (AMP) modeling and screening.

<img src="pepagent/assets/amp_agent_pipeline.png" alt="AMP-Agent pipeline overview" width="900" />

This project uses an AutoGen-style agent team (Planner/Critic/ML_Coder/Executor/Assistant) plus a toolbox of vetted Python functions to:

- generate or refine AMP training/inference code (LLM-assisted),
- screen peptide FASTA files for AMP likelihood and predicted MIC,
- annotate candidates with toxicity/hemolysis signals (optional),
- compare candidates to known AMPs (e.g., via DBAASP lookup + similarity metrics),
- support an end-to-end “metagenomics → smORFs → mature peptides → AMP/MIC” workflow.

The high-level pipeline is captured in `pepagent/workspace/amp_pipeline_diagram.mmd` (Mermaid).

## Demos (notebooks)

- `pepagent/code_exp.ipynb`: example “coder agent” workflow for generating AMP model training/inference code (e.g., ESM-family fine-tuning).
- `pepagent/amp_discover_exp1.ipynb`: example screening workflow (FASTA → AMP candidates → MIC prediction → optional safety filters / analysis).

Notebooks are checked in **without outputs** and read credentials from environment variables.

## Quickstart

1) Create and activate an environment (Python 3.10+ recommended).
2) Configure credentials and paths:

- Copy `.env.example` → `.env` (do not commit) and fill values, or export env vars in your shell.
- Required: `OPENAI_API_KEY`
- Optional: `HF_TOKEN` (Hugging Face), `PEPAGENT_AMP_MODEL_DIR`, `PEPAGENT_MIC_MODEL_DIR`

3) Open the notebooks:

- `pepagent/code_exp.ipynb`
- `pepagent/exp1.ipynb`

## Configuration

Environment variables (see `.env.example`):

- `OPENAI_API_KEY`: required for LLM-backed agents.
- `OPENAI_BASE_URL`: optional; defaults to OpenAI if unset.
- `HF_TOKEN`: optional; used for Hugging Face model downloads when needed.
- `PEPAGENT_AMP_MODEL_DIR`: optional; directory containing AMP classifier checkpoint artifacts.
- `PEPAGENT_MIC_MODEL_DIR`: optional; directory containing MIC regressor checkpoint artifacts.
- `PEPAGENT_WORKSPACE_DIR`: optional; where tools write intermediate files (default: `pepagent/workspace`).

## Repo layout

- `pepagent/agents.py`: agent definitions and orchestration wiring.
- `pepagent/llm_config.py`: LLM/tool configuration (no credentials in-code).
- `pepagent/agent_functions.py`: “toolbox” functions callable by agents (I/O helpers, screening utilities, etc.).
- `pepagent/workspace/`: local scratch space (ignored by git except the Mermaid diagram + README).
- `pepagent/USPNet/`: signal peptide prediction code (third-party; see `pepagent/USPNet/LICENSE`).

## Model files (checkpoints)

This repo does not ship large model checkpoints by default (see `.gitignore`).

- AMP classifier: set `PEPAGENT_AMP_MODEL_DIR` to a local directory containing the required checkpoint artifacts (e.g., `tokenizer/`, `lora_adapter/`, and weights such as `classifier_weights.pth`).
- MIC regressor: set `PEPAGENT_MIC_MODEL_DIR` similarly (e.g., `tokenizer/`, `lora_adapter/`, and `regression_head.pth`).

If you need access to pretrained/fine-tuned checkpoints used in our experiments, please contact the maintainers.

## Data and outputs

- Example/small data may live under `pepagent/data/`.
- Intermediate outputs (CSVs/FASTA/logs) are written to `pepagent/workspace/` by default and are intentionally ignored by git.

## Contact

- For questions, please open a GitHub Issue.
- For checkpoint/data files download requests: contact the GitHub repository owner via email.

## License

- Project license: **MIT** (see `LICENSE`).
- Third-party components: `pepagent/USPNet/` is distributed with its own license (`pepagent/USPNet/LICENSE`).

## Notes

- Model checkpoints, large intermediate files, and local outputs are intentionally ignored by git (`.gitignore`). If you want to distribute weights, use Git LFS or a release artifact.
- Some tooling in `pepagent/agent_functions.py` expects external executables (e.g., toxicity/hemolysis predictors) to be installed and available on `PATH`.
