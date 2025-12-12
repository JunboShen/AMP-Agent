# AMP-Agent

Multi-agent workflows for antimicrobial peptide (AMP) modeling and screening.

![AMP-Agent pipeline overview](pepagent/assets/amp_agent_pipeline.png)

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

## Notes

- Model checkpoints, large intermediate files, and local outputs are intentionally ignored by git (`.gitignore`). If you want to distribute weights, use Git LFS or a release artifact.
- Some tooling in `pepagent/agent_functions.py` expects external executables (e.g., toxicity/hemolysis predictors) to be installed and available on `PATH`.
