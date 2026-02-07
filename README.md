# AMP-Agent

Multi-agent workflows for antimicrobial peptide (AMP) and anti-fungal peptide (AFP) modeling, screening, and optimization.

<img src="pepagent/assets/amp_agent_pipeline.png" alt="AMP-Agent pipeline overview" width="900" />

This project uses an AutoGen-style agent team (Planner/Critic/ML_Coder/Executor/Assistant) plus a toolbox of vetted Python functions to:

- generate or refine AMP training/inference code (LLM-assisted),
- screen peptide FASTA files for AMP/AFP likelihood and predicted MIC,
- annotate candidates with toxicity/hemolysis signals,
- compare candidates to known verified AFP/AMP sets via embedding and sequence identity signals,
- run ranking pipelines that integrate potency, safety, and novelty,
- run textgrad-based optimization baselines and multi-agent optimization loops,
- support an end-to-end “metagenomics → smORFs → mature peptides → AMP/MIC” workflow.

The high-level pipeline is captured in `pepagent/workspace/amp_pipeline_diagram.mmd` (Mermaid).

## Demos (notebooks)

- `pepagent/code_exp.ipynb`: example “coder agent” workflow for generating AMP model training/inference code.
- `pepagent/exp1.ipynb`: example screening workflow (FASTA -> AMP candidates -> MIC prediction -> optional safety/novelty analysis).

Notebooks are checked in **without outputs** and read credentials from environment variables.

## Quickstart

1) Create and activate an environment (Python 3.10+ recommended).
2) Configure credentials and paths:

- Copy `.env.example` → `.env` (do not commit) and fill values, or export env vars in your shell.
- Required: `OPENAI_API_KEY`
- Optional: `OPENAI_BASE_URL`, `HF_TOKEN`, `PEPAGENT_AMP_MODEL_DIR`, `PEPAGENT_MIC_MODEL_DIR`

3) Run script workflows (recommended):

```bash
python pepagent/exp_amp.py
python pepagent/exp_afp_benchmark_screening.py --input-fasta <input.faa> --top-k 100
python pepagent/exp2_afp_benchmark_screening.py --input-fasta <input.faa> --top-k 600
python pepagent/exp_afp_benchmark_optimization.py --input-csv <screened.csv>
python pepagent/exp2_afp_benchmark_optimization.py --input-csv <screened.csv>
python pepagent/exp_afp_metagenomics.py --input-fasta <large_input.faa>
```

4) Optional: open notebooks for interactive experiments:

- `pepagent/code_exp.ipynb`
- `pepagent/exp1.ipynb`

## Configuration

Environment variables (see `.env.example`):

- `OPENAI_API_KEY`: required for LLM-backed agents.
- `OPENAI_BASE_URL`: optional; defaults to OpenAI if unset.
- `OPENAI_API_BASE`: optional alias for base URL.
- `HF_TOKEN`: optional; used for Hugging Face model downloads when needed.
- `PEPAGENT_AMP_MODEL_DIR`: optional; directory containing AMP classifier checkpoint artifacts.
- `PEPAGENT_MIC_MODEL_DIR`: optional; directory containing MIC regressor checkpoint artifacts.
- `PEPAGENT_WORKSPACE_DIR`: optional; where tools write intermediate files (default: `pepagent/workspace`).

LLM model behavior:

- `pepagent/llm_config.py` currently enforces a strict model (`gpt-5.2`) for all agents by default.
- If you need a different model policy, update `pepagent/llm_config.py` intentionally.

## Repo layout

- `pepagent/agents.py`: agent definitions and orchestration wiring.
- `pepagent/llm_config.py`: LLM/tool configuration (no credentials in-code).
- `pepagent/agent_functions.py`: toolbox functions callable by agents (I/O helpers, screening, ranking, optimization utilities).
- `pepagent/exp_amp.py`: AMP discovery experiment entry point.
- `pepagent/exp_afp_benchmark_screening.py`, `pepagent/exp2_afp_benchmark_screening.py`: AFP screening benchmark scripts.
- `pepagent/exp_afp_benchmark_optimization.py`, `pepagent/exp2_afp_benchmark_optimization.py`: AFP optimization benchmark scripts.
- `pepagent/exp_afp_metagenomics.py`: AFP discovery workflow for metagenomics-scale inputs.
- `pepagent/textgrad_baseline/textgrad_amp_baseline.py`: textgrad AMP optimization baseline.
- `pepagent/data/Verified_AFPs_Database_cleaned.fasta`: verified AFP reference database used for novelty/similarity scoring.
- `pepagent/workspace/`: local scratch space (ignored by git except the Mermaid diagram + README).
- `pepagent/USPNet/`: USPNet signal peptide prediction package (see details below; licensed under `pepagent/USPNet/LICENSE`).

## USPNet (signal peptide prediction)

`pepagent/USPNet/` contains **USPNet / USPNet-fast**, a signal peptide predictor (Nature Computational Science, 2024).
AMP-Agent uses it to infer cleavage sites and trim proteins/smORFs into *mature peptide* candidates before downstream AMP screening.

See `pepagent/USPNet/README.md` for full usage instructions and the paper citation.

## Model files (checkpoints)

This repo does not ship large model checkpoints by default (see `.gitignore`).

- AMP classifier: set `PEPAGENT_AMP_MODEL_DIR` to a local directory containing the required checkpoint artifacts (for example `tokenizer/`, `lora_adapter/`, and `classifier_weights.pth`).
- MIC regressor: set `PEPAGENT_MIC_MODEL_DIR` similarly (for example `tokenizer/`, `lora_adapter/`, and `regression_head.pth`).

If you need access to pretrained/fine-tuned checkpoints used in our experiments, please contact the repository owner via email.

## Data and outputs

- Example/small data may live under `pepagent/data/`.
- Intermediate outputs (CSV/FASTA/logs/chat transcripts) are written to `pepagent/workspace/` by default and are intentionally ignored by git.
- For large AFP/AMP screening and optimization experiments, prefer script execution over notebooks for reproducibility.

## Practical workflow notes

- Screening flows typically run: AMP+MIC prediction -> toxicity/hemolysis augmentation -> novelty/similarity scoring -> composite ranking.
- Optimization flows typically run generation (textgrad/genetic), then the same screening/ranking stack on generated candidates.
- Safety is handled as a ranking objective (not always hard filtering), so final candidate sets can preserve potency/novelty trade-offs.

## References

1. Yuksekgonul, M., et al. *Optimizing generative AI by backpropagating language model feedback.* Nature 639(8055):609-616 (2025). [https://doi.org/10.1038/s41586-025-08661-4](https://doi.org/10.1038/s41586-025-08661-4)
2. Rathore, A. S., et al. *ToxinPred3.0: An improved method for predicting the toxicity of peptides.* Computers in Biology and Medicine 179:108926 (2024). [https://doi.org/10.1016/j.compbiomed.2024.108926](https://doi.org/10.1016/j.compbiomed.2024.108926)
3. Rathore, A. S., et al. *Prediction of hemolytic peptides and their hemolytic concentration.* Communications Biology 8:176 (2025). [https://doi.org/10.1038/s42003-025-07615-w](https://doi.org/10.1038/s42003-025-07615-w)
4. Lin, Z., et al. *Evolutionary-scale prediction of atomic-level protein structure with a language model.* Science 379(6637):1123-1130 (2023). [https://doi.org/10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)
5. Shen, J., et al. *Unbiased organism-agnostic and highly sensitive signal peptide predictor with deep protein language model.* Nature Computational Science 4(1):29-42 (2024). [https://doi.org/10.1038/s43588-023-00576-2](https://doi.org/10.1038/s43588-023-00576-2)

## Contact

- For questions, please open a GitHub Issue.
- For checkpoint/data files download requests: contact the GitHub repository owner via email.

## License

- Project license: **MIT** (see `LICENSE`).
- Third-party components: `pepagent/USPNet/` is distributed with its own license (`pepagent/USPNet/LICENSE`).

## Notes

- Model checkpoints, large intermediate files, and local outputs are intentionally ignored by git (`.gitignore`). If you want to distribute weights, use Git LFS or a release artifact.
- Some tooling in `pepagent/agent_functions.py` expects external executables (e.g., toxicity/hemolysis predictors) to be installed and available on `PATH`.
