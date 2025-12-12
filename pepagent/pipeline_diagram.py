#!/usr/bin/env python3
"""
Generate Mermaid diagrams summarizing the multi-agent workflow and AMP discovery pipeline.

Output:
  workspace/amp_pipeline_diagram.mmd  (render with any Mermaid viewer or https://mermaid.live)
"""
from pathlib import Path
from textwrap import dedent
from datetime import datetime


ROOT = Path(__file__).parent
OUT_FILE = ROOT / "workspace" / "amp_pipeline_diagram.mmd"


def build_content() -> str:
    """Assemble the Markdown + Mermaid content."""
    agent_diagram = dedent(
        r"""
        ```mermaid
        flowchart LR
            subgraph Orchestration
                U[Human PI / user_proxy]
                P[Planner<br/>(study design + tool selection)]
                C[Critic<br/>(methodological sanity check)]
                M[ML_Coder<br/>(authors training/inference code)]
                E[Executor<br/>(runs code in sandboxed workspace)]
                S[Assistant<br/>(calls vetted tools + prepares inputs)]
            end
            subgraph Tools["agent_functions.py analysis toolkit"]
                T1[download_top_fastq_and_build_smorf<br/>(ENA taxon → smORFs)]
                T2[write_mature_faa_by_predicted_cleavage<br/>(USPNet signal peptide trimming)]
                T3[amp_predict_fasta / amp_then_mic_from_fasta<br/>(ESM2+LoRA AMP triage)]
                T4[mic_predict_csv<br/>(ESM2+LoRA MIC regression)]
                T5[augment_with_toxicity_and_hemolysis<br/>(ToxinPred3 + HemoPI2)]
                T6[amp_esm_similarity_and_sequence_identity<br/>(ESM embeddings + identity)]
                T7[fetch_amps<br/>(DBAASP reference set)]
            end

            U --> P
            P --> C
            C -->|feedback| P
            P --> M
            M --> E
            E --> S
            P --> S
            S --> T1
            S --> T2
            S --> T3
            S --> T4
            S --> T5
            S --> T6
            S --> T7
            T1 -.-> S
            T2 -.-> S
            T3 -.-> S
            T4 -.-> S
            T5 -.-> S
            T6 -.-> S
            T7 -.-> S
            S --> U
        ```
        """
    ).strip()

    data_pipeline = dedent(
        r"""
        ```mermaid
        flowchart TD
            R[Metagenomic short reads<br/>(ENA taxon keyword e.g., human gut)] --> FP[Quality control<br/>(fastp)]
            FP --> ASM[Metagenome assembly<br/>(metaSPAdes)]
            ASM --> ORF[smORF discovery<br/>(pyrodigal; length-bounded ORFs)]
            ORF --> SP[Signal peptide inference<br/>(USPNet) → mature peptide FASTA]
            SP --> AMP[AMP triage<br/>(ESM2+LoRA classifier: amp_predict_fasta / amp_then_mic_from_fasta)]
            AMP --> Filt[Retain high-confidence AMPs<br/>(probability threshold ± top_k)]
            Filt --> MIC[MIC estimation<br/>(ESM2+LoRA regression: mic_predict_csv or amp_then_mic_from_fasta)]
            MIC --> Tox[Safety annotation<br/>(ToxinPred3.0 + HemoPI2 via augment_with_toxicity_and_hemolysis)]
            Tox --> Sim[Novelty check<br/>(ESM embedding similarity + sequence identity vs DBAASP)]
            Sim --> Rank[Prioritized export<br/>(ranked CSV/FASTA of AMP candidates)]
            Fetch[Reference AMPs<br/>(fetch_amps from DBAASP)] --> Sim
            Train[Custom model training<br/>(ML_Coder-authored AMP/MIC scripts)] --> AMP
            Train --> MIC
        ```
        """
    ).strip()

    return dedent(
        f"""# AMP Multi-Agent Workflow & Data Pipeline
Generated: {datetime.now().isoformat(timespec="seconds")}

Annotated Mermaid diagrams conveying the orchestration of the agent team and the analytical pipeline from raw metagenomic reads to prioritized AMP candidates. Render with any Mermaid-capable Markdown viewer or paste into https://mermaid.live .

## Agent orchestration (roles and responsibilities)
{agent_diagram}

## Experimental/analytical pipeline (end-to-end flow)
{data_pipeline}
"""
    )


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(build_content(), encoding="utf-8")
    print(f"Wrote Mermaid diagram to {OUT_FILE.resolve()}")
    print("Open it in a Mermaid viewer or paste a block into https://mermaid.live")


if __name__ == "__main__":
    main()
