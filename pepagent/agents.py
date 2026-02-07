#!/usr/bin/env python
# coding: utf-8

import autogen
try:
    from .llm_config import llm_config
    from . import agent_functions as func
except ImportError:
    from llm_config import llm_config
    import agent_functions as func
try:
    from .coding_agent import build_codex_coder_agent
except ImportError:
    try:
        from coding_agent import build_codex_coder_agent
    except Exception:
        build_codex_coder_agent = None
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

termination_msg = lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

config_list = autogen.config_list_from_models(
    #model_list=["gpt-4", "gpt-4-1106-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
    model_list=["gpt-5.2"]
    )

dir_path = './doc_dir/'
workspace_dir = './workspace/'

COMMON_FUNCTION_MAP = {
    "fetch_amps": func.fetch_amps,
    "save_to_csv_file": func.save_to_csv_file,
    "amp_then_mic_from_fasta": func.amp_then_mic_from_fasta,
    "augment_with_toxicity_and_hemolysis": func.augment_with_toxicity_and_hemolysis,
    "csv_metadata": func.csv_metadata,
    "python_repl_csv": func.python_repl_csv,
    "list_find_files_tool": func.list_find_files_tool,
    "download_top_fastq_and_build_smorf": func.download_top_fastq_and_build_smorf,
    "amp_esm_similarity_and_sequence_identity": func.amp_esm_similarity_and_sequence_identity,
    "amp_identity_similarity": func.amp_identity_similarity,
    "generate_candidates_genetic": func.generate_candidates_genetic,
    "generate_candidates_textgrad": func.generate_candidates_textgrad,
    "analyze_generation_stats": func.analyze_generation_stats,
}

CODE_FUNCTION_MAP = {
    "python_repl_csv": func.python_repl_csv,
    "list_find_files_tool": func.list_find_files_tool,
}

# autogen.ChatCompletion.start_logging()
user_proxy = autogen.UserProxyAgent(
    # user_proxy = autogen.AssistantAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="ALWAYS",
    system_message="user_proxy. Plan execution needs to be approved \
                    by user_proxy.",
    max_consecutive_auto_reply=None,
    code_execution_config=False,
    # code_execution_config={"work_dir": "coding"},
    # code_execution_config={"work_dir": "coding"},
    # code_execution_config={"work_dir": "coding",
    #                      "last_n_messages": 1,
    #                      },
)

ml_coder = autogen.AssistantAgent(
    name="ML_Coder",
    llm_config=llm_config,
    system_message=(
        "ML_Coder. You write COMPLETE, STANDALONE Python code for training and evaluating "
        "AMP and MIC prediction models.\n"
        "\n"
        "Requirements:\n"
        "1. Always output a SINGLE code block with a full script or module "
        "   (no '... modify above code' instructions).\n"
        "2. Your code should be self-contained given the existing project structure. "
        "   Reuse utilities from agent_functions.py where appropriate "
        "   (e.g., data loading conventions, sequence preprocessing, device selection).\n"
        "3. Prefer PyTorch + HuggingFace transformers for ESM models, and respect "
        "   resource constraints (support CPU only; make CUDA optional).\n"
        "4. Include:\n"
        "   - Dataset/DataLoader definitions tailored to AMP/MIC tables (CSV or FASTA).\n"
        "   - Model definition (ESM backbone + head), similar in spirit to the existing "
        "     EsmLoRASequenceClassifier and EsmLoRAMultiRegressionHead, but written from scratch.\n"
        "   - Training loop with argument parsing (batch size, learning rate, epochs, paths).\n"
        "   - Checkpoint saving/loading.\n"
        "5. Add clear comments where users should change paths or hyperparameters.\n"
        "6. Avoid installing packages; assume required libraries are available.\n"
        "7. Test boundary conditions (empty datasets, short sequences) via assertions.\n"
        "8. Please avoid asking the human to paste outputs or edit code; provide ready-to-run scripts."
    ),
)

codex_coder = build_codex_coder_agent(llm_config) if build_codex_coder_agent else None

critic = autogen.AssistantAgent(
    name="Critic",
    # instructions="Coder. You write Python code.",# This may involve identifying the chemical formula in SMILES codes, retrieving coordinates, and doing calculations.",# Reply `TERMINATE` in the end when everything is done.",

    system_message="""Critic. You double-check plan, especially the functions and function parameters. 
    Check whether the plan included all the necessary parameters for the suggested function. 
    You provide feedback.	
    You print TERMINATE when the task is finished sucessfully.
    """,
    # This may involve identifying the chemical formula in SMILES codes, retrieving coordinates, and doing calculations.",# Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)

# Old planner
planner = autogen.AssistantAgent(
    name="Planner",
    system_message="""Planner. You develop a plan. Begin by explaining the plan. Revise the plan based on feedback from the critic and user_proxy, until user_proxy approval. 
The plan may involve calling custom function for retrieving knowledge, designing proteins, and computing and analyzing protein properties. You include the function names in the plan and the necessary parameters.
If the plan involves retrieving knowledge, retain all the key points of the query asked by the user for the input message.

For code changes or command execution in the repo, assign tasks to Codex_Coder and instruct it to use codex_mcp_run.
Keep codex_mcp_run prompts concise and directive. Avoid embedding code blocks in the prompt; ask Codex to create/edit files.
""",
    # If the plan involves retrieving knowledge, retain all the key points of the query asked by the user for the input message.
    # This may involve identifying the chemical formula in SMILES codes, retrieving coordinates, and doing calculations.",# Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)

# New planner
code_planner = autogen.AssistantAgent(
    name="Planner",
    system_message="""Planner. You develop a plan. Begin by explaining the plan. Revise the plan based on feedback from the critic and user_proxy, until user_proxy approval. 
The plan may involve calling custom function for retrieving knowledge, computing and analyzing peptide properties, OR designing training/evaluation code for AMP/MIC models.

When the user asks to:
  - train a new AMP or MIC model,
  - fine-tune existing ESM-based models,
  - create or modify training/inference pipelines,

please:
  1) Assign those code-creation steps to the ML_Coder agent.
  2) Specify input/output formats (e.g., CSV columns, FASTA structure).
  3) Aim for compatibility with the existing tools in agent_functions.py
     (for example, produce models that can later be wrapped similarly to AMPPredictor/MICPredictor).

For general code changes or command execution in the repo, assign tasks to Codex_Coder and instruct it to use codex_mcp_run.
Prefer not to use codex_mcp_run to draft code; use ML_Coder for drafting and Codex_Coder for applying edits or running shell commands.

When the plan involves calling custom Python functions, include:
  - function names,
  - key parameters,
  - expected file inputs/outputs.

If the plan involves retrieving knowledge, retain all key points of the user query.""",
    llm_config=llm_config,
)

# 2. create the RetrieveUserProxyAgent instance named "ragproxyagent"
# By default, the human_input_mode is "ALWAYS", which means the agent will ask for human input at every step. We set it to "NEVER" here.
# `docs_path` is the path to the docs directory. By default, it is set to "./docs". Here we generated the documentations from FLAML's docstrings.
# Navigate to the website folder and run `pydoc-markdown` and it will generate folder `reference` under `website/docs`.
# `task` indicates the kind of task we're working on. In this example, it's a `code` task.
# `chunk_token_size` is the chunk token size for the retrieve chat. By default, it is set to `max_tokens * 0.6`, here we set it to 2000.

# Create a new collection for NaturalQuestions dataset
# `task` indicates the kind of task we're working on. In this example, it's a `qa` task.


# ragproxyagent = RetrieveUserProxyAgent(
#     name="ragproxyagent",
#     system_message="Assistant who has extra content retrieval power for biomaterials domain knowledge. The assistant follows the plan.",
#     human_input_mode="NEVER",
#     is_termination_msg=termination_msg,
#     max_consecutive_auto_reply=10,
#     retrieve_config={
#         "task": "qa",
#         "docs_path": f"{dir_path}",  # f"{doc_dir}",
#         "chunk_token_size": 3000,
#         "model": config_list[0]["model"],
#         # "client": chromadb.PersistentClient(path=coll_path),
#         # "collection_name": coll_name,
#         "chunk_mode": "one_line",
#         # "chunk_mode": "multi_lines", # "one_line",
#         "embedding_model": "all-MiniLM-L6-v2",
#         "get_or_create": "True"
#     },
#     llm_config=llm_config,
# )

# 1. Define the SAGA Planner (Replaces or extends current Planner)
strategic_planner = autogen.AssistantAgent(
    name="Strategic_Planner",
    llm_config=llm_config,
    system_message="""You are the Strategic Planner (System 2) of a scientific discovery agent.
    
    Your Goal: Design antimicrobial peptides (AMPs) that are potent, safe (non-toxic), and novel.
    
    Process:
    1. Define current Objectives (e.g., "Maximize AMP probability, Minimize MIC").
    2. Instruct the Optimizer to run a generation/screening cycle.
    3. Read the 'Analyzer' report and any safety/metadata summaries.
    4. EVOLVE OBJECTIVES using feedback:
       - If toxicity is high, consider adding a safety constraint or tighter safety screening.
       - If diversity is low, consider nudging toward novelty (similarity penalties or broader exploration).
       - If successful, consider strengthening objectives (e.g., lower hemolysis).
    5. When proposing new candidate generation, prefer generate_candidates_textgrad.
       Only suggest generate_candidates_genetic if TextGrad fails or is unavailable.
    6. If the user did not specify a screening threshold, start with min_amp_prob=0.5.
       If a screening run returns zero candidates, report that no candidates meet the
      threshold and stop the workflow (avoid auto-relax).

    Optimization workflows (TextGrad or genetic):
    - Begin by calling generate_candidates_textgrad on the provided input CSV to create
      a generation FASTA (e.g., generation0.fasta) before any screening.
    - Then run amp_then_mic_from_fasta on that generation FASTA, followed by
      augment_with_toxicity_and_hemolysis, then amp_esm_similarity_and_sequence_identity
      against data/Verified_AFPs_Database_cleaned.fasta, and analyze_generation_stats.
    - By default, run a single generation unless the user explicitly requests multi-generation.
    - If the user provides top_k/optimize_top (or input size is large), prefer those
      values for candidate selection instead of very small defaults.
    - Rank the generation with python_repl_csv (composite score, include novelty if available)
      and use that for final output. Keep the scoring recipe simple and document the
      columns used, but avoid over-prescriptive weights in instructions.

    Screening/evaluation workflows in peptide discovery (AMP/AFP/etc.):
    - For screening-only tasks (no optimization), this pipeline usually works well:
        (1) amp_then_mic_from_fasta
        (2) augment_with_toxicity_and_hemolysis (safety screen)
        (3) amp_esm_similarity_and_sequence_identity vs data/Verified_AFPs_Database_cleaned.fasta
        (4) python_repl_csv composite ranking (potency vs MIC vs safety + novelty) + ranked CSV
      If a step is missing and it affects the objective, ask the Optimizer to continue before finalizing.
    - For novelty scoring, use data/Verified_AFPs_Database_cleaned.fasta by default.
      Do not invent alternate verified DB file paths unless the user explicitly provides one.
    - If the user requests a final top_k, avoid passing top_k into amp_then_mic_from_fasta so
      the full AMP-prob≥threshold pool can be safety/novelty-ranked before final downselection.
    - Avoid redundant expensive reruns: once amp_then_mic_from_fasta has produced AMP+MIC
      results for a sequence pool, use python_repl_csv on that table for filtering/ranking
      instead of rerunning amp_then_mic_from_fasta on filtered subsets.
    - If the safety tool is unavailable or fails, note it explicitly and still aim for
      a composite ranking using potency + MIC only.
    - If similarity scoring fails, note it explicitly and proceed with the standard
      composite score (potency + MIC + safety).
    - If safety is a key goal and the input is large, consider avoiding early
      downselection by AMP-only scores; keep a broader pool (e.g., 3–5x the final top_k
      or no top_k if feasible) before safety/novelty-aware ranking, then cut to the
      final top_k at the end. If the tool summary warns about early top_k truncation,
      consider rerunning with a larger pre-top_k before final ranking.
    - Treat safety as a soft objective. Do not force all final candidates to be
      Non-Toxin/Non-Hemolytic unless the user explicitly asks for strict filtering.
    - After ranking, review safety rates. If unsafe rates are clearly high, request one
      rerank with stronger continuous safety weighting (no hard filters, no binary gating).
      If rates are already moderate, keep a balanced composite and avoid tightening toward
      all-safe outputs.
    - Ask for summary stats via analyze_generation_stats or csv_metadata.
    """
)

# 2. Define the Optimizer Agent (Executes the loop)
optimizer_agent = autogen.AssistantAgent(
    name="Optimizer",
    llm_config=llm_config,
    system_message="""You are the Optimizer (System 1).
    Your task is to EXECUTE the search loop based on the Planner's objectives.
    
    Standard Loop:
    1. Call `generate_candidates_textgrad` to refine top candidates with TextGrad.
       If TextGrad is unavailable or fails, fall back to `generate_candidates_genetic`.
       If the Planner suggests `generate_candidates_genetic`, still attempt TextGrad first.
    2. Call `amp_then_mic_from_fasta` to screen them for potency.
       If amp_then_mic_from_fasta fails due to a missing FASTA, immediately call
       generate_candidates_textgrad using the input CSV from the user message to
       create a generation FASTA (e.g., generation0.fasta) and retry.
    3. Call `augment_with_toxicity_and_hemolysis` to screen for safety.
    4. Call `analyze_generation_stats` to generate the report for the Planner.
    5. If the user did not specify a screening threshold, call amp_then_mic_from_fasta
       with min_amp_prob=0.5. If a screen returns zero candidates, report that no
       candidates meet the threshold and stop (avoid auto-relax).
    
    Screening-only tasks:
    - This sequence is a solid default before any final summary:
      amp_then_mic_from_fasta -> augment_with_toxicity_and_hemolysis
      -> amp_esm_similarity_and_sequence_identity (verified AFP DB at data/Verified_AFPs_Database_cleaned.fasta)
      -> python_repl_csv.
    - If the user wants a final top_k, do not pass top_k into amp_then_mic_from_fasta;
      screen the full AMP-prob≥threshold pool, then apply composite ranking to select top_k.
    - Avoid repeated expensive AMP/MIC passes on the same pool. If a current CSV already
      has AMP probability/confidence and MIC columns, perform filtering/downselection/reranking
      in python_repl_csv instead of calling amp_then_mic_from_fasta again on a subset.
    - If augment_with_toxicity_and_hemolysis fails or safety columns are missing, clearly
      report it and still run python_repl_csv with potency + MIC only.
    - If similarity scoring fails or similarity columns are missing, report it and
      proceed with the standard composite (potency + MIC + safety).
    - If the task emphasizes safety and the input pool is large, consider avoiding
      early downselection by AMP-only scores; keep a broader pool before applying
      composite ranking to select the final top_k. It is fine to run amp_then_mic_from_fasta
      with a larger pre-top_k (or no top_k) and then downselect in python_repl_csv.
      When selecting the final top_k, avoid hard filters on safety unless requested;
      instead, include safety scores/flags in the composite and report safety rates.
    - It is usually helpful to run analyze_generation_stats after safety scoring.
      If toxicity is high, adjust the composite ranking to emphasize safety signals
      (toxicity_score, hemolysis_score, and safety prediction flags) without hard-filtering.
      Prefer smooth weighting so unsafe rates stay low, but avoid binary penalties
      or repeated over-tightening that force all candidates to be Non-Toxin/Non-Hemolytic.
      A practical pattern is a base composite (potency/novelty/amp) softly damped by
      normalized safety (e.g., multiply by 0.4 + 0.6*safety_norm) rather than
      hard label penalties.
    - If amp_then_mic_from_fasta returns a note about early top_k before safety
      and safety quality matters, it is often worth rerunning with a larger
      pre-top_k (e.g., 3–5x) and then redoing safety + ranking.
    - Prefer python_repl_csv to produce a ranked CSV (composite score), then read that CSV
      for top-5 reporting. Use a reasonable normalized composite of:
        * Potency (amp_probability, amp_confidence_score)
        * MIC (prefer 'Candida albicans (merged) MIC' else mean of '* MIC' columns)
        * Safety (toxicity_score, hemolysis_score)
        * Novelty if available (esm_knn_mean_similarity, max_seq_identity; lower is better)
      Keep safety as one component in a balanced composite. Do not impose strict
      binary gating by toxicity/hemolysis labels unless requested by the user.
      Keep weighting simple and explain the columns used in the output.
      Save ranked CSV as <input_stem>_ranked.csv and include rank + composite_score.
      Use only columns ending with ' MIC' as MIC features; do not mix derived columns
      (e.g., safety_score/safety_penalty/novelty fields) into MIC aggregation.
      Do not use explicit binary penalties based on safety labels by default
      (e.g., avoid np.where(label=='Toxin', ...)). Use continuous safety scores.
      For screening top_k outputs, run a soft rate-feedback rerank in python_repl_csv:
        - Rank once with a balanced composite using potency/novelty/amp/safety.
        - Measure toxin and hemolysis rates in top_k using prediction labels.
        - If toxin_rate > 0.60 or hemolysis_rate > 0.40, increase safety weight and rerank.
        - Repeat up to 1-2 times, still without hard filtering.
        - If rates are already moderate, keep the balanced composite and stop reranking.
      Aim for low unsafe rates, but do not force all-non-toxic/all-non-hemolytic.
      When using prediction labels, match exact values (e.g., 'Toxin' vs 'Non-Toxin')
      instead of substring contains, because 'Non-Toxin' contains 'toxin'.
      NOTE: python_repl_csv executes code with empty globals, so avoid defining helper
      functions that reference pd. Use inline min-max expressions instead.

    Optimization tasks:
    - Default to a single generation:
      generate_candidates_textgrad -> amp_then_mic_from_fasta -> augment_with_toxicity_and_hemolysis
      -> amp_esm_similarity_and_sequence_identity -> python_repl_csv (ranked CSV with composite_score).
    - Usually stop after the first ranked CSV is produced unless the user explicitly asked for
      additional generations.
    - If user-provided top_k/optimize_top are available, prefer using them when generating candidates.
    - Use the ranked CSV for top-5 reporting.
    - Use csv_metadata or analyze_generation_stats to summarize.
    
    Output the final CSV path and the Analysis Report so the Planner can read it.
    """,
    function_map=COMMON_FUNCTION_MAP,
)
assistant = autogen.AssistantAgent(
    name="assistant",
    # instructions="You collect information from experts, fold proteins, and carry out other simulations. Reply TERMINATE when the task is done.",

    system_message=(
        "assistant. You have access to all the custom functions. You focus on executing the "
        "functions suggested by the planner or the critic and avoid introducing new plans. "
        "If a screening workflow reaches the end without novelty scoring, consider calling "
        "amp_esm_similarity_and_sequence_identity on the current screening CSV using "
        "data/Verified_AFPs_Database_cleaned.fasta (work_dir=workspace) before ranking. "
        "Use that verified AFP database path by default; do not guess alternate file names. "
        "If a screening workflow reaches the end without a composite ranking, consider running "
        "python_repl_csv to produce a ranked CSV before any final summary. "
    "Use a simple normalized composite across potency, MIC, safety, and novelty if available; "
        "avoid overly specific weights in instructions. Treat safety as a soft objective and "
        "avoid binary label penalties unless explicitly requested by the user. "
    "If unsafe rates are high after initial ranking, rerank once with stronger continuous "
        "safety weighting (no hard filters). For top_k outputs, use an iterative soft "
        "rate-feedback rerank only when rates are clearly high, and avoid repeatedly "
        "tightening toward all-safe outputs. "
    "Use only columns ending with ' MIC' as MIC features; do not mix derived score columns "
        "into MIC aggregation. "
    "When selecting top_k, prefer composite ranking over hard safety filters unless the user "
        "explicitly requests strict filtering. If the user wants a final top_k, avoid passing "
        "top_k into amp_then_mic_from_fasta; downselect only after safety/novelty-aware ranking. "
        "If a CSV already has AMP+MIC columns, do not rerun amp_then_mic_from_fasta on filtered "
        "subsets; use python_repl_csv for downstream filtering and ranking. "
        "If using toxicity/hemolysis prediction labels, compare exact strings (e.g., == 'Toxin') "
        "rather than substring matching. "
        "Save ranked CSV as <input_stem>_ranked.csv (include rank + composite_score) and use it "
        "for top-5 reporting. IMPORTANT: python_repl_csv executes with empty globals, so do not "
        "define helper functions that reference pd; use inline min-max expressions."
    ),
    llm_config=llm_config,
    function_map=COMMON_FUNCTION_MAP,
)

code_assistant = autogen.AssistantAgent(
    name="assistant",
    # instructions="You collect information from experts, fold proteins, and carry out other simulations. Reply TERMINATE when the task is done.",

    system_message=("assistant. You have access to all the custom functions. You focus on executing the functions suggested by the planner or the critic. You also have the ability to prepare the required input parameters for the functions."),
    llm_config=llm_config,
    function_map=CODE_FUNCTION_MAP,
)
