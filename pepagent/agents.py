#!/usr/bin/env python
# coding: utf-8

import autogen
try:
    from .llm_config import llm_config
    from . import agent_functions as func
except ImportError:
    from llm_config import llm_config
    import agent_functions as func
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

termination_msg = lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

try:
    config_list = autogen.config_list_from_models(
        model_list=["gpt-4", "gpt-4-1106-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
    )
except Exception:
    config_list = []

dir_path = './doc_dir/'
workspace_dir = './workspace/'

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
'''
coder=GPTAssistantAgent(
#coder = autogen.AssistantAgent(
    name="Coder",
    instructions="Coder. You write Python code.",# This may involve identifying the chemical formula in SMILES codes, retrieving coordinates, and doing calculations.",# Reply `TERMINATE` in the end when everything is done.",

    #system_message="Coder. You write Python code.",# This may involve identifying the chemical formula in SMILES codes, retrieving coordinates, and doing calculations.",# Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)
'''

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
        "2. Your code must be self-contained given the existing project structure. "
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
        "6. Do NOT install packages; assume required libraries are available.\n"
        "7. Test boundary conditions (empty datasets, short sequences) via assertions.\n"
        "8. Never ask the human to paste outputs or edit code; provide ready-to-run scripts."
    ),
)

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

executor = autogen.UserProxyAgent(
    name="Executor",
    system_message="Executor. You follow the plan. Execute the code written by the coder and return outcomes.",
    human_input_mode="NEVER",
    code_execution_config={"last_n_messages": 12, "work_dir": workspace_dir},
    llm_config=llm_config,
)

# Old planner
planner = autogen.AssistantAgent(
    name="Planner",
    system_message="""Planner. You develop a plan. Begin by explaining the plan. Revise the plan based on feedback from the critic and user_proxy, until user_proxy approval. 
The plan may involve calling custom function for retrieving knowledge, designing proteins, and computing and analyzing protein properties. You include the function names in the plan and the necessary parameters.
If the plan involves retrieving knowledge, retain all the key points of the query asked by the user for the input message.
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

you MUST:
  1) Explicitly assign those code-creation steps to the ML_Coder agent.
  2) Specify input/output formats (e.g., CSV columns, FASTA structure).
  3) Ensure compatibility with the existing tools in agent_functions.py
     (for example, produce models that can later be wrapped similarly to AMPPredictor/MICPredictor).

When the plan involves calling custom Python functions, include:
  - function names,
  - required parameters,
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


'''
reviewer=GPTAssistantAgent(
#reviewer = autogen.AssistantAgent(
    name="Scientific_Reviewer",
   # is_termination_msg=termination_msg,
    #system_message="You are a scientific reviewer who offers additional background that will be incorporated into the answer. ",
    instructions="Materials scientist. You follow the plan. As a materials scientist you offer additional background that will be incorporated into the answer.",
    llm_config=llm_config,
)
'''
# reviewer=GPTAssistantAgent(
reviewer = autogen.AssistantAgent(
    name="Scientific_Reviewer",
    # is_termination_msg=termination_msg,
    system_message="Materials scientist. You follow the plan. As a materials scientist you offer additional background that will be incorporated into the answer.",
    # instructions="Materials scientist. You follow the plan. As a materials scientist you offer additional background that will be incorporated into the answer.",
    llm_config=llm_config,
)

assistant = autogen.AssistantAgent(
    name="assistant",
    # instructions="You collect information from experts, fold proteins, and carry out other simulations. Reply TERMINATE when the task is done.",

    system_message=("assistant. You have access to all the custom functions. You focus on executing the functions suggested by the planner or the critic. You also have the ability to prepare the required input parameters for the functions."),
    llm_config=llm_config,
function_map = {
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
},
)

code_assistant = autogen.AssistantAgent(
    name="assistant",
    # instructions="You collect information from experts, fold proteins, and carry out other simulations. Reply TERMINATE when the task is done.",

    system_message=("assistant. You have access to all the custom functions. You focus on executing the functions suggested by the planner or the critic. You also have the ability to prepare the required input parameters for the functions."),
    llm_config=llm_config,
function_map = {
    "python_repl_csv": func.python_repl_csv,
    "list_find_files_tool": func.list_find_files_tool,
},
)
'''
sequence_retriever =  GPTAssistantAgent(

    name="sequence_retriever",
    #instructions="You collect information from experts, fold proteins, and carry out other simulations. Reply TERMINATE when the task is done.",

    instructions="Sequence retriever. You identify amino acid sequences based on the name of the protein.",
    llm_config=llm_config ,

    #code_execution_config={"work_dir": "coding"},
)
'''
