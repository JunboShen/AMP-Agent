#!/usr/bin/env python
# coding: utf-8
from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path

workspace_dir = './workspace/'
device = 'cpu'

import os

try:
    os.mkdir(workspace_dir)
except:
    pass

import pandas as pd
try:
    from . import utils
except ImportError:
    import utils

REPO_ROOT = Path(__file__).resolve().parents[1]
_PARENT_ROOT = REPO_ROOT.parent

def _pick_existing_path(candidates: list[Path]) -> str:
    for p in candidates:
        if p.exists():
            return str(p)
    # fallback to first candidate for clearer error messages downstream
    return str(candidates[0])

DEFAULT_AMP_MODEL_DIR = _pick_existing_path([
    REPO_ROOT / "AMP_fungus" / "checkpoint" / "ESM2_650M_adapted_lora_full_dataset",
    _PARENT_ROOT / "AMP_fungus" / "checkpoint" / "ESM2_650M_adapted_lora_full_dataset",
])
DEFAULT_MIC_MODEL_DIR = _pick_existing_path([
    REPO_ROOT / "AMP_fungus" / "checkpoint" / "ESM2_650M_adapted_lora_MIC",
    _PARENT_ROOT / "AMP_fungus" / "checkpoint" / "ESM2_650M_adapted_lora_MIC",
])
DEFAULT_SKIN_FASTA = _pick_existing_path([
    REPO_ROOT / "AMP_fungus" / "data" / "skin" / "skin.fasta",
    _PARENT_ROOT / "AMP_fungus" / "data" / "skin" / "skin.fasta",
])

from Bio.PDB import PDBParser, DSSP
from collections import Counter
import json

import openai

try:
    import autogen
    from autogen import AssistantAgent
    from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
    from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
except ModuleNotFoundError:
    autogen = None
    AssistantAgent = None
    RetrieveAssistantAgent = None
    RetrieveUserProxyAgent = None
import re

import requests, sys

import subprocess

try:
    from llama_index.core import StorageContext, load_index_from_storage
except ModuleNotFoundError:
    StorageContext = None
    load_index_from_storage = None

##########################################################################################################
# storage_context = StorageContext.from_defaults(persist_dir="protein_index")
# new_index = load_index_from_storage(storage_context)
# query_engine = new_index.as_query_engine(similarity_top_k=20)

# Path to Autoregressive transformer model, ForceGPT
model_path = '###'


def retrieve_content(message, n_results=3):
    ragproxyagent.n_results = n_results  # Set the number of results to be retrieved.
    # Check if we need to update the context.
    update_context_case1, update_context_case2 = ragproxyagent._check_update_context(message)
    if (update_context_case1 or update_context_case2) and ragproxyagent.update_context:
        ragproxyagent.problem = message if not hasattr(ragproxyagent, "problem") else ragproxyagent.problem
        _, ret_msg = ragproxyagent._generate_retrieve_user_reply(message)
    else:
        ret_msg = ragproxyagent.generate_init_message(message, n_results=n_results)
    return ret_msg if ret_msg else message


# def retrieve_content_LlamaIndex(message, ):
#     # message='For these topics, provide detailed information: ' + message
#     print(f'the message is: {message}')
#     response = query_engine.query(message)
#     response = response.response
#     # agent
#     # response = agent.chat("Describe several biologically inspired composite ideas.")
#     # print(str(response))
#
#     return response if response else message

def add_missing_column(file_path):
    # Read all lines from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    header = False
    for line in lines:
        LINE = str(line).split(sep=' ')
        for item in LINE:
            if re.search('HEADER', item):
                header = True
    # Process lines
    modified_lines = []
    if not header:
        for line in lines:
            if line.startswith('ATOM'):
                columns = line.split()
                # Assuming the missing column is the atom type, which should be the 12th column
                if len(columns) < 13:
                    atom_type = columns[2][0]  # Extract atom type (3rd column in ATOM line)
                    # Add the atom type to the end of the line
                    modified_line = line.strip() + '    ' + atom_type + '\n'
                    modified_lines.append(modified_line)
                else:
                    modified_lines.append(line)
            else:
                modified_lines.append(line)

        # Write the modified lines back to the file
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)


# https://www.ebi.ac.uk/proteins/api/doc/#!/proteins/search
def get_FASTA_from_name(protein_name):
    size = 128
    requestURL = f"https://www.ebi.ac.uk/proteins/api/proteins?offset=0&size={size}&protein={protein_name}"
    # requestURL = f"https://www.ebi.ac.uk/proteins/api/proteins?offset=0&size={size}&protein={name}"

    r = requests.get(requestURL, headers={"Accept": "application/json"})

    if not r.ok:
        r.raise_for_status()
        sys.exit()

    responseBody = r.text
    # print(responseBody)
    json_object = json.loads(responseBody)
    if len(json_object) > 0:

        res = json_object[0]['sequence']['sequence']
    else:
        res = 'No results found.'

    return res

def save_to_csv_file(
    input_JSON_dictionary,
    output_csv_name,
    work_dir: str = workspace_dir
) -> str:
    """
    Creates and stores a CSV file from the provided JSON string (list/dict),
    and RETURNS a JSON STRING summary so Autogen can relay a proper message.

    Returns (str): JSON string with keys:
        - status: "ok" | "error"
        - output_file: absolute path to the CSV (if ok)
        - num_rows: number of rows written (if ok)
        - columns: list of column names (if ok)
        - error: error message (if error)
    """
    try:
        # Parse JSON payload into a DataFrame
        data_dictionary = json.loads(input_JSON_dictionary)
        df = pd.DataFrame(data_dictionary)

        # Ensure output directory exists; resolve relative path under work_dir
        os.makedirs(work_dir, exist_ok=True)
        out_path = output_csv_name
        if not os.path.isabs(out_path):
            out_path = os.path.join(work_dir, output_csv_name)

        # Write CSV
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        # Provide a concise, machine-readable summary back to the agent
        summary = {
            "status": "ok",
            "output_file": os.path.abspath(out_path),
            "num_rows": int(len(df)),
            "columns": list(df.columns),
            "note": "CSV written successfully."
        }

        # Optional: still print a human log line for local debugging
        print(f"The results have been saved to CSV: {out_path}")

        # IMPORTANT: Autogen expects a STRING return — not None, not a dict.
        return json.dumps(summary)
    except Exception as e:
        # Never raise; always return a JSON string so the chat has content
        return json.dumps({
            "status": "error",
            "error": str(e)
        })


def fetch_amps(
    name=None,
    uniprot=None,
    target_group=None,
    target_species=None,
    sequence=None,
    sequence_length_range=None,  # tuple (min_len, max_len)
    synthesis_type=None,         # "Ribosomal", "Nonribosomal", "Synthetic"
    complexity=None,             # "monomer", "multimer", "multi_peptide"
    n_terminus=None,
    c_terminus=None,
    unusual_aa=None,
    kingdom=None,
    source=None,
    has_structure=None,          # "without_structure" | "pdb" | "md_model" | "pdb_md_model"
    limit=100,
    offset=0,
    work_dir: str = "workspace",
    output_csv: str = "fetched_amps.csv",
    # --- validation controls ---
    min_len: int = 5,
    allow_UO: bool = True,       # allow selenocysteine (U) and pyrrolysine (O)
    allow_ambiguous: bool = False,  # allow B/Z/J (ambiguous) if True
):
    """
    Fetch AMP data from the DBAASP API using various filters, omit invalid peptides,
    and save to CSV at {work_dir}/{output_csv}.

    Invalid peptide criteria (by default):
      - sequence length < min_len (default 5)
      - contains characters outside the allowed amino-acid set:
        standard 20 AAs (ACDEFGHIKLMNPQRSTVWY) plus optional U/O (if allow_UO),
        and optional ambiguous B/Z/J (if allow_ambiguous). 'X' is *not* allowed.
      - any lowercase letters, digits, spaces, or punctuation.

    Returns:
        str: JSON string of VALID AMP records.
    """

    def _is_valid_sequence(seq: str) -> bool:
        if not isinstance(seq, str):
            return False
        s = seq.strip()
        if len(s) < min_len:
            return False
        # Build allowed set
        allowed = set("ACDEFGHIKLMNPQRSTVWY")
        if allow_UO:
            allowed |= {"U", "O"}
        if allow_ambiguous:
            allowed |= {"B", "Z", "J"}
        # All characters must be uppercase and within allowed set
        return all(ch in allowed for ch in s)  # rejects lowercase, digits, symbols, 'X', etc.

    url = "https://dbaasp.org/peptides"
    params = {"limit": limit, "offset": offset}

    # Basic text filters
    if name:
        params["name.value"] = name
    if uniprot:
        params["uniprot.value"] = uniprot
    if sequence:
        params["sequence.value"] = sequence

    # Target-related filters
    if target_group:
        params["targetGroup.value"] = target_group
    if target_species:
        params["targetSpecies.value"] = target_species

    # Sequence length range (e.g., (8, 50) -> "8-50")
    if sequence_length_range and len(sequence_length_range) == 2:
        params["sequence.length"] = f"{sequence_length_range[0]}-{sequence_length_range[1]}"

    # Synthesis type and complexity
    if synthesis_type:
        params["synthesisType.value"] = synthesis_type
    if complexity:
        params["complexity.value"] = complexity

    # Termini modifications
    if n_terminus:
        params["nTerminus.value"] = n_terminus
    if c_terminus:
        params["cTerminus.value"] = c_terminus

    # Unusual amino acids
    if unusual_aa:
        params["unusualAminoAcid.value"] = unusual_aa

    # Source information
    if kingdom:
        params["kingdom.value"] = kingdom
    if source:
        params["source.value"] = source

    # 3D structure availability
    if has_structure:
        params["threeDStructure.value"] = has_structure

    # Request
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json().get("data", [])

    # Remove entries with missing/empty sequences and those failing validity checks
    clean_data = [p for p in data if p.get("sequence")]
    valid_data = [p for p in clean_data if _is_valid_sequence(p["sequence"])]

    # ---- Save to CSV ----
    os.makedirs(work_dir, exist_ok=True)
    out_path = os.path.join(work_dir, output_csv)

    # Flatten nested fields to columns like "field.subfield"
    df = pd.json_normalize(valid_data, sep=".")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    return json.dumps(valid_data, ensure_ascii=False)



import os
import json
from typing import List, Dict
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, EsmConfig, EsmModel
from transformers.modeling_outputs import ModelOutput
from peft import PeftModel
from tqdm import tqdm

# Reuse heavy models within the same process to avoid repeated load times.
_AMP_PREDICTOR_CACHE: dict[tuple[str, str], "AMPPredictor"] = {}
_MIC_PREDICTOR_CACHE: dict[tuple[str, str], "MICPredictor"] = {}

def _get_amp_predictor(model_dir: str, device: str):
    key = (os.path.abspath(model_dir), device)
    if key not in _AMP_PREDICTOR_CACHE:
        _AMP_PREDICTOR_CACHE[key] = AMPPredictor(model_dir=model_dir, device=device)
    return _AMP_PREDICTOR_CACHE[key]

def _get_mic_predictor(model_dir: str, device: str):
    key = (os.path.abspath(model_dir), device)
    if key not in _MIC_PREDICTOR_CACHE:
        _MIC_PREDICTOR_CACHE[key] = MICPredictor(model_dir=model_dir, device=device)
    return _MIC_PREDICTOR_CACHE[key]


# -----------------------
#   Model definitions
# -----------------------
class EsmLoRAMultiRegressionHead(nn.Module):
    def __init__(self, esm_model, config, num_targets: int):
        super().__init__()
        self.esm = esm_model
        self.config = config
        hidden_size = config.hidden_size
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_targets),
        )
        for m in self.reg_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state

        # mean pool over non-padding tokens, excluding CLS (0) and EOS (last non-pad)
        modified_attention_mask = attention_mask.clone()
        modified_attention_mask[:, 0] = 0  # drop CLS
        seq_lens = attention_mask.sum(dim=1)
        for i, L in enumerate(seq_lens):
            if L > 1:
                modified_attention_mask[i, L - 1] = 0  # drop EOS

        expanded_mask = modified_attention_mask.unsqueeze(-1).expand_as(last_hidden_states).float()
        masked_hidden = last_hidden_states * expanded_mask
        sum_hidden = masked_hidden.sum(dim=1)
        denom = expanded_mask.sum(dim=1).clamp(min=1e-9)
        pooled = sum_hidden / denom

        logits = self.reg_head(pooled)
        loss = None
        if labels is not None:
            mask = ~torch.isnan(labels)
            loss = nn.functional.mse_loss(logits[mask], labels[mask]) if mask.any() else torch.tensor(
                0.0, device=logits.device, requires_grad=True
            )
        return ModelOutput(loss=loss, logits=logits)


class MICPredictor:
    """MIC predictor (ESM-2 backbone + LoRA + regression head)"""

    def __init__(self, model_dir: str, device: str = "auto"):
        self.model_dir = model_dir
        # device selection (CUDA -> MPS -> CPU)
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        self.model_info = self._load_model_info()
        self.max_length = self.model_info.get("max_length", 100)
        self.species: List[str] = self.model_info["species"]
        self.num_targets = len(self.species)
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        print("MIC Predictor initialized successfully!")

    def _load_model_info(self) -> Dict:
        info_path = os.path.join(self.model_dir, "model_info.json")
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Model info file not found: {info_path}")
        with open(info_path, "r") as f:
            return json.load(f)

    def _load_tokenizer(self):
        tok_path = os.path.join(self.model_dir, "tokenizer")
        if os.path.exists(tok_path):
            print(f"Loading tokenizer from: {tok_path}")
            return AutoTokenizer.from_pretrained(tok_path)
        base_model = self.model_info["base_model_type"]
        print(f"Loading tokenizer from base model: {base_model}")
        return AutoTokenizer.from_pretrained(base_model)

    def _load_model(self):
        # Load ESM WITHOUT meta init to avoid meta-tensor issues
        base_model = self.model_info.get("base_model_type", "facebook/esm2_t33_650M_UR50D")
        custom_model_path = self.model_info.get("custom_model_path")
        try:
            if custom_model_path and os.path.exists(custom_model_path):
                print(f"Loading ESM model from: {custom_model_path}")
                config = EsmConfig.from_pretrained(custom_model_path)
                esm_model = EsmModel.from_pretrained(custom_model_path, torch_dtype=torch.float32, device_map=None)
                model_source = "custom"
            else:
                raise FileNotFoundError("Custom model not found")
        except Exception:
            print(f"Loading ESM model from: {base_model}")
            config = EsmConfig.from_pretrained(base_model)
            esm_model = EsmModel.from_pretrained(base_model, torch_dtype=torch.float32, device_map=None)
            model_source = "base"

        # Attach LoRA adapter (materialized, no sharding)
        lora_adapter_path = os.path.join(self.model_dir, "lora_adapter")
        if not os.path.exists(lora_adapter_path):
            raise FileNotFoundError(f"LoRA adapter not found: {lora_adapter_path}")
        print(f"Loading LoRA adapter from: {lora_adapter_path}")
        esm_model = PeftModel.from_pretrained(
            esm_model,
            lora_adapter_path,
            device_map=None,
            torch_dtype=torch.float32,
            is_trainable=False,
        )

        # Build regression head and load its weights
        model = EsmLoRAMultiRegressionHead(esm_model, config, num_targets=self.num_targets)
        reg_head_weights = os.path.join(self.model_dir, "regression_head.pth")
        if not os.path.exists(reg_head_weights):
            raise FileNotFoundError(f"Regression head weights not found: {reg_head_weights}")
        print(f"Loading regression head weights from: {reg_head_weights}")
        state_dict = torch.load(reg_head_weights, map_location="cpu")
        model.reg_head.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()
        print(f"Model loaded successfully with {model_source} ESM backbone + LoRA + regression head")
        return model

    @torch.no_grad()
    def predict_sequences(
        self,
        sequences: List[str],
        seq_ids: List[str] | None = None,
        amp_probs: List[float] | None = None,
        batch_size: int = 64,
    ) -> pd.DataFrame:
        if seq_ids is None:
            seq_ids = [f"seq_{i}" for i in range(len(sequences))]
        if len(sequences) != len(seq_ids):
            raise ValueError("Number of sequences and sequence IDs must match")
        if amp_probs is None:
            amp_probs = [np.nan] * len(sequences)
        if len(amp_probs) != len(sequences):
            raise ValueError("Number of amp_probability values must match the number of sequences")

        rows = []
        print(f"Predicting {len(sequences)} sequences in batches of {batch_size}...")
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_sequences = sequences[i:i + batch_size]
            batch_ids = seq_ids[i:i + batch_size]
            batch_probs = amp_probs[i:i + batch_size]

            tokenized = self.tokenizer(
                batch_sequences,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.detach().cpu().numpy()

            for sid, seq, prob, pred in zip(batch_ids, batch_sequences, batch_probs, preds):
                row = {"seq_id": sid, "sequence": seq, "amp_probability": prob}
                row.update({sp: float(v) for sp, v in zip(self.species, pred)})
                rows.append(row)

        return pd.DataFrame(rows)

    def predict_csv(
        self,
        input_csv: str,
        output_prefix: str,
        seq_col: str = "sequence",
        id_col: str = "seq_id",
        amp_prob_col: str = "amp_probability",
        batch_size: int = 256,
        chunk_size: int = 100_000,
        skip_chunks: int = 0,
    ) -> List[str]:
        """
        Predict MIC for sequences in a CSV file (chunked). Returns list of output CSV paths.
        """
        print(f"Reading input: {input_csv}")
        outputs: List[str] = []

        reader = pd.read_csv(input_csv, chunksize=chunk_size)
        chunk_idx = 1
        total = 0

        # Skip initial chunks if requested
        for _ in range(skip_chunks):
            print(f"Skipping chunk {chunk_idx}")
            next(reader)
            chunk_idx += 1
            total += chunk_size

        for chunk in reader:
            sequences = chunk[seq_col].astype(str).tolist()
            if id_col in chunk.columns:
                seq_ids = chunk[id_col].astype(str).tolist()
            else:
                seq_ids = [f"seq_{i + total}" for i in range(len(sequences))]

            amp_probs = chunk[amp_prob_col].tolist() if amp_prob_col in chunk.columns else [np.nan] * len(sequences)

            print(f"Predicting chunk {chunk_idx}, size: {len(sequences)}")
            results_df = self.predict_sequences(sequences, seq_ids, amp_probs, batch_size=batch_size)

            outname = f"{output_prefix}_{chunk_idx:05d}.csv"
            results_df.to_csv(outname, index=False)
            print(f"Chunk {chunk_idx} saved to: {outname}")
            outputs.append(os.path.abspath(outname))

            total += len(sequences)
            chunk_idx += 1

        print(f"Finished. Total sequences processed: {total}")
        return outputs


# -----------------------
#   Agent-callable tool
# -----------------------
def mic_predict_csv(
    model_dir=DEFAULT_MIC_MODEL_DIR,
    input_csv: str = None,
    output_prefix="./mic_predictions",
    batch_size: int = 256,
    chunk_size: int = 100_000,
    skip_chunks: int = 0,

    seq_col: str = "sequence",
    id_col: str = "seq_id",
    amp_prob_col: str = "amp_probability",
) -> str:
    """
    Agent-callable function that runs MIC prediction and returns a JSON STRING
    describing the outputs (to satisfy Autogen's expectation that function
    responses are strings, not lists/dicts).
    """
    try:
        predictor = MICPredictor(model_dir=model_dir, device=device)
        outputs = predictor.predict_csv(
            input_csv=input_csv,
            output_prefix=output_prefix,
            seq_col=seq_col,
            id_col=id_col,
            amp_prob_col=amp_prob_col,
            batch_size=batch_size,
            chunk_size=chunk_size,
            skip_chunks=skip_chunks,
        )
        summary = {
            "status": "ok",
            "device": str(predictor.device),
            "output_files": outputs,         # absolute paths
            "num_output_files": len(outputs),
            "note": "Each CSV has columns: seq_id, sequence, amp_probability (if provided), and one column per species.",
        }
        return json.dumps(summary)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

import os
import json
from typing import List, Dict
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, EsmConfig, EsmModel
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import PeftModel
from tqdm import tqdm


# -----------------------
#   Model definitions
# -----------------------
class EsmLoRASequenceClassifier(nn.Module):
    """ESM-2 + LoRA + Custom Classifier for AMP prediction"""

    def __init__(self, esm_model, config):
        super().__init__()
        self.esm = esm_model
        self.config = config
        self.num_labels = config.num_labels

        hidden_size = config.hidden_size  # 1280 for ESM-2 650M
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_labels),
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state

        # mean pool over non-padding tokens, excluding CLS (0) and EOS (last non-pad)
        modified_attention_mask = attention_mask.clone()
        modified_attention_mask[:, 0] = 0  # drop CLS
        seq_lens = attention_mask.sum(dim=1)
        for i, L in enumerate(seq_lens):
            if L > 1:
                modified_attention_mask[i, L - 1] = 0  # drop EOS

        expanded_mask = modified_attention_mask.unsqueeze(-1).expand_as(last_hidden_states).float()
        masked_hidden = last_hidden_states * expanded_mask
        sum_hidden = masked_hidden.sum(dim=1)
        denom = expanded_mask.sum(dim=1).clamp(min=1e-9)
        sequence_output = sum_hidden / denom

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=None, attentions=None)


class AMPPredictor:
    """AMP (Antimicrobial Peptide) Predictor using trained LoRA model"""

    def __init__(self, model_dir: str, device: str = "auto"):
        self.model_dir = model_dir

        # Device selection (CUDA -> MPS -> CPU)
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        self.model_info = self._load_model_info()
        self.max_length = self.model_info.get("max_length", 100)
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

        print("AMP Predictor initialized successfully!")

    def _load_model_info(self) -> Dict:
        info_path = os.path.join(self.model_dir, "model_info.json")
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Model info file not found: {info_path}")
        with open(info_path, "r") as f:
            return json.load(f)

    def _load_tokenizer(self):
        tokenizer_path = os.path.join(self.model_dir, "tokenizer")
        if os.path.exists(tokenizer_path):
            print(f"Loading tokenizer from: {tokenizer_path}")
            return AutoTokenizer.from_pretrained(tokenizer_path)
        base_model = self.model_info["base_model_type"]
        print(f"Loading tokenizer from base model: {base_model}")
        return AutoTokenizer.from_pretrained(base_model)

    def _load_model(self):
        """Load ESM backbone + LoRA + classifier (avoid meta-tensor flow)."""
        base_model = self.model_info.get("base_model_type", "facebook/esm2_t33_650M_UR50D")
        custom_model_path = self.model_info.get("custom_model_path")
        try:
            if custom_model_path and os.path.exists(custom_model_path):
                print(f"Loading base ESM model from: {custom_model_path}")
                config = EsmConfig.from_pretrained(custom_model_path, num_labels=2)
                esm_model = EsmModel.from_pretrained(
                    custom_model_path, torch_dtype=torch.float32, device_map=None
                )
                model_source = "custom"
            else:
                raise FileNotFoundError("Custom model not found")
        except Exception:
            print(f"Loading base ESM model from: {base_model}")
            config = EsmConfig.from_pretrained(base_model, num_labels=2)
            esm_model = EsmModel.from_pretrained(base_model, torch_dtype=torch.float32, device_map=None)
            model_source = "base"

        # Attach LoRA adapter (materialized params; no meta tensors)
        lora_adapter_path = os.path.join(self.model_dir, "lora_adapter")
        if not os.path.exists(lora_adapter_path):
            raise FileNotFoundError(f"LoRA adapter not found: {lora_adapter_path}")
        print(f"Loading LoRA adapter from: {lora_adapter_path}")
        esm_model = PeftModel.from_pretrained(
            esm_model,
            lora_adapter_path,
            device_map=None,
            torch_dtype=torch.float32,
            is_trainable=False,
        )

        # Build classifier + load weights
        model = EsmLoRASequenceClassifier(esm_model, config)
        classifier_weights_path = os.path.join(self.model_dir, "classifier_weights.pth")
        if not os.path.exists(classifier_weights_path):
            raise FileNotFoundError(f"Classifier weights not found: {classifier_weights_path}")

        print(f"Loading classifier weights from: {classifier_weights_path}")
        classifier_state_dict = torch.load(classifier_weights_path, map_location="cpu")
        model.classifier.load_state_dict(classifier_state_dict)

        model.to(self.device)
        model.eval()
        print(f"Model loaded successfully with {model_source} ESM backbone + LoRA + classifier")
        return model

    @torch.no_grad()
    def predict_sequences(self, sequences: List[str], seq_ids: List[str] | None = None, batch_size: int = 64) -> pd.DataFrame:
        if seq_ids is None:
            seq_ids = [f"seq_{i}" for i in range(len(sequences))]
        if len(sequences) != len(seq_ids):
            raise ValueError("Number of sequences and sequence IDs must match")

        all_predictions, all_probabilities, all_confidences = [], [], []
        print(f"Predicting {len(sequences)} sequences in batches of {batch_size}...")

        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_sequences = sequences[i : i + batch_size]
            tokenized = self.tokenizer(
                batch_sequences,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            pred_classes = np.argmax(probs, axis=1)
            confidence_scores = np.max(probs, axis=1)

            all_predictions.extend(pred_classes)
            all_probabilities.extend(probs)
            all_confidences.extend(confidence_scores)

        results_df = pd.DataFrame({
            "seq_id": seq_ids,
            "sequence": sequences,
            "predicted_label": all_predictions,
            "prediction": ["AMP" if pred == 1 else "Non-AMP" for pred in all_predictions],
            "confidence_score": all_confidences,
            "amp_probability": [float(prob[1]) for prob in all_probabilities],
            "non_amp_probability": [float(prob[0]) for prob in all_probabilities],
        })
        return results_df

    def predict_fasta(
        self,
        fasta_path: str,
        output_prefix: str,
        batch_size: int = 256,
        chunk_size: int = 100_000,
        skip_chunks: int = 0,
    ) -> List[str]:
        """
        Chunked FASTA prediction. Returns list of output CSV paths.
        """
        print(f"Loading sequences from: {fasta_path}")
        outputs: List[str] = []

        sequences, seq_ids = [], []
        chunk_idx, total = 1, 0

        def write_chunk(_seqs: List[str], _ids: List[str], _idx: int):
            if not _seqs:
                return
            if _idx <= skip_chunks:
                print(f"Skipping chunk {_idx}, size: {len(_seqs)}")
                return
            print(f"Predicting chunk {_idx}, size: {len(_seqs)}")
            df = self.predict_sequences(_seqs, _ids, batch_size=batch_size)
            outname = f"{output_prefix}_{_idx:05d}.csv"
            df.to_csv(outname, index=False)
            print(f"Chunk {_idx} saved to: {outname}")
            outputs.append(os.path.abspath(outname))

        # stream FASTA
        from Bio import SeqIO
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences.append(str(record.seq))
            seq_ids.append(record.id)
            total += 1
            if len(sequences) == chunk_size:
                write_chunk(sequences, seq_ids, chunk_idx)
                sequences, seq_ids = [], []
                chunk_idx += 1

        # tail
        if sequences:
            write_chunk(sequences, seq_ids, chunk_idx)

        print(f"Finished. Total sequences processed: {total}")
        return outputs


# -----------------------
#   Agent-callable tool
# -----------------------
def amp_predict_fasta(
    model_dir=DEFAULT_AMP_MODEL_DIR,
    input_fasta=DEFAULT_SKIN_FASTA,
    output_prefix="./amp_predictions",
    device: str = "auto",
    batch_size: int = 256,
    chunk_size: int = 100_000,
    skip_chunks: int = 0,
) -> str:
    """
    Run AMP classification over a FASTA and write chunked CSVs.
    Returns a JSON STRING summarizing outputs (paths, counts, device).
    """
    try:
        predictor = AMPPredictor(model_dir=model_dir, device=device)
        outputs = predictor.predict_fasta(
            fasta_path=input_fasta,
            output_prefix=output_prefix,
            batch_size=batch_size,
            chunk_size=chunk_size,
            skip_chunks=skip_chunks,
        )
        summary = {
            "status": "ok",
            "device": str(predictor.device),
            "output_files": outputs,
            "num_output_files": len(outputs),
            "columns": [
                "seq_id",
                "sequence",
                "predicted_label",
                "prediction",
                "confidence_score",
                "amp_probability",
                "non_amp_probability",
            ],
        }
        return json.dumps(summary)
    except Exception as e:
        # If anything fails, fall back to passthrough CSV so the pipeline can continue.
        try:
            if "df_in" in locals() and "abs_out" in locals() and "abs_in" in locals():
                df_in.to_csv(abs_out, index=False)
                return json.dumps({
                    "status": "skipped",
                    "reason": f"Toxicity/hemolysis tools failed: {e}",
                    "input_csv": abs_in,
                    "output_csv": abs_out,
                    "num_rows": int(len(df_in)),
                })
        except Exception:
            pass
        return json.dumps({"status": "error", "error": str(e)})


def amp_then_mic_from_fasta(
    amp_model_dir: str = DEFAULT_AMP_MODEL_DIR,
    mic_model_dir: str = DEFAULT_MIC_MODEL_DIR,
    input_fasta: str = None,
    output_csv: str = "./amp_then_mic.csv",
    work_dir: str = workspace_dir,
    device: str = "auto",
    amp_batch_size: int = 256,
    mic_batch_size: int = 256,
    chunk_size: int = 100_000,
    skip_chunks: int = 0,
    min_amp_prob: float = 0.5,
    top_k: int | None = None,
) -> str:
    """
    Agent-callable pipeline:
      1) AMP classify sequences from FASTA (ESM2+LoRA)
      2) Keep only predicted AMPs with amp_probability >= min_amp_prob
      3) Rank by AMP confidence_score (desc), optionally keep top_k when provided
      4) MIC regression (ESM2+LoRA) on the remaining AMPs
      5) Save a single CSV with AMP + MIC results; return JSON string summary.

    Notes:
    - `work_dir` is the base directory where FASTA is read from and CSV is written to.
      Relative `input_fasta`/`output_csv` paths are resolved against `work_dir`.
    - If min_amp_prob <= 0, the AMP filter is disabled (pass-through).

    Returns: JSON string (for Autogen), including output file path and counts.
    """
    import json
    import os
    from Bio import SeqIO

    try:
        # ---------- Resolve paths relative to work_dir ----------
        work_dir = os.path.abspath(work_dir or ".")
        if not os.path.isdir(work_dir) or not os.access(work_dir, os.W_OK):
            fallback = os.path.abspath(os.path.join(os.path.dirname(__file__), "workspace"))
            if os.path.isdir(fallback) and os.access(fallback, os.W_OK):
                work_dir = fallback
        if input_fasta is None:
            raise ValueError("input_fasta must be provided")
        if not os.path.isabs(input_fasta):
            input_fasta = os.path.join(work_dir, input_fasta)
        if not os.path.exists(input_fasta):
            # Fallback: if an absolute path was provided but is invalid, try basename under work_dir.
            candidate = os.path.join(work_dir, os.path.basename(input_fasta))
            if os.path.exists(candidate):
                input_fasta = candidate
            else:
                raise FileNotFoundError(f"FASTA not found: {input_fasta}")

        if not os.path.isabs(output_csv):
            output_csv = os.path.join(work_dir, output_csv)

        # ---------- Load predictors (cached within process) ----------
        amp_pred = _get_amp_predictor(model_dir=amp_model_dir, device=device)
        mic_pred = _get_mic_predictor(model_dir=mic_model_dir, device=device)

        # ---------- Stream FASTA in chunks; run AMP ----------
        total = 0
        chunk_idx = 1
        amp_rows = []  # accumulate AMP predictions (per-sequence rows from AMP predictor)

        seq_buf, id_buf = [], []

        def run_amp_on_buffer(buf_seqs, buf_ids, idx):
            nonlocal amp_rows
            if not buf_seqs:
                return
            if idx <= skip_chunks:
                # skip this chunk entirely
                return
            df_amp = amp_pred.predict_sequences(buf_seqs, buf_ids, batch_size=amp_batch_size)
            amp_rows.append(df_amp)

        for rec in SeqIO.parse(input_fasta, "fasta"):
            seq_buf.append(str(rec.seq))
            id_buf.append(rec.id)
            total += 1
            if len(seq_buf) == chunk_size:
                run_amp_on_buffer(seq_buf, id_buf, chunk_idx)
                seq_buf, id_buf = [], []
                chunk_idx += 1

        # tail
        if seq_buf:
            run_amp_on_buffer(seq_buf, id_buf, chunk_idx)

        # If we skipped all chunks or no sequences
        if not amp_rows:
            summary = {
                "status": "ok",
                "note": "No AMP predictions computed (no chunks processed or empty FASTA after skipping).",
                "total_sequences": total,
                "num_predicted_amps": 0,
                "output_file": None,
            }
            return json.dumps(summary)

        import pandas as pd
        amp_all = pd.concat(amp_rows, ignore_index=True)

        # ---------- Filter to AMPs by label + probability threshold ----------
        min_amp_prob = float(min_amp_prob)
        if min_amp_prob <= 0.0:
            # Allow full pass-through when the threshold is explicitly disabled.
            amp_filtered = amp_all.copy()
        else:
            amp_filtered = amp_all[(amp_all["prediction"] == "AMP") & (amp_all["amp_probability"] >= min_amp_prob)]

        # If nothing passes the filter:
        if amp_filtered.empty:
            # still save a small CSV with just AMP results (empty after filter)
            outdir = os.path.dirname(os.path.abspath(output_csv)) or "."
            os.makedirs(outdir, exist_ok=True)
            amp_filtered.to_csv(output_csv, index=False)

            summary = {
                "status": "ok",
                "device": str(amp_pred.device),
                "total_sequences": int(total),
                "num_amp_candidates": int((amp_all["prediction"] == "AMP").sum()),
                "num_predicted_amps": 0,
                "filter_min_amp_prob": float(min_amp_prob),
                "top_k": top_k,
                "output_file": os.path.abspath(output_csv),
                "columns": list(amp_filtered.columns),
                "preview_rows": [],
                "note": "No sequences met the AMP probability threshold; saved empty filtered AMP CSV.",
            }
            return json.dumps(summary)

        # ---------- Rank by confidence_score (desc) ----------
        amp_filtered = amp_filtered.sort_values("confidence_score", ascending=False)
        pre_topk_count = int(len(amp_filtered))
        if top_k is not None and top_k > 0:
            pre_factor_env = os.environ.get("AMP_PRE_SAFETY_FACTOR")
            if pre_factor_env is not None and str(pre_factor_env).strip() != "":
                try:
                    pre_factor = float(pre_factor_env)
                except Exception:
                    pre_factor = 1.0
            else:
                # Auto-expand small top_k to preserve safety/novelty options before ranking.
                # This keeps runtime reasonable while avoiding over-truncation.
                if top_k <= 200 and len(amp_filtered) >= int(top_k * 3):
                    pre_factor = 5.0
                else:
                    pre_factor = 1.0
            if pre_factor > 1:
                expanded_k = int(top_k * pre_factor)
                amp_filtered = amp_filtered.head(min(expanded_k, len(amp_filtered)))
            else:
                amp_filtered = amp_filtered.head(top_k)
        pre_safety_pool_size = int(len(amp_filtered))

        # ---------- Prepare inputs for MIC ----------
        seq_ids = amp_filtered["seq_id"].astype(str).tolist()
        sequences = amp_filtered["sequence"].astype(str).tolist()
        amp_probs = amp_filtered["amp_probability"].astype(float).tolist()

        # ---------- MIC on AMP subset ----------
        mic_df = mic_pred.predict_sequences(
            sequences=sequences,
            seq_ids=seq_ids,
            amp_probs=amp_probs,               # will be included in output
            batch_size=mic_batch_size,
        )
        # add " MIC" to species columns
        species_cols = [col for col in mic_df.columns if col not in ["seq_id", "sequence", "amp_probability"]]
        mic_df = mic_df.rename(columns={col: f"{col} MIC" for col in species_cols})

        # Add AMP confidence to MIC table
        mic_df = mic_df.merge(
            amp_filtered[["seq_id", "confidence_score"]],
            on="seq_id",
            how="left",
        ).rename(columns={"confidence_score": "amp_confidence_score"})

        # Optional: Reorder columns (seq_id, sequence, amp_probability, amp_confidence_score, species...)
        base_cols = ["seq_id", "sequence", "amp_probability", "amp_confidence_score"]
        species_cols = [c for c in mic_df.columns if c not in base_cols]

        mic_df = mic_df[base_cols + species_cols]

        # ---------- Save final CSV ----------
        outdir = os.path.dirname(os.path.abspath(output_csv)) or "."
        os.makedirs(outdir, exist_ok=True)
        mic_df.to_csv(output_csv, index=False)

        # ---------- Build JSON summary ----------
        note = None
        if top_k is not None and top_k > 0 and pre_safety_pool_size < pre_topk_count:
            note = (
                "Top_k was applied before safety/novelty scoring. "
                "If safety is critical, consider deferring downselection until after safety ranking "
                "or using a broader pre-safety pool."
            )
        if top_k is not None and top_k > 0 and pre_safety_pool_size > top_k:
            note = (note + " " if note else "") + (
                f"Auto-expanded pre-safety pool to {pre_safety_pool_size} for safer reranking."
            )

        summary = {
            "status": "ok",
            "device": str(amp_pred.device),
            "total_sequences": int(total),
            "num_amp_candidates": int(((amp_all["prediction"] == "AMP")).sum()),
            "num_predicted_amps": int(len(mic_df)),
            "filter_min_amp_prob": float(min_amp_prob),
            "top_k": top_k,
            "pre_safety_pool_size": pre_safety_pool_size,
            "output_file": os.path.abspath(output_csv),
        }
        if note:
            summary["note"] = note
        return json.dumps(summary)

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

import os, sys, shlex, shutil, json, subprocess, glob, re
from typing import Optional, List, Dict
import pandas as pd

# ------------------ helpers (robust parsing + normalization) ------------------

def _run_command(cmd: str, work_dir: str=".", stream: bool=True, timeout: Optional[int]=None) -> str:
    proc = subprocess.Popen(
        cmd, cwd=work_dir, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, executable="/bin/bash",
    )
    out = []
    try:
        for line in proc.stdout:
            if stream:
                print(line, end="")
            out.append(line)
        ret = proc.wait(timeout=timeout)
    except Exception as e:
        proc.kill()
        raise RuntimeError(f"Command failed/interrupted: {cmd}\n{e}")
    if ret != 0:
        raise RuntimeError(f"Command returned non-zero exit ({ret}): {cmd}\n{''.join(out)}")
    return "".join(out)

def _pick_col(df: pd.DataFrame, keywords: List[str], prefer_prefix: Optional[List[str]] = None) -> Optional[str]:
    cols = list(df.columns)
    low = {c: c.lower().strip() for c in cols}
    # contains match
    matches = [c for c in cols if any(k.lower() in low[c] for k in keywords)]
    if not matches:
        # normalize spaces/underscores and retry
        canon = {re.sub(r"[\s_]+", "", low[c]): c for c in cols}
        for k in keywords:
            nk = re.sub(r"[\s_]+", "", k.lower().strip())
            if nk in canon:
                return canon[nk]
        return None
    if prefer_prefix:
        for pref in prefer_prefix:
            for c in matches:
                if low[c].startswith(pref.lower()):
                    return c
    return matches[0]

def _norm_seq(s: str) -> str:
    """Uppercase and strip to A–Z (so joins survive casing/whitespace)."""
    return re.sub(r"[^A-Z]", "", str(s).upper())

def _write_fasta_from_csv(df: pd.DataFrame, fasta_path: str) -> int:
    os.makedirs(os.path.dirname(fasta_path) or ".", exist_ok=True)
    n = 0
    with open(fasta_path, "w") as f:
        for _, row in df.iterrows():
            sid = str(row["seq_id"])
            seq = _norm_seq(str(row["sequence"]))
            if not seq:
                continue
            f.write(f">{sid}\n{seq}\n")
            n += 1
    return n

def _ver_tuple(v: Optional[str]) -> tuple:
    try:
        return tuple(int(x) for x in (v or "0").split(".")[:3])
    except Exception:
        return (0,)

def _split_cli_out(path: str, default_dir: str) -> tuple[str, str, str]:
    if os.path.isabs(path):
        out_dir = os.path.dirname(path) or default_dir
        out_name = os.path.basename(path)
    else:
        out_dir = os.path.abspath(default_dir)
        out_name = path
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, out_name, os.path.join(out_dir, out_name)

def _read_tool_csv(path: str, header_tokens: List[str]) -> tuple[pd.DataFrame, Dict[str, str]]:
    """
    Robustly read tool output that may contain banner text before header
    and may be comma/TSV/semicolon/pipe/whitespace separated.

    Returns (df, meta) where meta includes {"sep":..., "skiprows":..., "columns":[...]}
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()

    # locate header line: first line containing any expected token (case-insensitive)
    header_idx = 0
    token_re = re.compile("|".join(re.escape(t.lower()) for t in header_tokens))
    for i, ln in enumerate(lines):
        if token_re.search(ln.lower()):
            header_idx = i
            break

    # try candidate delimiters
    seps = [",", "\t", ";", "|"]
    meta = {"sep": None, "skiprows": header_idx, "columns": []}

    for sep in seps:
        try:
            df = pd.read_csv(path, sep=sep, header=header_idx, engine="python")
            # strip column names
            df.columns = [str(c).strip() for c in df.columns]
            if df.shape[1] > 1:
                meta["sep"] = repr(sep)
                meta["columns"] = list(df.columns)
                return df, meta
        except Exception:
            pass

    # try whitespace-delimited
    try:
        df = pd.read_csv(path, delim_whitespace=True, header=header_idx, engine="python")
        df.columns = [str(c).strip() for c in df.columns]
        if df.shape[1] > 1:
            meta["sep"] = "whitespace"
            meta["columns"] = list(df.columns)
            return df, meta
    except Exception:
        pass

    # last resort: read with default and attempt to split single column by commas/tabs
    try:
        df0 = pd.read_csv(path, header=header_idx, engine="python")
        if df0.shape[1] == 1:
            col = df0.columns[0]
            ser = df0[col].astype(str)
            # heuristic: split on tabs if tabs appear
            if ser.str.contains("\t").any():
                df = ser.str.split("\t", expand=True)
            else:
                df = ser.str.split(",", expand=True)
            df.columns = [f"col{i+1}" for i in range(df.shape[1])]
        else:
            df = df0
        df.columns = [str(c).strip() for c in df.columns]
        meta["sep"] = "fallback"
        meta["columns"] = list(df.columns)
        return df, meta
    except Exception as e:
        raise RuntimeError(f"Failed to parse tool CSV: {path}\n{e}")

# ------------------ main function (same signature) ------------------
def augment_with_toxicity_and_hemolysis(
    input_csv: str,
    output_csv: str = "amp_mic_tox_hemo.csv",
    work_dir: str = workspace_dir,
    # ToxinPred3.0 options
    toxin_model: int = 2,
    toxin_threshold: float = 0.45,
    toxin_display: int = 2,
    # HemoPI2 options
    hemo_job: int = 1,
    hemo_model: int = 4,
    hemo_threshold: float | None = None,
    hemo_display: int = 2,
    hemo_window_len: int = 8,
    keep_intermediates: bool = True,  # preserved for API compatibility (files are removed)
    allow_pip_install: bool = True,
    # required pins
    required_sklearn_version: str = "1.0.2",
    required_numpy_version: str = "1.23.5",
) -> str:
    """
    Append four columns to an AMP/MIC CSV by running two CLIs.

    Output CSV (input columns preserved) will contain these new columns:
      - toxicity_score (float 0–1)            [from ToxinPred3.0 'Hybrid Score' (or 'ML Score' fallback)]
      - toxicity_prediction ('Toxin'|'Non-Toxin')
      - hemolysis_score (float 0–1)           [from HemoPI2 'Hybrid Score' (or other score fallback)]
      - hemolysis_prediction ('Hemolytic'|'Non-Hemolytic')
      - safety_score (float 0–1)              [1 - mean(toxicity_score, hemolysis_score), higher is safer]
      - safety_penalty (float 0–1)            [(1-toxicity_score)*(1-hemolysis_score), stronger penalty]

    Returns a JSON STRING with paths, row counts, and column mapping metadata.
    """
    try:
        # --- Setup & checks
        abs_in = os.path.abspath(os.path.join(work_dir, input_csv) if not os.path.isabs(input_csv) else input_csv)
        abs_out = os.path.abspath(os.path.join(work_dir, output_csv) if not os.path.isabs(output_csv) else output_csv)
        os.makedirs(os.path.dirname(abs_out) or ".", exist_ok=True)
        if not os.path.exists(abs_in):
            return json.dumps({"status": "error", "error": f"Input CSV not found: {abs_in}"})

        df_in = pd.read_csv(abs_in)

        def _skip(reason: str):
            df_in.to_csv(abs_out, index=False)
            return json.dumps({
                "status": "skipped",
                "reason": reason,
                "input_csv": abs_in,
                "output_csv": abs_out,
                "num_rows": int(len(df_in)),
            })

        if shutil.which("toxinpred3") is None:
            return _skip("toxinpred3 CLI not found on PATH.")
        if shutil.which("hemopi2_classification") is None:
            return _skip("hemopi2_classification CLI not found on PATH.")

        # --- Detect input ID & sequence columns robustly
        id_in = None
        for cand in ["seq_id", "seqid", "subject", "id", "identifier", "name"]:
            id_in = id_in or _pick_col(df_in, [cand])
        seq_in = None
        for cand in ["sequence", "peptide sequence", "peptide", "aaseq", "aa sequence"]:
            seq_in = seq_in or _pick_col(df_in, [cand])

        if id_in is None or seq_in is None:
            return json.dumps({
                "status": "error",
                "error": f"Could not detect required columns in input CSV. "
                         f"Need an ID and a sequence column. Detected id={id_in}, seq={seq_in}."
            })

        # --- Write FASTA for CLIs (rename to the expected column names for the helper)
        fasta_path = os.path.abspath(os.path.join(work_dir, "tmp_sequences_for_tools.fa"))
        df_for_fa = df_in[[id_in, seq_in]].rename(columns={id_in: "seq_id", seq_in: "sequence"})
        _ = _write_fasta_from_csv(df_for_fa, fasta_path)

        # --- Pin sklearn & numpy for ToxinPred3.0
        prev_sklearn = None
        prev_numpy = None
        try:
            import sklearn  # noqa
            prev_sklearn = getattr(sys.modules.get("sklearn"), "__version__", None)
        except Exception:
            prev_sklearn = None
        try:
            import numpy  # noqa
            prev_numpy = getattr(sys.modules.get("numpy"), "__version__", None)
        except Exception:
            prev_numpy = None

        need_numpy_pin = _ver_tuple(prev_numpy) >= _ver_tuple("1.24.0")
        need_sklearn_pin = (prev_sklearn or "") != required_sklearn_version
        notes = []

        if allow_pip_install:
            if need_numpy_pin:
                try:
                    _run_command(f"{shlex.quote(sys.executable)} -m pip install --no-input numpy=={required_numpy_version}",
                                 work_dir=work_dir, stream=False, timeout=1800)
                    notes.append(f"Pinned numpy=={required_numpy_version}")
                except RuntimeError as e:
                    notes.append(f"Skip numpy pin: {e}")
            if need_sklearn_pin:
                try:
                    _run_command(f"{shlex.quote(sys.executable)} -m pip install --no-input scikit-learn=={required_sklearn_version}",
                                 work_dir=work_dir, stream=False, timeout=1800)
                    notes.append(f"Pinned scikit-learn=={required_sklearn_version}")
                except RuntimeError as e:
                    notes.append(f"Skip scikit-learn pin: {e}")
        else:
            if need_numpy_pin or need_sklearn_pin:
                notes.append("Skipped numpy/scikit-learn pinning (allow_pip_install=False).")

        # --- ToxinPred3.0
        toxin_csv_abs = os.path.abspath(os.path.join(work_dir, "toxinpred_out.csv"))
        tox_cmd = (
            f"toxinpred3 -i {shlex.quote(fasta_path)} "
            f"-o {shlex.quote(toxin_csv_abs)} -m {int(toxin_model)} -t {float(toxin_threshold)} -d {int(toxin_display)}"
        )
        try:
            _run_command(tox_cmd, work_dir=work_dir, stream=True, timeout=7200)
        except RuntimeError as e:
            msg = str(e)
            if "numpy.dtype size changed" in msg or "binary incompatibility" in msg:
                _run_command(f"{shlex.quote(sys.executable)} -m pip install --no-input numpy=={required_numpy_version}",
                             work_dir=work_dir, stream=False, timeout=1800)
                _run_command(f"{shlex.quote(sys.executable)} -m pip install --no-input scikit-learn=={required_sklearn_version}",
                             work_dir=work_dir, stream=False, timeout=1800)
                notes.append("Detected NumPy ABI error; re-pinned numpy & sklearn and retried.")
                _run_command(tox_cmd, work_dir=work_dir, stream=True, timeout=7200)
            else:
                raise

        # --- HemoPI2 (use -wd <dir> and -o <basename>)
        hemo_csv_requested = os.path.abspath(os.path.join(work_dir, "hemopi2_out.csv"))
        hemo_out_dir, hemo_out_name, hemo_csv_abs = _split_cli_out(hemo_csv_requested, default_dir=os.path.abspath(work_dir))

        if hemo_threshold is None:
            hemo_threshold = 0.55 if hemo_model in (3, 4) else 0.46

        hemo_parts = [
            "hemopi2_classification",
            "-i", fasta_path,
            "-o", hemo_out_name,               # basename only
            "-j", str(int(hemo_job)),
            "-m", str(int(hemo_model)),
            "-t", str(float(hemo_threshold)),
            "-wd", hemo_out_dir,               # directory to write results
            "-d", str(int(hemo_display)),
        ]
        if hemo_job == 2:
            hemo_parts += ["-w", str(int(hemo_window_len))]
        hemo_cmd = " ".join(shlex.quote(p) for p in hemo_parts)
        _run_command(hemo_cmd, work_dir=work_dir, stream=True, timeout=7200)

        # --- Restore env (best-effort)
        if need_sklearn_pin and prev_sklearn and prev_sklearn != required_sklearn_version:
            _run_command(f"{shlex.quote(sys.executable)} -m pip install --no-input scikit-learn=={shlex.quote(prev_sklearn)}",
                         work_dir=work_dir, stream=False, timeout=1800)
            notes.append(f"Restored scikit-learn=={prev_sklearn}")
        elif need_sklearn_pin and not prev_sklearn:
            notes.append(f"No previous scikit-learn detected; kept {required_sklearn_version}.")

        if need_numpy_pin and prev_numpy and prev_numpy != required_numpy_version:
            _run_command(f"{shlex.quote(sys.executable)} -m pip install --no-input numpy=={shlex.quote(prev_numpy)}",
                         work_dir=work_dir, stream=False, timeout=1800)
            notes.append(f"Restored numpy=={prev_numpy}")
        elif need_numpy_pin and not prev_numpy:
            notes.append(f"No previous numpy detected; kept {required_numpy_version}.")

        # --- Merge outputs (join by seq_id from FASTA headers)
        df_tox = pd.read_csv(toxin_csv_abs)
        df_hemo = pd.read_csv(hemo_csv_abs)

        # ----- ToxinPred3.0 -----
        # Expect columns: Subject (ID), Hybrid Score (preferred), Prediction
        if "Subject" not in df_tox.columns:
            raise RuntimeError(
                f"ToxinPred output missing 'Subject' column. Found: {list(df_tox.columns)}"
            )
        tox_score_col = "Hybrid Score" if "Hybrid Score" in df_tox.columns else (
            "ML Score" if "ML Score" in df_tox.columns else None
        )
        if tox_score_col is None:
            # last resort: try to find any score-like column
            for cand in ["Score", "score", "Prob", "Probability", "PPV"]:
                if cand in df_tox.columns:
                    tox_score_col = cand
                    break
        tox_pred_col = "Prediction" if "Prediction" in df_tox.columns else None

        tox_out = df_tox.rename(columns={
            "Subject": "merge_id",
            **({tox_score_col: "toxicity_score"} if tox_score_col else {}),
            **({tox_pred_col: "toxicity_prediction"} if tox_pred_col else {}),
        })
        keep = ["merge_id"]
        if "toxicity_score" in tox_out.columns:
            tox_out["toxicity_score"] = pd.to_numeric(tox_out["toxicity_score"], errors="coerce")
            keep.append("toxicity_score")
        if "toxicity_prediction" in tox_out.columns:
            tox_out["toxicity_prediction"] = tox_out["toxicity_prediction"].astype(str)
            keep.append("toxicity_prediction")
        tox_out = tox_out[keep].drop_duplicates(subset=["merge_id"])

        # ----- HemoPI2 -----
        # Expect columns: SeqID (ID), Hybrid Score, Prediction
        if "SeqID" not in df_hemo.columns:
            raise RuntimeError(
                f"HemoPI2 output missing 'SeqID' column. Found: {list(df_hemo.columns)}"
            )
        hemo_score_col = "Hybrid Score" if "Hybrid Score" in df_hemo.columns else None
        if hemo_score_col is None:
            for cand in ["Probability Score", "Score for Hemolysis", "Score", "score"]:
                if cand in df_hemo.columns:
                    hemo_score_col = cand
                    break
        hemo_pred_col = "Prediction" if "Prediction" in df_hemo.columns else None

        hemo_out = df_hemo.rename(columns={
            "SeqID": "merge_id",
            **({hemo_score_col: "hemolysis_score"} if hemo_score_col else {}),
            **({hemo_pred_col: "hemolysis_prediction"} if hemo_pred_col else {}),
        })
        keep = ["merge_id"]
        if "hemolysis_score" in hemo_out.columns:
            hemo_out["hemolysis_score"] = pd.to_numeric(hemo_out["hemolysis_score"], errors="coerce")
            keep.append("hemolysis_score")
        if "hemolysis_prediction" in hemo_out.columns:
            hemo_out["hemolysis_prediction"] = hemo_out["hemolysis_prediction"].astype(str)
            keep.append("hemolysis_prediction")
        hemo_out = hemo_out[keep].drop_duplicates(subset=["merge_id"])

        # Detect the ID column in your input table (should be 'seq_id' per your files)
        id_in = "seq_id" if "seq_id" in df_in.columns else None
        if id_in is None:
            raise RuntimeError(f"Input table must contain 'seq_id' column. Found: {list(df_in.columns)}")

        # Merge by ID using neutral right key name 'merge_id' (and drop it safely)
        df_aug = df_in.merge(tox_out, left_on=id_in, right_on="merge_id", how="left")
        df_aug = df_aug.drop(columns=["merge_id"], errors="ignore")

        df_aug = df_aug.merge(hemo_out, left_on=id_in, right_on="merge_id", how="left")
        df_aug = df_aug.drop(columns=["merge_id"], errors="ignore")

        # Derive safety columns (soft signals, no hard filtering)
        tox_series = pd.to_numeric(df_aug.get("toxicity_score"), errors="coerce")
        hemo_series = pd.to_numeric(df_aug.get("hemolysis_score"), errors="coerce")
        tox_filled = tox_series.fillna(0.5)
        hemo_filled = hemo_series.fillna(0.5)
        df_aug["safety_score"] = 1.0 - pd.concat([tox_filled, hemo_filled], axis=1).mean(axis=1)
        df_aug["safety_penalty"] = (1.0 - tox_filled) * (1.0 - hemo_filled)

        # Write final single CSV with original AMP/MIC cols + safety columns
        df_aug.to_csv(abs_out, index=False)

        # --- Cleanup: remove temp FASTA and tool CSVs (plus common scratch files)
        for p in [fasta_path, toxin_csv_abs, hemo_csv_abs]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        for pat in ["merci_*.txt", "Sequence_*", "seq.*", "*.aac", "*.dpc", "*.pred", "*.fasta.tmp"]:
            for p in glob.glob(os.path.join(work_dir, pat)):
                try:
                    os.remove(p)
                except Exception:
                    pass

        return json.dumps({
            "status": "ok",
            "device": "cpu",
            "input_csv": abs_in,
            "output_csv": abs_out,
            "num_input_rows": int(len(df_in)),
            "num_written_rows": int(len(df_aug)),
            "intermediate_files": {
                "fasta": "(deleted) " + fasta_path,
                "toxin_csv": "(deleted) " + toxin_csv_abs,
                "hemo_csv": "(deleted) " + hemo_csv_abs
            },
            "column_mapping": {
                "input_id_col": id_in,
                "tox_id_col": "Subject", "tox_score_col": tox_score_col or "Hybrid Score", "tox_pred_col": tox_pred_col or "Prediction",
                "hemo_id_col": "SeqID",  "hemo_score_col": hemo_score_col or "Hybrid Score", "hemo_pred_col": hemo_pred_col or "Prediction",
            },
            "notes": "Temp files removed.",
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

import json, io, contextlib, pandas as pd
from typing import Optional

def _ok(**kw):
    out = {"status": "ok"}; out.update(kw); return json.dumps(out, ensure_ascii=False)
def _err(e):
    return json.dumps({"status": "error", "error": str(e)})

def _preview_df(df: pd.DataFrame, n=5):
    try:
        p = df.head(n).copy()
        for c in p.columns:
            if p[c].dtype == "object":
                p[c] = p[c].astype(str).str.slice(0, 200)
        return p.to_dict(orient="records")
    except Exception:
        return []

# 1) METADATA (context creation)
def csv_metadata(csv_path: str, n_sample: int = 3, work_dir: str = workspace_dir) -> str:
    """
    Return minimal context for prompt augmentation: shape, columns, dtypes, sample rows.
    csv_path may be relative to work_dir.
    """
    try:
        resolved_path = os.path.abspath(os.path.join(work_dir, csv_path)) if not os.path.isabs(csv_path) else csv_path
        df = pd.read_csv(resolved_path)
        meta = {
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "sample": df.head(max(1, n_sample)).to_dict(orient="records"),
        }
        return _ok(**meta)
    except Exception as e:
        return _err(e)

# 2) PYTHON REPL (code generation + execution with df preloaded)
def python_repl_csv(csv_path: str, code: str, save_csv: Optional[str] = None, work_dir: Optional[str] = workspace_dir) -> str:
    """
    Load df from csv_path (resolved under work_dir if relative), then execute arbitrary Python 'code' that assumes df exists.
    If user code sets a variable named RESULT (DataFrame or serializable), it is returned/previewed.
    If save_csv is provided and RESULT is a DataFrame, save it there (resolved under work_dir if relative).
    Returns stdout, basic info, and small previews.
    """
    try:
        base_dir = os.path.abspath(work_dir) if work_dir else os.getcwd()
        in_csv = csv_path if os.path.isabs(csv_path) else os.path.join(base_dir, csv_path)

        df = pd.read_csv(in_csv)
        before_shape = list(df.shape)

        # Build a minimal execution environment, preloading df/pandas/numpy.
        # Use the same dict for globals/locals so helper functions can see these names.
        env = {"pd": pd, "df": df, "np": np}
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(code, env, env)  # df/pd/np are available in globals and locals

        # Collect outputs
        out = {
            "stdout": stdout.getvalue(),
            "before_shape": before_shape,
            "work_dir": base_dir,
            "csv_path": in_csv,
        }

        # If code mutated df (common case), show after-shape
        new_df = env.get("df", None)
        if isinstance(new_df, pd.DataFrame):
            out["after_shape"] = list(new_df.shape)
            out["df_preview"] = _preview_df(new_df)

        # If user provided RESULT, surface it
        result = env.get("RESULT", None)
        if isinstance(result, pd.DataFrame):
            out["result_kind"] = "dataframe"
            out["result_preview"] = _preview_df(result)
            if save_csv:
                out_csv = save_csv if os.path.isabs(save_csv) else os.path.join(base_dir, save_csv)
                os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
                result.to_csv(out_csv, index=False)
                out["saved_csv"] = out_csv
        elif result is not None:
            # Try to JSON-serialize small Python objects
            try:
                json.dumps(result)
                out["result_kind"] = type(result).__name__
                out["result"] = result
            except Exception:
                out["result_kind"] = type(result).__name__
                out["result_repr"] = repr(result)

        return _ok(**out)
    except Exception as e:
        return _err(e)

import os, glob, json
from typing import List, Optional, Dict, Any
from datetime import datetime

def list_find_files_tool(
    work_dir: str = workspace_dir,
    patterns: Optional[List[str]] = None,     # e.g. ["*.csv","*.fa","**/*.json"]
    recursive: bool = True,
    include_hidden: bool = False,
    limit: int = 500,
    sort_by: str = "mtime",                   # "name" | "mtime" | "size"
    descending: bool = True,
    peek_csv: bool = False,
    peek_rows: int = 5,
) -> str:
    """
    General-purpose file lister/finder for agents.
    Returns a JSON string with metadata and (optional) tiny CSV previews.

    Example:
      json.loads(list_find_files_tool("/path", ["*.csv","*.fa"]))
    """
    try:
        base = os.path.abspath(work_dir)
        if not os.path.isdir(base):
            return json.dumps({"status":"error","error":f"work_dir not found: {base}"})

        pats = patterns or ["*"]
        seen, out = set(), []

        def add(path: str):
            ap = os.path.abspath(path)
            if ap in seen or not os.path.isfile(ap):
                return
            if not include_hidden and any(p.startswith(".") for p in ap.split(os.sep)):
                return
            st = os.stat(ap)
            info: Dict[str, Any] = {
                "path": ap,
                "name": os.path.basename(ap),
                "dir": os.path.dirname(ap),
                "ext": os.path.splitext(ap)[1].lower(),
                "size_bytes": st.st_size,
                "mtime": st.st_mtime,
                "mtime_iso": datetime.fromtimestamp(st.st_mtime).isoformat(),
            }
            # tiny CSV peek (safe + optional)
            if peek_csv and info["ext"] == ".csv":
                try:
                    import pandas as pd  # lazy import
                    df = pd.read_csv(ap, nrows=max(1, peek_rows))
                    info["csv_preview"] = {
                        "columns": list(df.columns),
                        "rows": df.to_dict(orient="records"),
                    }
                except Exception as e:
                    info["csv_preview_error"] = str(e)
            seen.add(ap); out.append(info)

        for pat in pats:
            gpat = os.path.join(base, "**", pat) if recursive and "**" not in pat else os.path.join(base, pat)
            for p in glob.iglob(gpat, recursive=recursive):
                add(p)

        key = {"name": lambda x: x["name"].lower(),
               "mtime": lambda x: x["mtime"],
               "size": lambda x: x["size_bytes"]}.get(sort_by, lambda x: x["mtime"])
        out.sort(key=key, reverse=descending)
        if limit and limit > 0:
            out = out[:limit]

        return json.dumps({"status":"ok","work_dir":base,"count":len(out),"files":out})
    except Exception as e:
        return json.dumps({"status":"error","error":str(e)})
#test def amp_then_mic_from_fasta in main



import os, shutil, subprocess, requests
from urllib.parse import quote_plus
from typing import Tuple
from shutil import which as _which


def _run(cmd: list, cwd: str | None = None, env: dict | None = None):
    try:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"Required tool not found: {cmd[0]}") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed ({e.returncode}): {' '.join(cmd)}") from e


def _find_exe(name: str) -> str | None:
    # Try current PATH first
    p = _which(name)
    if p:
        return p
    # Common Homebrew/conda paths on macOS
    candidates = [
        f"/opt/homebrew/bin/{name}",  # Apple Silicon
        f"/usr/local/bin/{name}",  # Intel / older installs
        f"/opt/anaconda3/envs/agent/bin/{name}",  # your env
        f"/opt/anaconda3/bin/{name}",
    ]
    for c in candidates:
        if os.path.exists(c) and os.access(c, os.X_OK):
            return c
    return None


def _cpu_threads() -> int:
    env = os.environ.get("THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, (os.cpu_count() or 1))


def _https_from_ftp(ftp_path: str) -> str:
    if ftp_path.startswith("ftp://"):
        return "https://" + ftp_path[len("ftp://"):]
    if ftp_path.startswith("ftp."):
        return "https://" + ftp_path
    return ftp_path if ftp_path.startswith("http") else "https://" + ftp_path


def download_top_fastq_and_build_smorf(keyword: str, out_dir: str = "workspace", min_len: int = 10, max_len: int = 50) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # --- 1) Taxon lookup
    tax_lookup_url = f"https://www.ebi.ac.uk/ena/taxonomy/rest/scientific-name/{quote_plus(keyword)}"
    tax_resp = requests.get(tax_lookup_url, timeout=60)
    tax_resp.raise_for_status()
    tax_data = tax_resp.json()
    if not isinstance(tax_data, list) or not tax_data or "taxId" not in tax_data[0]:
        raise RuntimeError(f"No taxonomic entry found for '{keyword}'.")
    tax_id = tax_data[0]["taxId"]

    # --- 2) Advanced search for read runs (top 1)
    search_url = "https://www.ebi.ac.uk/ena/portal/api/search"
    params = {
        "result": "read_run",
        "query": f'tax_eq({tax_id}) AND library_layout="PAIRED"',
        "fields": "run_accession,fastq_ftp,library_layout",
        "format": "tsv",
        "limit": 1,
    }
    r = requests.get(search_url, params=params, timeout=60)
    r.raise_for_status()
    lines = r.text.strip().split("\n")
    if len(lines) < 2:
        raise RuntimeError(f"No runs found for taxon ID '{tax_id}'.")
    header = lines[0].split("\t")
    values = lines[1].split("\t")
    rec = dict(zip(header, values))
    fastq_list = rec["fastq_ftp"].split(";")
    if len(fastq_list) < 2:
        raise RuntimeError("Top run is not paired-end (need 2 FASTQs).")

    # --- 3) Download R1/R2
    R1_url = _https_from_ftp(fastq_list[0])
    R2_url = _https_from_ftp(fastq_list[1])
    R1_path = os.path.join(out_dir, os.path.basename(R1_url))
    R2_path = os.path.join(out_dir, os.path.basename(R2_url))
    for url, dest in [(R1_url, R1_path), (R2_url, R2_path)]:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(8192):
                    if chunk: f.write(chunk)

    # --- 4) fastp
    THREADS = str(_cpu_threads())
    clean1 = os.path.join(out_dir, "clean_1.fq.gz")
    clean2 = os.path.join(out_dir, "clean_2.fq.gz")
    _run([
        "fastp", "-i", R1_path, "-I", R2_path,
        "-o", clean1, "-O", clean2,
        "-w", THREADS,
        "-h", os.path.join(out_dir, "fastp.html"),
        "-j", os.path.join(out_dir, "fastp.json"),
    ])

    # --- 5) metaSPAdes
    metaspades_out = os.path.join(out_dir, "metaspades_out")
    contigs_path = os.path.join(metaspades_out, "contigs.fasta")
    _run([
        "metaspades.py",
        "-1", clean1, "-2", clean2,
        "-o", metaspades_out,
        "-t", THREADS, "-m", "16", "--only-assembler",
    ])

    # --- 6) pyrodigal
    all_prot = os.path.join(out_dir, "all_prot.faa")
    all_genes = os.path.join(out_dir, "all_genes.fna")
    all_gff = os.path.join(out_dir, "all_genes.gff")
    # min-gene is in nucleotides; align with min_len (aa) * 3
    min_nt = str(max(1, int(min_len) * 3))
    _run([
        "pyrodigal", "-i", contigs_path, "-p", "meta", "-g", "11",
        "--min-gene", min_nt, "--min-edge-gene", min_nt, "--max-overlap", "29",
        "-a", all_prot, "-d", all_genes, "-o", all_gff,
])

    # --- 7) SMORFs {min_len}–{max_len} aa: try seqkit (no pipe), else pure-Python fallback
    smorfs_path = os.path.join(out_dir, f"smorfs_{min_len}_{max_len}aa.faa")
    seqkit = _find_exe("seqkit")

    if seqkit:
        # Step A: filter length & remove gaps, unwrap
        tmp1 = os.path.join(out_dir, "tmp_seqkit_1.faa")
        _run([seqkit, "seq", "-m", str(min_len), "-M", str(max_len), "-g", "-w", "0", all_prot, "-o", tmp1])
        # Step B: remove trailing '*' with seqkit replace
        _run([seqkit, "replace", "-s", "-p", r"\*$", "-r", "", tmp1, "-o", smorfs_path])
        try:
            os.remove(tmp1)
        except OSError:
            pass
    else:
        # Pure-Python fallback: strip gaps, trailing '*', and length-filter
        def _iter_fasta(path):
            name, seq = None, []
            with open(path, "r") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if line.startswith(">"):
                        if name is not None: yield name, "".join(seq)
                        name, seq = line, []
                    else:
                        seq.append(line)
                if name is not None:
                    yield name, "".join(seq)

        with open(smorfs_path, "w") as out:
            for header, seq in _iter_fasta(all_prot):
                # remove gaps/spaces, trailing '*'
                s = seq.replace("-", "").replace(" ", "")
                if s.endswith("*"):
                    s = s[:-1]
                L = len(s)
                if int(min_len) <= L <= int(max_len):
                    out.write(f"{header}\n{s}\n")

    # --- 8) Clean everything except the SMORFs file
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        if os.path.abspath(path) == os.path.abspath(smorfs_path):
            continue
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except Exception as e:
            raise RuntimeError(f"Failed to delete '{path}': {e}")

    return json.dumps({"status": "ok", "smorfs_path": os.path.abspath(smorfs_path)})


import os, json, numpy as np, pandas as pd, torch
from typing import List, Optional
from transformers import AutoTokenizer, EsmModel
from sklearn.neighbors import NearestNeighbors

def _ok(**kw):
    out = {"status": "ok"}; out.update(kw); return json.dumps(out, ensure_ascii=False)

def _err(e):
    return json.dumps({"status": "error", "error": str(e)})

def _seq_identity(s1: str, s2: str) -> float:
    """
    Return global-alignment sequence identity in [0,1].
    Tries Biopython pairwise2 (globalms with gap penalties), else falls back to a simple ratio.
    """
    try:
        # Lazy import, suppress Biopython's pairwise2 deprecation warning.
        import warnings
        try:
            from Bio import BiopythonDeprecationWarning
        except Exception:
            BiopythonDeprecationWarning = Warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BiopythonDeprecationWarning)
            from Bio import pairwise2
        # match=1, mismatch=0, gap open=-1, gap extend=-0.5 (discourage gaps)
        aln = pairwise2.align.globalms(s1, s2, 1, 0, -1, -0.5, one_alignment_only=True, score_only=False)
        if not aln:
            # Fallback if no alignment produced (unlikely)
            L = max(len(s1), len(s2)) or 1
            return sum(a == b for a, b in zip(s1, s2)) / L
        a1, a2, _score, _beg, _end = aln[0]
        matches = sum((c1 == c2) and (c1 != "-") and (c2 != "-") for c1, c2 in zip(a1, a2))
        L = len(a1) if len(a1) == len(a2) else max(len(a1), len(a2))
        return matches / (L or 1)
    except Exception:
        # Simple, fast fallback (no gaps): compare up to min length, normalize by max length
        L = max(len(s1), len(s2)) or 1
        return sum(a == b for a, b in zip(s1, s2)) / L

@torch.no_grad()
def amp_esm_similarity_and_sequence_identity(
    pred_csv: str = "amp_mic_tox_hemo.csv",     # metagenomics predictions
    verified_csv: str = "fetched_amps.csv",     # verified AMPs (CSV or FASTA)
    # columns
    pred_seq_col: str = "sequence",
    ver_seq_col: str  = "sequence",
    pred_id_col: str  = "seq_id",
    ver_id_col: str   = "seq_id",
    # model/runtime
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    max_length: int = 100,
    batch_size: int = 64,
    # search
    top_k: int = 5,
    # env
    device: str = "auto",
    work_dir: str = "./workspace",
    # output columns
    out_col: str = "esm_knn_mean_similarity",
    out_identity_col: str = "max_seq_identity",
) -> str:
    """
    For each peptide in pred_csv:
      1) Embed with ESM2 and retrieve top_k nearest verified AMPs (cosine).
      2) Write mean of those top_k similarities to `out_col`.
      3) Compute sequence identity vs those same top_k neighbors and store the MAX in `out_identity_col`.
    Finally, sort pred_csv by `out_col` (desc) and overwrite it in place.

    Returns a JSON string summarizing the run (paths, counts, preview).
    """
    try:
        # Resolve paths
        base = os.path.abspath(work_dir)
        os.makedirs(base, exist_ok=True)
        pred_path = pred_csv if os.path.isabs(pred_csv) else os.path.join(base, pred_csv)
        ver_path = verified_csv if os.path.isabs(verified_csv) else os.path.join(base, verified_csv)
        if not os.path.exists(ver_path) and not os.path.isabs(verified_csv):
            repo_root = os.path.abspath(os.path.join(base, os.pardir))
            alt_path = os.path.join(repo_root, verified_csv)
            if os.path.exists(alt_path):
                ver_path = alt_path

        # Load predictions (keep a raw copy to preserve all rows)
        pred_raw = pd.read_csv(pred_path)

        # Load verified AFPs (CSV or FASTA)
        ver_ext = os.path.splitext(ver_path)[1].lower()

        def _read_fasta_to_df(path: str) -> pd.DataFrame:
            ids: List[str] = []
            seqs: List[str] = []
            seq_id = None
            seq_lines: List[str] = []
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(">"):
                        if seq_id is not None:
                            seqs.append("".join(seq_lines))
                            ids.append(seq_id)
                        header = line[1:].strip()
                        seq_id = header.split()[0] if header else f"ver_{len(ids)}"
                        seq_lines = []
                    else:
                        seq_lines.append(line)
                if seq_id is not None:
                    seqs.append("".join(seq_lines))
                    ids.append(seq_id)
            return pd.DataFrame({"seq_id": ids, "sequence": seqs})

        if ver_ext in {".fa", ".fasta", ".faa", ".fsa", ".fas"}:
            ver_df = _read_fasta_to_df(ver_path)
        else:
            ver_df = pd.read_csv(ver_path)

        # Basic checks
        if pred_seq_col not in pred_raw.columns:
            return _err(f"Missing column '{pred_seq_col}' in {pred_path}")
        if ver_seq_col not in ver_df.columns:
            return _err(f"Missing column '{ver_seq_col}' in {ver_path}")
        if pred_id_col not in pred_raw.columns:
            pred_raw[pred_id_col] = [f"pred_{i}" for i in range(len(pred_raw))]
        if ver_id_col not in ver_df.columns:
            ver_df[ver_id_col] = [f"ver_{i}" for i in range(len(ver_df))]

        # Clean sequences
        ver_df = ver_df.dropna(subset=[ver_seq_col]).copy()
        ver_df[ver_seq_col] = ver_df[ver_seq_col].astype(str)
        if len(ver_df) == 0:
            return _err("verified_csv has no sequences after cleaning.")

        valid_mask = pred_raw[pred_seq_col].notna()
        pred_df = pred_raw.loc[valid_mask].copy()
        if len(pred_df) == 0:
            return _err("pred_csv has no sequences after cleaning.")
        pred_df[pred_seq_col] = pred_df[pred_seq_col].astype(str)

        # Device
        if device == "auto":
            if torch.cuda.is_available():
                dev = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = torch.device("mps")
            else:
                dev = torch.device("cpu")
        else:
            dev = torch.device(device)

        # Model & tokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        esm = EsmModel.from_pretrained(model_name).to(dev).eval()

        def _embed(seqs: List[str]) -> np.ndarray:
            """Mean-pool token embeddings excluding CLS/EOS, then L2-normalize."""
            outs = []
            for i in range(0, len(seqs), batch_size):
                batch = seqs[i:i+batch_size]
                toks = tok(batch, padding="longest", truncation=True, max_length=max_length, return_tensors="pt")
                input_ids = toks["input_ids"].to(dev)
                attn      = toks["attention_mask"].to(dev)

                h = esm(input_ids=input_ids, attention_mask=attn).last_hidden_state  # [B,T,H]

                # drop CLS and EOS
                mod_mask = attn.clone()
                mod_mask[:, 0] = 0
                seq_lens = attn.sum(dim=1)
                for j, L in enumerate(seq_lens):
                    if L > 1:
                        mod_mask[j, L - 1] = 0
                mod_mask = mod_mask.unsqueeze(-1).float()

                pooled = (h * mod_mask).sum(dim=1) / mod_mask.sum(dim=1).clamp(min=1e-9)
                outs.append(pooled.detach().cpu().numpy())
            X = np.concatenate(outs, axis=0)
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            return X

        # Embeddings
        pred_seqs = pred_df[pred_seq_col].tolist()
        ver_seqs  = ver_df[ver_seq_col].tolist()
        X_pred = _embed(pred_seqs)
        X_ver  = _embed(ver_seqs)

        # kNN over verified set
        k = max(1, min(top_k, len(ver_df)))
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1)
        nn.fit(X_ver)
        dists, idxs = nn.kneighbors(X_pred, n_neighbors=k, return_distance=True)  # [Npred, k]
        sims = 1.0 - dists  # cosine similarity

        # Mean top-k similarity
        mean_sims = sims.mean(axis=1)

        # Max sequence identity vs the same k neighbors
        max_identities = []
        for i, pred_seq in enumerate(pred_seqs):
            neighbor_idxs = idxs[i]
            best_id = 0.0
            for j in neighbor_idxs:
                ident = _seq_identity(pred_seq, ver_seqs[int(j)])
                if ident > best_id:
                    best_id = ident
            max_identities.append(best_id)

        # Write back to full table (keep NaNs where sequence missing)
        pred_raw[out_col] = np.nan
        pred_raw.loc[valid_mask, out_col] = mean_sims
        pred_raw[out_identity_col] = np.nan
        pred_raw.loc[valid_mask, out_identity_col] = max_identities

        # Sort by mean similarity desc
        pred_raw = pred_raw.sort_values(out_col, ascending=False).reset_index(drop=True)
        pred_raw.to_csv(pred_path, index=False)

        # Preview
        prev_cols = [c for c in [pred_id_col, pred_seq_col, out_col, out_identity_col] if c in pred_raw.columns]
        preview = pred_raw.loc[:, prev_cols].head(5).to_dict(orient="records")

        return _ok(
            work_dir=base,
            updated_csv=os.path.abspath(pred_path),
            model=model_name,
            device=str(dev),
            pred_total=int(len(pred_raw)),
            pred_with_sequences=int(valid_mask.sum()),
            verified_count=int(len(ver_df)),
            top_k=int(k),
            embedding_dim=int(X_pred.shape[1]),
            mean_similarity_col=out_col,
            max_identity_col=out_identity_col,
            preview=preview
        )
    except Exception as e:
        return _err(e)


def amp_identity_similarity(
    pred_csv: str = "amp_mic_tox_hemo.csv",
    train_fasta: str = "data/train_set.fasta",
    # columns
    pred_seq_col: str = "sequence",
    pred_id_col: str = "seq_id",
    # env
    work_dir: str = "./workspace",
    # output columns
    out_identity_col: str = "max_seq_identity",
    # optional
    sort_by_identity: bool = True,
    sort_desc: bool = False,
) -> str:
    """
    For each peptide in pred_csv, compute the MAX global sequence identity (0–1)
    against all sequences in the training FASTA (data/train_set.fasta), write it to
    `out_identity_col`, and overwrite pred_csv in place.

    Note: This does NOT compute ESM similarity.
    """
    try:
        from Bio import SeqIO

        base = os.path.abspath(work_dir)
        os.makedirs(base, exist_ok=True)
        pred_path = pred_csv if os.path.isabs(pred_csv) else os.path.join(base, pred_csv)

        # Resolve train_fasta relative to this file first (repo layout), then work_dir, then CWD.
        if os.path.isabs(train_fasta):
            train_path = train_fasta
        else:
            candidates = [
                os.path.join(os.path.dirname(__file__), train_fasta),
                os.path.join(base, train_fasta),
                os.path.abspath(train_fasta),
            ]
            train_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

        if not os.path.exists(pred_path):
            return _err(f"pred_csv not found: {pred_path}")
        if not os.path.exists(train_path):
            return _err(f"train_fasta not found: {train_path}")

        pred_raw = pd.read_csv(pred_path)
        if pred_seq_col not in pred_raw.columns:
            return _err(f"Missing column '{pred_seq_col}' in {pred_path}")
        if pred_id_col not in pred_raw.columns:
            pred_raw[pred_id_col] = [f"pred_{i}" for i in range(len(pred_raw))]

        train_seqs = []
        for record in SeqIO.parse(train_path, "fasta"):
            s = str(record.seq).strip()
            if s:
                train_seqs.append(s)
        if not train_seqs:
            return _err(f"train_fasta has no sequences after parsing: {train_path}")
        # De-duplicate while preserving order (can save work)
        train_seqs = list(dict.fromkeys(train_seqs))

        valid_mask = pred_raw[pred_seq_col].notna()
        pred_df = pred_raw.loc[valid_mask].copy()
        if len(pred_df) == 0:
            return _err("pred_csv has no sequences after cleaning.")
        pred_df[pred_seq_col] = pred_df[pred_seq_col].astype(str)
        pred_seqs = pred_df[pred_seq_col].tolist()

        # Compute max identity with simple caching for repeated sequences
        cache: dict[str, float] = {}
        max_identities = []
        for pred_seq in pred_seqs:
            if pred_seq in cache:
                max_identities.append(cache[pred_seq])
                continue

            best_id = 0.0
            L1 = len(pred_seq)
            for train_seq in train_seqs:
                if train_seq == pred_seq:
                    best_id = 1.0
                    break
                L2 = len(train_seq)
                # Safe upper bound on identity given only lengths
                upper = (min(L1, L2) / max(L1, L2)) if (L1 and L2) else 0.0
                if upper <= best_id:
                    continue
                ident = _seq_identity(pred_seq, train_seq)
                if ident > best_id:
                    best_id = ident
                    if best_id >= 1.0:
                        break

            cache[pred_seq] = best_id
            max_identities.append(best_id)

        pred_raw[out_identity_col] = np.nan
        pred_raw.loc[valid_mask, out_identity_col] = max_identities

        if sort_by_identity:
            pred_raw = pred_raw.sort_values(out_identity_col, ascending=(not sort_desc)).reset_index(drop=True)

        pred_raw.to_csv(pred_path, index=False)

        prev_cols = [c for c in [pred_id_col, pred_seq_col, out_identity_col] if c in pred_raw.columns]
        preview = pred_raw.loc[:, prev_cols].head(5).to_dict(orient="records")

        return _ok(
            work_dir=base,
            updated_csv=os.path.abspath(pred_path),
            train_fasta=os.path.abspath(train_path),
            pred_total=int(len(pred_raw)),
            pred_with_sequences=int(valid_mask.sum()),
            train_count=int(len(train_seqs)),
            max_identity_col=out_identity_col,
            preview=preview,
        )
    except Exception as e:
        return _err(e)


import os, json, pandas as pd, shutil

def _ok(**kw):
    out = {"status": "ok"}; out.update(kw); return json.dumps(out, ensure_ascii=False)

def _err(e):
    return json.dumps({"status": "error", "error": str(e)})

def _wrap60(s: str) -> str:
    return "\n".join(s[i:i+60] for i in range(0, len(s), 60))

def write_mature_faa_by_predicted_cleavage(
    # ========================== NEW: upstream controls ==========================
    keyword: str,                         # e.g., "Escherichia coli"
    # ===========================================================================
    output_faa: str = "mature_after_cleavage.faa",
    work_dir: str = "./workspace",
) -> str:
    seq_col = "sequence"
    cleavage_col = "predicted_cleavage"
    id_prefix = "seq"
    require_prefix_match = True    # cut only if cleavage is an N-term prefix
    drop_empty_after_cut = True
    uspnet_dir = "./USPNet"
    min_len = 30                    # aa, for smORF range
    max_len = 100                   # aa, for smORF range
    """
    Pipeline:
      1) Download top paired-end FASTQs from ENA by taxon keyword, assemble, call genes,
         and produce smORFs FASTA for [min_len, max_len] aa.
      2) Run USPNet preprocessing + prediction to produce results.csv in work_dir.
      3) For rows with non-empty predicted_cleavage, cut the sequence by that cleavage
         (prefix by default) and write *only the mature* parts to `output_faa`.
      4) Cleanup intermediates: data_list.txt, kingdom_list.txt, results.csv,
         smorfs_{min_len}_{max_len}aa.faa (+ _part copy). Keep only `output_faa`.

    Returns a JSON string with summary and a small preview.
    """
    try:
        base = os.path.abspath(work_dir)
        os.makedirs(base, exist_ok=True)

        # ============================ NEW: Step 1 =============================
        # Build smORFs with your helper (keeps only the smORFs FASTA).
        # This function is assumed to be already defined/imported.
        sm_json = download_top_fastq_and_build_smorf(
            keyword=keyword, out_dir=base, min_len=min_len, max_len=max_len
        )
        if isinstance(sm_json, str):
            sm_res = json.loads(sm_json)
        else:
            sm_res = sm_json
        if sm_res.get("status") != "ok" or "smorfs_path" not in sm_res:
            return _err(f"smORF step failed: {sm_json}")
        smorfs_path = sm_res["smorfs_path"]
        # ============================ NEW: Step 2 =============================
        # Run USPNet preprocessing & prediction to generate results.csv in work_dir
        _run([
            "python", os.path.join(uspnet_dir, "data_processing.py"),
            "--fasta_file", smorfs_path,
            "--data_processed_dir", base,
        ])
        _run([
            "python", os.path.join(uspnet_dir, "predict_fast.py"),
            "--data_dir", base,
            "--group_info", "no_group_info",
            "--model_dir", uspnet_dir,
        ])

        # ========================== CHANGED: input CSV =========================
        in_csv = os.path.join(base, "results.csv")  # always use generated results.csv

        # ============================ Existing logic ===========================
        out_faa = output_faa if os.path.isabs(output_faa) else os.path.join(base, output_faa)
        df = pd.read_csv(in_csv)

        if seq_col not in df.columns:
            return _err(f"Missing column '{seq_col}' in {in_csv}")
        if cleavage_col not in df.columns:
            return _err(f"Missing column '{cleavage_col}' in {in_csv}")

        # Keep only rows with a non-empty cleavage string
        df = df.dropna(subset=[cleavage_col]).copy()
        df[cleavage_col] = df[cleavage_col].astype(str).str.strip()
        df = df[df[cleavage_col] != ""].copy()
        if df.empty:
            open(out_faa, "w").close()

            # ============================ NEW: Cleanup ============================
            # Remove intermediates and keep only out_faa
            for name in [
                "data_list.txt", "kingdom_list.txt", "results.csv",
                f"smorfs_{int(min_len)}_{int(max_len)}aa.faa",
            ]:
                p = os.path.join(base, name)
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

            return _ok(
                updated_faa=os.path.abspath(out_faa),
                written=0,
                note="No rows with predicted_cleavage; wrote empty FASTA and cleaned intermediates.",
                preview=[]
            )

        # Clean sequences (remove spaces, hyphens, trailing '*')
        df[seq_col] = (
            df[seq_col]
            .astype(str)
            .str.replace(r"[\s\-]", "", regex=True)
            .str.replace("*", "", regex=False)
        )

        kept_rows = []
        for i, row in df.iterrows():
            seq = row[seq_col]
            cle = row[cleavage_col]

            if require_prefix_match:
                if not seq.startswith(cle):
                    continue  # skip if cleavage is not at N-terminus
                mature = seq[len(cle):]
            else:
                idx = seq.find(cle)
                if idx == -1:
                    continue
                mature = seq[idx+len(cle):]

            if drop_empty_after_cut and len(mature) == 0:
                continue

            kept_rows.append((i, cle, mature))

        # Write FASTA
        written = 0
        with open(out_faa, "w") as fh:
            for n, (i, cle, mature) in enumerate(kept_rows, start=1):
                header = f">{id_prefix}_{n}|cleavage={cle}|mature_len={len(mature)}"
                fh.write(f"{header}\n{_wrap60(mature)}\n")
                written += 1

        preview = [
            {"id": f"{id_prefix}_{j+1}", "cleavage": cle, "mature_len": len(mat)}
            for j, (_, cle, mat) in enumerate(kept_rows[:5])
        ]

        # ============================ NEW: Cleanup ============================
        for name in [
            "data_list.txt", "kingdom_list.txt", "results.csv",
            f"smorfs_{int(min_len)}_{int(max_len)}aa.faa",
        ]:
            p = os.path.join(base, name)
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        return _ok(
            updated_faa=os.path.abspath(out_faa),
            input_keyword=keyword,
            mature_written=int(written),
            require_prefix_match=bool(require_prefix_match),
            work_dir=base,
            preview=preview
        )

    except Exception as e:
        return _err(e)
import random

# --- 1. OPTIMIZER TOOL: Genetic Algorithm Generator ---
def generate_candidates_genetic(
    input_csv: str,
    output_fasta: str = "generation_next.fasta",
    top_k: int = 10,
    mutation_rate: float = 0.1,
    offspring_per_parent: int = 5,
    work_dir: str = workspace_dir
) -> str:
    """
    Selects top candidates from the previous generation and mutates them 
    to create a new batch. 
    """
    try:
        csv_path = os.path.join(work_dir, input_csv)
        if not os.path.exists(csv_path):
            return json.dumps({"status": "error", "error": f"File not found: {csv_path}"})
        
        df = pd.read_csv(csv_path)
        
        # Simple ranking heuristic (Customise this based on current objectives if needed)
        # Here we rank by AMP probability (desc) and MIC (asc)
        if "amp_probability" in df.columns:
            df = df.sort_values(by=["amp_probability"], ascending=False)
        
        # Select parents
        parents = df.head(top_k)["sequence"].tolist()
        if not parents:
            return json.dumps({"status": "error", "error": "No parents found."})

        new_sequences = set()
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"

        # Generate offspring with at least one mutation to encourage diversity
        for p in parents:
            new_sequences.add(p)
            for _ in range(max(1, int(offspring_per_parent))):
                seq_list = list(p)
                if not seq_list:
                    continue
                num_mut = max(1, int(round(len(seq_list) * mutation_rate)))
                num_mut = min(num_mut, len(seq_list))
                positions = random.sample(range(len(seq_list)), k=num_mut)
                for pos in positions:
                    orig = seq_list[pos]
                    choices = [aa for aa in amino_acids if aa != orig]
                    seq_list[pos] = random.choice(choices) if choices else orig
                new_sequences.add("".join(seq_list))
        
        # Save to FASTA for the next screening round
        out_path = os.path.join(work_dir, output_fasta)
        with open(out_path, "w") as f:
            for i, seq in enumerate(sorted(new_sequences)):
                f.write(f">candidate_{i}\n{seq}\n")
                
        return json.dumps({"status": "ok", "output_file": out_path, "count": len(new_sequences)})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

# --- 1b. OPTIMIZER TOOL: TextGrad-based Generator ---
def generate_candidates_textgrad(
    input_csv: str,
    output_fasta: str = "generation_next.fasta",
    top_k: int = 10,
    optimize_top: int = 5,
    steps: int = 2,
    engine: str | None = None,
    min_len: int = 10,
    max_len: int = 120,
    work_dir: str = workspace_dir,
    score_with_llm: bool = True,
    fallback_to_genetic: bool = True,
    mutation_rate: float = 0.1,
    offspring_per_parent: int = 5,
) -> str:
    """
    Uses TextGrad to refine top peptide candidates from a CSV.
    Falls back to the genetic optimizer if TextGrad is unavailable (optional).
    """
    import json
    import os
    import re

    AA20 = set("ACDEFGHIKLMNPQRSTVWY")
    HYDROPHOBIC = set("AILMFWVYC")

    def normalize_sequence(seq: str) -> str:
        seq = re.sub(r"\s+", "", str(seq).upper())
        return "".join(aa for aa in seq if aa in AA20)

    def heuristic_features(seq: str) -> tuple[int, float, float, float]:
        length = len(seq)
        if length == 0:
            return 0, 0.0, 0.0, 0.0
        pos = seq.count("K") + seq.count("R") + 0.1 * seq.count("H")
        neg = seq.count("D") + seq.count("E")
        net_charge = pos - neg
        hydrophobic = sum(seq.count(aa) for aa in HYDROPHOBIC)
        hydrophobic_frac = hydrophobic / length
        length_score = 1.0 - min(abs(length - 40), 40) / 40.0
        charge_score = max(min((net_charge + 1.0) / 10.0, 1.0), 0.0)
        hydro_score = 1.0 - min(abs(hydrophobic_frac - 0.45), 0.45) / 0.45
        heuristic_score = 0.35 * charge_score + 0.35 * hydro_score + 0.30 * length_score
        return length, net_charge, hydrophobic_frac, heuristic_score

    def parse_llm_score(text: str) -> float:
        text = text.strip()
        score = None
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                score = float(data.get("score"))
            except Exception:
                score = None
        if score is None:
            num = re.findall(r"(\d+(?:\.\d+)?)", text)
            if num:
                score = float(num[0])
        if score is None:
            score = 0.0
        return max(0.0, min(100.0, score))

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

    try:
        csv_path = os.path.join(work_dir, input_csv) if not os.path.isabs(input_csv) else input_csv
        if not os.path.exists(csv_path):
            return json.dumps({"status": "error", "error": f"File not found: {csv_path}"})

        df = pd.read_csv(csv_path)
        if "sequence" not in df.columns:
            return json.dumps({"status": "error", "error": "Input CSV missing 'sequence' column."})

        rank_col = None
        if "composite_score" in df.columns:
            rank_col = "composite_score"
        elif "amp_probability" in df.columns:
            rank_col = "amp_probability"
        elif "amp_confidence_score" in df.columns:
            rank_col = "amp_confidence_score"

        candidates = []
        for idx, row in df.iterrows():
            seq = normalize_sequence(row.get("sequence", ""))
            if not seq:
                continue
            length, net_charge, hydro_frac, h_score = heuristic_features(seq)
            if not (min_len <= length <= max_len):
                continue
            base_score = float(row.get(rank_col, h_score)) if rank_col else h_score
            seq_id = str(row.get("seq_id", f"seq_{idx}"))
            candidates.append(
                {
                    "seq_id": seq_id,
                    "sequence": seq,
                    "length": length,
                    "net_charge": net_charge,
                    "hydrophobic_frac": hydro_frac,
                    "heuristic_score": h_score,
                    "base_score": base_score,
                }
            )

        if not candidates:
            return json.dumps({"status": "error", "error": "No valid sequences available for TextGrad."})

        candidates.sort(key=lambda c: c["base_score"], reverse=True)
        candidates = candidates[: max(1, int(top_k))]

        try:
            import textgrad as tg
            from textgrad.engine import get_engine
        except Exception as exc:
            if fallback_to_genetic:
                result = generate_candidates_genetic(
                    input_csv=input_csv,
                    output_fasta=output_fasta,
                    top_k=top_k,
                    mutation_rate=mutation_rate,
                    offspring_per_parent=offspring_per_parent,
                    work_dir=work_dir,
                )
                try:
                    payload = json.loads(result)
                    payload["note"] = f"textgrad unavailable; fell back to genetic optimizer ({exc})."
                    return json.dumps(payload)
                except Exception:
                    return result
            return json.dumps({"status": "error", "error": f"textgrad import failed: {exc}"})

        engine_name = resolve_engine_name(engine or os.environ.get("TEXTGRAD_ENGINE", "gpt-5.2"))
        engine_obj = get_engine(engine_name, cache=False if engine_name.startswith("experimental:") else True)
        # TextGrad may already have an engine set from a prior generation.
        tg.set_backward_engine(engine_obj, override=True)

        llm_scores = {}
        if score_with_llm:
            score_system_prompt = tg.Variable(
                "You are a strict antimicrobial peptide evaluator. "
                "Output only JSON with numeric score and a one-sentence rationale.",
                requires_grad=False,
                role_description="system prompt for AMP scoring",
            )
            scorer = tg.BlackboxLLM(engine=engine_obj, system_prompt=score_system_prompt)
            for cand in candidates:
                prompt = (
                    "Score the following peptide for antimicrobial activity (0-100, higher is better). "
                    "Consider charge, amphipathicity, hydrophobicity balance, stability, "
                    "and likely toxicity. Respond with JSON only: "
                    "{\"score\": <number>, \"rationale\": \"<one sentence>\"}.\n\n"
                    f"Sequence:\n{cand['sequence']}"
                )
                out = scorer(tg.Variable(prompt, requires_grad=False, role_description="AMP scoring prompt"))
                llm_scores[cand["seq_id"]] = parse_llm_score(out.get_value())

        def rank_key(cand: dict) -> float:
            if score_with_llm and cand["seq_id"] in llm_scores:
                return llm_scores[cand["seq_id"]]
            return cand["base_score"]

        candidates.sort(key=rank_key, reverse=True)
        to_optimize = candidates[: min(int(optimize_top), len(candidates))]

        optimized = []
        for cand in to_optimize:
            seq = cand["sequence"]
            constraints = [
                f"Output only uppercase amino-acid letters (ACDEFGHIKLMNPQRSTVWY) with length exactly {cand['length']}.",
                "No spaces, punctuation, or extra text.",
            ]
            role_desc = (
                f"Antimicrobial peptide sequence (length {cand['length']}) "
                "using only canonical amino acids."
            )
            seq_var = tg.Variable(seq, requires_grad=True, role_description=role_desc)
            eval_instruction = tg.Variable(
                "You are evaluating an antimicrobial peptide sequence. Provide concise, actionable "
                "feedback to improve antimicrobial potency and spectrum while maintaining stability "
                "and minimizing toxicity. Focus on cationic charge and amphipathic structure. "
                "Do NOT rewrite the sequence; only provide feedback.",
                requires_grad=False,
                role_description="evaluation criteria for AMP optimization",
            )
            loss_fn = tg.TextLoss(eval_instruction, engine=engine_obj)
            optimizer = tg.TextualGradientDescent(
                parameters=[seq_var],
                engine=engine_obj,
                constraints=constraints,
                verbose=0,
            )
            for _ in range(max(1, int(steps))):
                loss = loss_fn(seq_var)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                cleaned = normalize_sequence(seq_var.get_value())
                if len(cleaned) != cand["length"]:
                    cleaned = seq
                seq_var.set_value(cleaned)
            improved = seq_var.get_value()
            optimized.append(
                {
                    "parent_id": cand["seq_id"],
                    "sequence": improved,
                    "length": cand["length"],
                }
            )

        new_sequences = []
        seen = set()
        for cand in candidates:
            seq = cand["sequence"]
            if seq not in seen:
                seen.add(seq)
                new_sequences.append(("orig", cand["seq_id"], seq))
        for idx, cand in enumerate(optimized, start=1):
            seq = cand["sequence"]
            if seq not in seen:
                seen.add(seq)
                new_sequences.append(("textgrad", cand["parent_id"], seq))

        out_path = os.path.join(work_dir, output_fasta) if not os.path.isabs(output_fasta) else output_fasta
        with open(out_path, "w") as fh:
            for i, (method, parent_id, seq) in enumerate(new_sequences, start=1):
                fh.write(f">candidate_{i}|{method}|parent={parent_id}\n{seq}\n")

        return json.dumps(
            {
                "status": "ok",
                "output_file": out_path,
                "count": len(new_sequences),
                "optimized": len(optimized),
                "engine": engine_name,
            }
        )
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

# --- 2. ANALYZER TOOL: Population Statistics ---
def analyze_generation_stats(
    input_csv: str,
    work_dir: str = workspace_dir
) -> str:
    """
    Analyzes the population to detect failure modes (e.g., high toxicity, low diversity).
    Returns a text report for the Planner.
    """
    try:
        csv_path = os.path.join(work_dir, input_csv)
        df = pd.read_csv(csv_path)
        
        report = []
        report.append(f"Analysis of {len(df)} candidates:")

        if len(df) == 0:
            report.append("- No candidates available for statistics. Consider lowering the AMP threshold or increasing input diversity.")
            return "\n".join(report)
        
        # 1. Potency Stats
        if "amp_probability" in df.columns:
            avg_amp = df["amp_probability"].mean()
            report.append(f"- Avg AMP Probability: {avg_amp:.2f}")
            
        # 2. Safety Check (Reward Hacking Detection)
        if "toxicity_prediction" in df.columns:
            toxic_count = df[df["toxicity_prediction"] == "Toxin"].shape[0]
            toxic_rate = toxic_count / len(df)
            report.append(f"- Toxicity Rate: {toxic_rate:.2%}")
            if toxic_rate > 0.3:
                report.append(
                    "WARNING: High toxicity detected. Consider strengthening continuous safety terms "
                    "(toxicity_score and hemolysis_score) in the composite score, while avoiding hard filters "
                    "or label-gated penalties."
                )
        if "hemolysis_prediction" in df.columns:
            hemo_count = df[df["hemolysis_prediction"] == "Hemolytic"].shape[0]
            hemo_rate = hemo_count / len(df)
            report.append(f"- Hemolysis Rate: {hemo_rate:.2%}")
            if hemo_rate > 0.3:
                report.append(
                    "WARNING: High hemolysis detected. Consider increasing the impact of hemolysis signals in the composite score."
                )
        
        # 3. Diversity Check
        unique_seqs = df["sequence"].nunique()
        diversity_ratio = unique_seqs / len(df)
        report.append(f"- Diversity Ratio: {diversity_ratio:.2f}")
        if diversity_ratio < 0.2:
            report.append("WARNING: Low diversity. The optimizer has collapsed to a local optimum.")

        return "\n".join(report)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


    
if __name__ == "__main__":

    # result_json = amp_then_mic_from_fasta(
    #     input_fasta="./workspace/smorfs_30_100aa_part.faa",
    
    # )
    #test amp_identity_similarity
    # result_json = amp_identity_similarity(
    #     pred_csv="amp_then_mic.csv",
    #     train_fasta="data/train_set.fasta",
    # )
    # print(result_json)

    # print(result_json)
    #test augment_with_toxicity_and_hemolysis in main
    result_json = augment_with_toxicity_and_hemolysis(
        input_csv="amp_then_mic.csv"
    )
    print(result_json)
    #test amp_similarity_via_esm in main
    # result_json = amp_esm_similarity_and_sequence_identity(
    #
    # )
    # print(result_json)
    # #test csv_metadata function
    # result_json = csv_metadata(
    #     csv_path="amp_then_mic.csv"
    # )
    # print(result_json)
    # #{"status": "ok", "shape": [81, 33], "columns": ["seq_id", "sequence", "amp_probability", "amp_confidence_score", "Candida albicans ATCC 10231 MIC", "Candida albicans ATCC 90028 MIC", "Trichosporon beigelii KCTC 7707 MIC", "Candida albicans (merged) MIC", "Candida albicans ATCC 14053 MIC", "Aspergillus fumigatus MIC", "Candida albicans MIC", "Botrytis cinerea MIC", "Fusarium oxysporum MIC", "Cryptococcus neoformans MIC", "Saccharomyces cerevisiae MIC", "Aspergillus niger MIC", "Cryptococcus neoformans ATCC 208821 MIC", "Candida glabrata MIC", "Candida krusei ATCC 6258 MIC", "Candida parapsilosis ATCC 22019 MIC", "Candida tropicalis (merged) MIC", "Candida tropicalis MIC", "Candida albicans ATCC 2002 MIC", "Candida albicans MDM8 MIC", "Cryptococcus neoformans ATCC 32045 MIC", "Candida albicans NCPF 1467 MIC", "Candida albicans SC5314 MIC", "Cryptococcus neoformans var. grubii H99 MIC", "Candida albicans F7-39/IDE99 MIC", "Candida albicans NCYC 1467 MIC", "Candida albicans CGMCC 2.2086 MIC", "Candida tropicalis CGMCC 2.1975 MIC", "Candida parapsilosis CGMCC 2.3989 MIC"], "dtypes": {"seq_id": "object", "sequence": "object", "amp_probability": "float64", "amp_confidence_score": "float64", "Candida albicans ATCC 10231 MIC": "float64", "Candida albicans ATCC 90028 MIC": "float64", "Trichosporon beigelii KCTC 7707 MIC": "float64", "Candida albicans (merged) MIC": "float64", "Candida albicans ATCC 14053 MIC": "float64", "Aspergillus fumigatus MIC": "float64", "Candida albicans MIC": "float64", "Botrytis cinerea MIC": "float64", "Fusarium oxysporum MIC": "float64", "Cryptococcus neoformans MIC": "float64", "Saccharomyces cerevisiae MIC": "float64", "Aspergillus niger MIC": "float64", "Cryptococcus neoformans ATCC 208821 MIC": "float64", "Candida glabrata MIC": "float64", "Candida krusei ATCC 6258 MIC": "float64", "Candida parapsilosis ATCC 22019 MIC": "float64", "Candida tropicalis (merged) MIC": "float64", "Candida tropicalis MIC": "float64", "Candida albicans ATCC 2002 MIC": "float64", "Candida albicans MDM8 MIC": "float64", "Cryptococcus neoformans ATCC 32045 MIC": "float64", "Candida albicans NCPF 1467 MIC": "float64", "Candida albicans SC5314 MIC": "float64", "Cryptococcus neoformans var. grubii H99 MIC": "float64", "Candida albicans F7-39/IDE99 MIC": "float64", "Candida albicans NCYC 1467 MIC": "float64", "Candida albicans CGMCC 2.2086 MIC": "float64", "Candida tropicalis CGMCC 2.1975 MIC": "float64", "Candida parapsilosis CGMCC 2.3989 MIC": "float64"}, "sample": [{"seq_id": "PPYF01661569.1_70", "sequence": "KRCW", "amp_probability": 1.0, "amp_confidence_score": 1.0, "Candida albicans ATCC 10231 MIC": 4.670016288757324, "Candida albicans ATCC 90028 MIC": 4.687061786651611, "Trichosporon beigelii KCTC 7707 MIC": 4.5220746994018555, "Candida albicans (merged) MIC": 4.3345866203308105, "Candida albicans ATCC 14053 MIC": 4.706592559814453, "Aspergillus fumigatus MIC": 4.0894622802734375, "Candida albicans MIC": 4.83432674407959, "Botrytis cinerea MIC": 5.043056011199951, "Fusarium oxysporum MIC": 4.699488639831543, "Cryptococcus neoformans MIC": 4.969705104827881, "Saccharomyces cerevisiae MIC": 4.916717529296875, "Aspergillus niger MIC": 4.731508255004883, "Cryptococcus neoformans ATCC 208821 MIC": 4.856675624847412, "Candida glabrata MIC": 4.608177185058594, "Candida krusei ATCC 6258 MIC": 4.682780742645264, "Candida parapsilosis ATCC 22019 MIC": 4.686709880828857, "Candida tropicalis (merged) MIC": 4.4560866355896, "Candida tropicalis MIC": 5.180983543395996, "Candida albicans ATCC 2002 MIC": 4.558351993560791, "Candida albicans MDM8 MIC": 5.138688087463379, "Cryptococcus neoformans ATCC 32045 MIC": 4.860554218292236, "Candida albicans NCPF 1467 MIC": 3.76721453666687, "Candida albicans SC5314 MIC": 4.906017303466797, "Cryptococcus neoformans var. grubii H99 MIC": 4.827498435974121, "Candida albicans F7-39/IDE99 MIC": 4.703360557556152, "Candida albicans NCYC 1467 MIC": 4.188450813293457, "Candida albicans CGMCC 2.2086 MIC": 4.429083824157715, "Candida tropicalis CGMCC 2.1975 MIC": 4.552253246307373, "Candida parapsilosis CGMCC 2.3989 MIC": 4.162430286407471}, {"seq_id": "PPYF01661569.1_153", "sequence": "RWLCLKIIRWTLKHHLAKLSFIILTM", "amp_probability": 1.0, "amp_confidence_score": 1.0, "Candida albicans ATCC 10231 MIC": 4.741568565368652, "Candida albicans ATCC 90028 MIC": 4.970732688903809, "Trichosporon beigelii KCTC 7707 MIC": 4.904779434204102, "Candida albicans (merged) MIC": 4.708497524261475, "Candida albicans ATCC 14053 MIC": 4.942145824432373, "Aspergillus fumigatus MIC": 4.573589324951172, "Candida albicans MIC": 5.075128078460693, "Botrytis cinerea MIC": 5.106470584869385, "Fusarium oxysporum MIC": 5.099607467651367, "Cryptococcus neoformans MIC": 5.199337482452393, "Saccharomyces cerevisiae MIC": 5.032827377319336, "Aspergillus niger MIC": 4.847152233123779, "Cryptococcus neoformans ATCC 208821 MIC": 5.350526809692383, "Candida glabrata MIC": 4.753293037414551, "Candida krusei ATCC 6258 MIC": 4.904621601104736, "Candida parapsilosis ATCC 22019 MIC": 4.851356506347656, "Candida tropicalis (merged) MIC": 4.788769245147705, "Candida tropicalis MIC": 5.216342449188232, "Candida albicans ATCC 2002 MIC": 4.981624126434326, "Candida albicans MDM8 MIC": 5.51904821395874, "Cryptococcus neoformans ATCC 32045 MIC": 4.967784404754639, "Candida albicans NCPF 1467 MIC": 4.2679243087768555, "Candida albicans SC5314 MIC": 5.06397008895874, "Cryptococcus neoformans var. grubii H99 MIC": 5.0572381019592285, "Candida albicans F7-39/IDE99 MIC": 5.02913236618042, "Candida albicans NCYC 1467 MIC": 4.613397121429443, "Candida albicans CGMCC 2.2086 MIC": 4.716075420379639, "Candida tropicalis CGMCC 2.1975 MIC": 4.953575611114502, "Candida parapsilosis CGMCC 2.3989 MIC": 4.581109046936035}, {"seq_id": "PPYF01661569.1_157", "sequence": "RYSHFGFKLLRVRL", "amp_probability": 0.9999998807907104, "amp_confidence_score": 0.9999999, "Candida albicans ATCC 10231 MIC": 4.415383338928223, "Candida albicans ATCC 90028 MIC": 4.439706325531006, "Trichosporon beigelii KCTC 7707 MIC": 4.50365686416626, "Candida albicans (merged) MIC": 4.1733317375183105, "Candida albicans ATCC 14053 MIC": 4.478733062744141, "Aspergillus fumigatus MIC": 3.884709358215332, "Candida albicans MIC": 4.598654270172119, "Botrytis cinerea MIC": 4.7141547203063965, "Fusarium oxysporum MIC": 4.6377034187316895, "Cryptococcus neoformans MIC": 4.690714359283447, "Saccharomyces cerevisiae MIC": 4.706359386444092, "Aspergillus niger MIC": 4.360422134399414, "Cryptococcus neoformans ATCC 208821 MIC": 4.607785701751709, "Candida glabrata MIC": 4.3909478187561035, "Candida krusei ATCC 6258 MIC": 4.437902927398682, "Candida parapsilosis ATCC 22019 MIC": 4.396844863891602, "Candida tropicalis (merged) MIC": 4.382416725158691, "Candida tropicalis MIC": 4.736248970031738, "Candida albicans ATCC 2002 MIC": 4.576724052429199, "Candida albicans MDM8 MIC": 4.916494369506836, "Cryptococcus neoformans ATCC 32045 MIC": 4.468189239501953, "Candida albicans NCPF 1467 MIC": 3.8617684841156006, "Candida albicans SC5314 MIC": 4.490790843963623, "Cryptococcus neoformans var. grubii H99 MIC": 4.593255519866943, "Candida albicans F7-39/IDE99 MIC": 4.586994647979736, "Candida albicans NCYC 1467 MIC": 4.110734939575195, "Candida albicans CGMCC 2.2086 MIC": 4.2391862869262695, "Candida tropicalis CGMCC 2.1975 MIC": 4.349660396575928, "Candida parapsilosis CGMCC 2.3989 MIC": 4.050870418548584}]}
    # #test python_repl_csv function, get means of all MIC columns
    # result_json = python_repl_csv(
    #     csv_path="amp_then_mic.csv",
    #     code="RESULT = df.filter(like='MIC').mean()\nprint(RESULT)",
    # )
    # print(result_json)
    # #test save RESULT as CSV
    # result_json = python_repl_csv(
    #     csv_path="amp_then_mic.csv",
    #     code="RESULT = df.filter(like='MIC').mean().to_frame().reset_index().rename(columns={'index':'MIC_type',0:'mean_value'})",
    #     save_csv="mic_means.csv"
    # )
    # print(result_json)
    # #test list_find_files_tool function
    # result_json = list_find_files_tool(
    #     patterns=["*.py","*.csv"],
    #     peek_csv=True,
    # )
    # print(result_json)
    #
    # # test fetch_amp function
    # result_json = download_top_fastq_and_build_smorf(keyword="human gut metagenome",min_len = 30,max_len=100)
    #result_json = fetch_amps(kingdom="Fungi")
