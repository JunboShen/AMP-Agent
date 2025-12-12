#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import autogen
import os

# IMPORTANT: never hardcode credentials in the repo.
# Set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`/`OPENAI_API_BASE`)
# via your shell environment (or a local .env that is NOT committed).

DEFAULT_MODEL_LIST = ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]


def build_config_list(model_list: list[str] | None = None) -> list[dict]:
    """
    Build an Autogen config list from model names.

    This is best-effort to keep `import pepagent.llm_config` working even when
    the user hasn't configured credentials yet.
    """
    try:
        return autogen.config_list_from_models(model_list=model_list or DEFAULT_MODEL_LIST)
    except Exception:
        return []


config_list = build_config_list()

llm_config = {
    # Generate functions config for the Tool
    "functions": [

        {
            "name": "get_FASTA_from_name",
            "description": "With a protein name as input, provides a FASTA sequence of amino acids.",
            "parameters": {
                "type": "object",
                "properties": {
                    "protein_name": {
                        "type": "string",
                        "description": "Name of a protein.",
                    }
                },
                "required": ["protein_name"],
            },
        },
        {
            "name": "save_to_csv_file",
            "description": "With a JSON dictionary as input, saves the data to a csv file with a provided name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_JSON_dictionary": {
                        "type": "string",
                        "description": "The input JSON dictionary.",
                    },
                    "output_csv_name": {
                        "type": "string",
                        "description": "The output name for the csv file.",
                    }
                },
                "required": ["input_JSON_dictionary", "output_csv_name"],
            },
        },

        {
            "name": "retrieve_content",
            "description": "An expert in retrieving knowledge about protein, their mechanical properties, structures, and PDB names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to be used to retrieve detailed knowledge. ",
                    }
                },
                "required": ["message"],
            },
        },

        {
            "name": "coords_from_SMILES",
            "description": "With a SMILES string as input, provides atom type and coordinates of a molecule.",
            "parameters": {
                "type": "object",
                "properties": {
                    "SMILES": {
                        "type": "string",
                        "description": "SMILES string.",
                    }
                },
                "required": ["SMILES"],
            },
        },
{
            "name": "fetch_amps",
            "description": (
                "Retrieve antimicrobial peptides (AMPs) from the DBAASP database using a variety of "
                "filters.  The function wraps the /peptides endpoint and supports filters such as "
                "peptide name, UniProt accession, target group and species, sequence or length range, "
                "synthesis type, structural complexity, termini modifications, presence of unusual "
                "amino acids, biological kingdom, source organism, and 3D structure availability.  "
                "Results are paginated via the limit and offset parameters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Filter peptides by name (name.value).",
                    },
                    "uniprot": {
                        "type": "string",
                        "description": "Filter by UniProt accession number (uniprot.value).",
                    },
                    "target_group": {
                        "type": "string",
                        "description": (
                            "Filter by the high-level target organism group (targetGroup.value). "
                            "Valid values include 'Archaea', 'Biofilm', 'Cancer', 'Fungus', 'Gram+', "
                            "'Gram-', 'Insect', 'Mammalian Cell', 'Mollicute', 'Nematode', 'Parasite', "
                            "'Protista' and 'Virus'."
                        ),
                        "enum": [
                            "Archaea", "Biofilm", "Cancer", "Fungus", "Gram+",
                            "Gram-", "Insect", "Mammalian Cell", "Mollicute",
                            "Nematode", "Parasite", "Protista", "Virus"
                        ],
                    },
                    "target_species": {
                        "type": "string",
                        "description": "Filter by the specific target species (targetSpecies.value).",
                    },
                    "sequence": {
                        "type": "string",
                        "description": "Filter by an exact amino‑acid sequence (sequence.value).",
                    },
                    "sequence_length_range": {
                        "type": "array",
                        "description": (
                            "Two‑element array specifying minimum and maximum peptide length as integers; "
                            "maps to sequence.length=min-max in the API."
                        ),
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "synthesis_type": {
                        "type": "string",
                        "description": (
                            "Filter by biosynthetic origin (synthesisType.value).  Valid values are "
                            "'Ribosomal', 'Nonribosomal', or 'Synthetic'."
                        ),
                        "enum": ["Ribosomal", "Nonribosomal", "Synthetic"],
                    },
                    "complexity": {
                        "type": "string",
                        "description": (
                            "Filter by structural complexity (complexity.value). "
                            "Valid values are 'monomer', 'multimer', or 'multi_peptide'."
                        ),
                        "enum": ["monomer", "multimer", "multi_peptide"],
                    },
                    "n_terminus": {
                        "type": "string",
                        "description": "Filter by N‑terminal modification (nTerminus.value).",
                    },
                    "c_terminus": {
                        "type": "string",
                        "description": "Filter by C‑terminal modification (cTerminus.value).",
                    },
                    "unusual_aa": {
                        "type": "string",
                        "description": "Filter by presence of unusual amino acids (unusualAminoAcid.value).",
                    },
                    "kingdom": {
                        "type": "string",
                        "description": (
                            "Filter by the biological kingdom of origin (kingdom.value). "
                            "Examples: 'Animalia', 'Archaea', 'Bacteria', 'Fungi', 'Plantae', 'Protista', 'Virus'."
                        ),
                    },
                    "source": {
                        "type": "string",
                        "description": "Filter by source organism (source.value).",
                    },
                    "has_structure": {
                        "type": "string",
                        "description": (
                            "Filter by 3D structure availability (threeDStructure.value).  "
                            "Valid values are 'without_structure', 'pdb', 'md_model', and 'pdb_md_model'."
                        ),
                        "enum": ["without_structure", "pdb", "md_model", "pdb_md_model"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of records to return (default 100).",
                        "default": 100,
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Offset for pagination (default 0).",
                        "default": 0,
                    },
                    "output_csv": {
                        "type": "string",
                        "description": "Path to output CSV file to write the results.",
                        "default": "fetched_amps.csv",
                    },
                    "work_dir": {
                        "type": "string",
                        "description": "Base directory used to resolve relative csv_path.",
                        "default": "./workspace"
                    },
                },

                "required": [],
            },
        },
{
  "name": "augment_with_toxicity_and_hemolysis",
  "description": "Append four columns to an AMP/MIC CSV by running two CLIs. Input CSV must include seq_id and sequence. Writes a single CSV (UTF-8) at `output_csv` containing the original columns plus: toxicity_score (float 0–1), toxicity_prediction ('toxic'|'non-toxic'), hemolysis_score (float 0–1), hemolysis_prediction ('hemolytic'|'non-hemolytic'). Returns a JSON STRING with paths, row counts, and column mapping used for merging.",
  "parameters": {
    "type": "object",
    "properties": {
      "input_csv": {"type": "string", "description": "Path to AMP/MIC CSV containing columns seq_id and sequence."},
      "output_csv": {"type": "string", "description": "Output CSV path to write.", "default": "amp_mic_tox_hemo.csv"},
      "work_dir": {"type": "string", "description": "Working directory for CLIs and temporary files.", "default": "./workspace"},
    },
    "required": ["input_csv"]
  }
},

{
  "name": "amp_then_mic_from_fasta",
  "description": "Run a 2-stage pipeline on a FASTA: (1) AMP classification → keep sequences with prediction=='AMP' AND amp_probability>=min_amp_prob, rank by confidence_score desc, optionally keep top_k; (2) MIC regression only on the kept AMPs. Writes ONE CSV at output_csv (UTF-8, comma-delimited) with columns: seq_id (string), sequence (string), amp_probability (float in [0,1]), amp_confidence_score (float in [0,1]), followed by one float column per species listed in the MIC model's model_info['species'] (predicted MIC values in the model’s native scale). Rows are sorted by amp_confidence_score (desc). Returns a JSON STRING (not a Python dict) with: { status: 'ok'|'error', device: 'cpu'|'cuda'|'mps', total_sequences: int, num_amp_candidates: int, num_predicted_amps: int, filter_min_amp_prob: float, top_k: int|null, output_file: absolute string path to the CSV, columns: string[], preview_rows: list of up to 3 row objects }. If no sequences pass the AMP filter, an empty CSV is still written with the same header, and num_predicted_amps==0.",
    "parameters": {
    "type": "object",
    "properties": {
      "input_fasta":   {"type": "string", "description": "Path to input FASTA"},
      "output_csv":    {"type": "string", "description": "Path for the combined AMP+MIC CSV", "default": "amp_then_mic.csv"},
      "min_amp_prob":  {"type": "number", "default": 0.5, "description": "Keep only AMPs with P(AMP) >= this"},
      "top_k":         {"type": "integer", "nullable": True, "description": "If set, keep only top-k by confidence_score"},
      "work_dir": {"type": "string", "description": "Working directory for CLIs and temporary files.", "default": "./workspace"},
    },
    "required": ["input_fasta"]
  }
},

{
    "name": "csv_metadata",
    "description": "Return minimal context (shape, columns, dtypes, small sample) for a CSV. If csv_path is relative, it is resolved under work_dir.",
    "parameters": {
        "type": "object",
        "properties": {
            "csv_path": {
                "type": "string",
                "description": "Path to the CSV file. May be relative to work_dir."
            },
            "n_sample": {
                "type": "integer",
                "description": "How many sample rows to include in the context.",
                "default": 3,
                "minimum": 1
            },
            "work_dir": {
                "type": "string",
                "description": "Base directory used to resolve relative csv_path.",
                "default": "./workspace"
            }
        },
        "required": ["csv_path"]
    }
},

{
    "name": "python_repl_csv",
    "description": "Execute arbitrary Python code with df preloaded from the given CSV (resolved under work_dir). If code assigns RESULT to a DataFrame or JSON-serializable object, it is returned/previewed. Optionally save RESULT as a CSV.",
    "parameters": {
        "type": "object",
        "properties": {
            "csv_path": {
                "type": "string",
                "description": "Path to the CSV to load as df. May be relative to work_dir."
            },
            "code": {
                "type": "string",
                "description": "Python code to run. Assumes a pandas DataFrame named df is already available. May mutate df and/or set RESULT."
            },
            "save_csv": {
                "type": "string",
                "description": "If provided and RESULT is a DataFrame, save it to this path (resolved under work_dir)."
            },
            "work_dir": {
                "type": "string",
                "description": "Base directory used to resolve relative paths.",
                "default": "./workspace"
            }
        },
        "required": ["csv_path", "code"]
    }
},

{
    "name": "list_find_files_tool",
    "description": "List/find files under work_dir using glob patterns. Returns file metadata (path, size, mtime, ext). Can optionally include tiny CSV previews.",
    "parameters": {
        "type": "object",
        "properties": {
            "work_dir": {
                "type": "string",
                "description": "Base directory to search.",
                "default": "./workspace"
            },
            "patterns": {
                "type": "array",
                "description": "Glob patterns to match (e.g., ['*.csv','*.fa','**/*.json']).",
                "items": {"type": "string"},
                "default": ["*"]
            },
            "recursive": {
                "type": "boolean",
                "description": "Recurse into subdirectories.",
                "default": True
            },
            "include_hidden": {
                "type": "boolean",
                "description": "Include hidden files and folders (names starting with '.').",
                "default": False
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of files to return.",
                "default": 500,
                "minimum": 1
            },
            "sort_by": {
                "type": "string",
                "description": "Sort key for results.",
                "enum": ["name", "mtime", "size"],
                "default": "mtime"
            },
            "descending": {
                "type": "boolean",
                "description": "Sort descending if true.",
                "default": True
            },
            "peek_csv": {
                "type": "boolean",
                "description": "If true, include small previews for CSV files (columns + first few rows).",
                "default": False
            },
            "peek_rows": {
                "type": "integer",
                "description": "How many rows to read for each CSV preview.",
                "default": 5,
                "minimum": 1
            }
        },
        "required": []
    }
},
{
  "name": "download_top_fastq_and_build_smorf",
  "description": "Search ENA by a taxonomic keyword (preferably metagenome), fetch the top paired-end read_run, download its FASTQs, run a lightweight assembly pipeline, and emit a FASTA of small ORFs (10–50 aa). Steps: (1) resolve taxId from the keyword via ENA taxonomy; (2) query ENA read_run for the first paired-end run; (3) download R1/R2 (FTP URLs are auto-converted to HTTPS); (4) quality-trim with fastp; (5) assemble with metaSPAdes; (6) call genes with pyrodigal (meta mode) and translate; (7) extract proteins 10–50 aa (uses seqkit if available, else a pure-Python fallback); (8) delete all intermediates and keep only the smORFs FASTA. Uses THREADS env var or CPU count for parallelism. Requires external CLIs: fastp, metaspades.py, pyrodigal (seqkit optional). Returns a JSON STRING like {\"status\":\"ok\",\"smorfs_path\":\"/abs/path/smorfs_10_50aa.faa\"}.",
  "parameters": {
    "type": "object",
    "properties": {
      "keyword": {
        "type": "string",
        "description": "Taxonomic keyword (e.g., species/genus name, example: human skin metagenome) used to resolve an ENA taxId and locate runs. Template example: 'xxxx metagenome'"
      },
      "out_dir": {
        "type": "string",
        "description": "Working/output directory. Will be created if missing; will be cleaned so only the smORFs FASTA remains.",
        "default": "workspace"
      }
    },
    "required": ["keyword"]
  }
},
	{
	  "name": "amp_esm_similarity_and_sequence_identity",
	  "description": "Embed peptides with ESM2 and, for each row in pred_csv, retrieve its top_k nearest neighbors from verified_csv (cosine on L2-normalized embeddings). Write the mean cosine similarity across those neighbors to out_col and the MAX global sequence identity (Biopython pairwise2 if available; simple fallback otherwise) against the same neighbors to out_identity_col. Then sort pred_csv by out_col (descending) and overwrite it in place. All relative paths resolve under work_dir. Returns a JSON STRING summarizing updated_csv path, counts, model/device, top_k, embedding_dim, and a small preview.",
	  "parameters": {
	    "type": "object",
	    "properties": {
      "pred_csv":      { "type": "string",  "description": "Metagenomic AMP predictions CSV to update/overwrite.", "default": "amp_mic_tox_hemo.csv" },
      "verified_csv":  { "type": "string",  "description": "Verified AMPs CSV used as the neighbor/reference set.", "default": "fetched_amps.csv" },

      "pred_seq_col":  { "type": "string",  "description": "Sequence column in pred_csv.", "default": "sequence" },
      "ver_seq_col":   { "type": "string",  "description": "Sequence column in verified_csv.", "default": "sequence" },
      "pred_id_col":   { "type": "string",  "description": "Identifier column in pred_csv (auto-filled if missing).", "default": "seq_id" },
      "ver_id_col":    { "type": "string",  "description": "Identifier column in verified_csv (auto-filled if missing).", "default": "seq_id" },

      "model_name":    { "type": "string",  "description": "ESM2 checkpoint to use.", "default": "facebook/esm2_t6_8M_UR50D" },
      "max_length":    { "type": "integer", "description": "Tokenizer truncation length.", "default": 100, "minimum": 16 },
      "batch_size":    { "type": "integer", "description": "Batch size for embedding.", "default": 64, "minimum": 1 },

      "top_k":         { "type": "integer", "description": "Number of nearest verified neighbors to average for similarity and to scan for max identity.", "default": 5, "minimum": 1 },

      "device":        { "type": "string",  "description": "Computation device.", "enum": ["auto","cpu","cuda","mps"], "default": "auto" },
      "work_dir":      { "type": "string",  "description": "Base directory for resolving relative paths.", "default": "./workspace" },

      "out_col":           { "type": "string", "description": "Name of the output column for mean cosine similarity written into pred_csv.", "default": "esm_knn_mean_similarity" },
      "out_identity_col":  { "type": "string", "description": "Name of the output column for max sequence identity (0–1) written into pred_csv.", "default": "max_seq_identity" }
	    },
	    "required": []
	  }
	},
	{
	  "name": "amp_identity_similarity",
	  "description": "For each row in pred_csv, compute the MAX global sequence identity (0–1) versus all sequences in train_fasta (default data/train_set.fasta). Write it to out_identity_col and overwrite pred_csv in place. Does not compute ESM similarity. Relative pred_csv resolves under work_dir; train_fasta resolves relative to the repo (agent_functions.py) first, then work_dir, then CWD. Returns a JSON STRING summarizing updated_csv path, counts, train_fasta path, and a small preview.",
	  "parameters": {
	    "type": "object",
	    "properties": {
	      "pred_csv":          { "type": "string",  "description": "Predictions CSV to update/overwrite.", "default": "amp_mic_tox_hemo.csv" },
	      "train_fasta":       { "type": "string",  "description": "Training-set FASTA to compare against.", "default": "data/train_set.fasta" },
	      "pred_seq_col":      { "type": "string",  "description": "Sequence column in pred_csv.", "default": "sequence" },
	      "pred_id_col":       { "type": "string",  "description": "Identifier column in pred_csv (auto-filled if missing).", "default": "seq_id" },
	      "work_dir":          { "type": "string",  "description": "Base directory for resolving relative pred_csv paths.", "default": "./workspace" },
	      "out_identity_col":  { "type": "string",  "description": "Name of the output column for max sequence identity (0–1) written into pred_csv.", "default": "max_seq_identity" },
	      "sort_by_identity":  { "type": "boolean", "description": "If true, sort pred_csv by out_identity_col after computing.", "default": True },
	      "sort_desc":         { "type": "boolean", "description": "Sort descending if sort_by_identity is true.", "default": False }
	    },
	    "required": []
	  }
	}
# {
#     "name": "mic_predict_csv",
#     "description": "Run the MIC regression model (ESM-2 + LoRA) over sequences in a CSV and write chunked result CSV(s). Returns a JSON string with output file paths and counts.",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "model_dir": {
#                 "type": "string",
#                 "description": "Directory with model_info.json, tokenizer/, lora_adapter/, and regression_head.pth"
#             },
#             "input_csv": {
#                 "type": "string",
#                 "description": "Input CSV path containing sequences (and optionally amp_probability)"
#             },
#             "output_prefix": {
#                 "type": "string",
#                 "description": "Prefix for output CSVs, e.g., '/tmp/mic_predictions'"
#             },
#             "device": {
#                 "type": "string",
#                 "description": "Compute device",
#                 "enum": ["auto", "cuda", "cpu", "mps"],
#                 "default": "auto"
#             },
#             "skip_chunks": { "type": "integer", "default": 0 },
#             "batch_size":  { "type": "integer", "default": 256 },
#             "chunk_size":  { "type": "integer", "default": 100000 },
#             "seq_col":     { "type": "string",  "default": "sequence" },
#             "id_col":      { "type": "string",  "default": "seq_id" },
#             "amp_prob_col":{ "type": "string",  "default": "amp_probability" }
#         },
#         "required": ["model_dir", "input_csv", "output_prefix"]
#     }
# },
# {
#   "name": "amp_predict_fasta",
#   "description": "Run the AMP classifier (ESM-2 + LoRA) over sequences in a FASTA and write chunked CSV(s). Returns a JSON string with output file paths and counts.",
#   "parameters": {
#     "type": "object",
#     "properties": {
#       "model_dir": {
#         "type": "string",
#         "description": "Directory with model_info.json, tokenizer/, lora_adapter/, classifier_weights.pth"
#       },
#       "input_fasta": { "type": "string", "description": "Input FASTA path with sequences to score" },
#       "output_prefix": { "type": "string", "description": "Prefix for output CSVs, e.g. '/tmp/amp_preds/output'" },
#       "device": { "type": "string", "enum": ["auto", "cuda", "cpu", "mps"], "default": "auto" },
#       "batch_size": { "type": "integer", "default": 256 },
#       "chunk_size": { "type": "integer", "default": 100000 },
#       "skip_chunks": { "type": "integer", "default": 0 }
#     },
#     "required": ["model_dir", "input_fasta", "output_prefix"]
#   }
# },
# {
#   "name": "amp_then_mic_from_fasta",
#   "description": "Run AMP classification from FASTA, keep AMPs by probability threshold, rank by confidence, then run MIC regression on those AMPs. Writes one combined CSV and returns a JSON string summary.",
#   "parameters": {
#     "type": "object",
#     "properties": {
#       "input_fasta":   {"type": "string", "description": "Path to input FASTA"},
#       "output_csv":    {"type": "string", "description": "Path for the combined AMP+MIC CSV"},
#       "min_amp_prob":  {"type": "number", "default": 0.5, "description": "Keep only AMPs with P(AMP) >= this"},
#       "top_k":         {"type": "integer", "nullable": True, "description": "If set, keep only top-k by confidence_score"}
#     },
#     "required": ["input_fasta", "output_csv"]
#   }
# },
    ],
    "config_list": config_list,  # Assuming you have this defined elsewhere
    # "request_timeout": 120,
}
