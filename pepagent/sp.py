'''
# Data processing. Processed data is saved in data_processed/ by default.
python data_processing.py --fasta_file <fasta_file_path> --data_processed_dir <data_directory_path>

# Prediction. use 'python predict_fast.py --data_dir <data_directory_path> --group_info no_group_info' if lack of organism group information.
python predict_fast.py --data_dir <data_directory_path>
'''
#30-100 in length

import os, json, pandas as pd, re

def _ok(**kw):
    out = {"status": "ok"}; out.update(kw); return json.dumps(out, ensure_ascii=False)

def _err(e):
    return json.dumps({"status": "error", "error": str(e)})

def _wrap60(s: str) -> str:
    return "\n".join(s[i:i+60] for i in range(0, len(s), 60))

def write_mature_faa_by_predicted_cleavage(
    csv_path: str = "results.csv",
    output_faa: str = "mature_after_cleavage.faa",
    seq_col: str = "sequence",
    cleavage_col: str = "predicted_cleavage",
    id_prefix: str = "seq",
    require_prefix_match: bool = True,   # True: only cut if cleavage is a prefix; False: cut at first occurrence
    drop_empty_after_cut: bool = True,   # drop rows where the remaining sequence is empty
    work_dir: str = "./workspace",
) -> str:
    """
    Read results.csv with columns [sequence, predicted_cleavage, ...].
    For rows where predicted_cleavage is non-empty:
      - If require_prefix_match=True, cut ONLY if sequence startswith(predicted_cleavage).
      - Else, cut at the first occurrence of predicted_cleavage.
    Write remaining parts to a FASTA (.faa). Rows without predicted_cleavage are skipped.

    Returns a JSON string with summary and a small preview.
    """
    try:
        base = os.path.abspath(work_dir)
        os.makedirs(base, exist_ok=True)
        in_csv  = csv_path if os.path.isabs(csv_path) else os.path.join(base, csv_path)
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
            return _ok(updated_faa=os.path.abspath(out_faa), written=0, preview=[])

        # Clean sequences (remove spaces, hyphens, trailing '*')
        df[seq_col] = (
            df[seq_col]
            .astype(str)
            .str.replace(r"[\s\-]", "", regex=True)
            .str.replace("*", "", regex=False)
        )

        # Do the cut
        mature_list = []
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
                    continue  # cleavage string not found
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

        # Preview
        preview = [
            {"id": f"{id_prefix}_{j+1}", "cleavage": cle, "mature_len": len(mat)}
            for j, (_, cle, mat) in enumerate(kept_rows[:5])
        ]

        return _ok(
            updated_faa=os.path.abspath(out_faa),
            input_csv=os.path.abspath(in_csv),
            rows_with_cleavage=int(len(df)),
            written=int(written),
            require_prefix_match=bool(require_prefix_match),
            preview=preview
        )
    except Exception as e:
        return _err(e)

if __name__ == "__main__":
    # Example usage
    result = write_mature_faa_by_predicted_cleavage(
        csv_path="results.csv",
        output_faa="mature_after_cleavage.faa",
        work_dir="./workspace"
    )
    print(result)