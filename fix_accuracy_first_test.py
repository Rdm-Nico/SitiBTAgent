"""
Script to fix accuracy columns in data/first_test/results_1.csv ... results_7.csv.

Ground truth is in labels_tecnico_with_msg.csv (48 rows).
Mapping: row i in the concatenated dataframe → label at index (i % 48).

New accuracy measure:
  - true != 0 and pred == true  → 1 (correct)
  - true != 0 and pred != true  → 0 (wrong)
  - true == 0 and pred != 0     → 0 + denominator++ (false positive penalty)
  - true == 0 and pred == 0     → ignored
  - all fields 0/false in both  → all acc = 1, tot_acc = 1.0
"""

import json
import pandas as pd
from pathlib import Path


BASE_DIR     = Path(__file__).parent / "data" / "first_test"
LABELS_PATH  = BASE_DIR / "labels_tecnico_with_msg.csv"
RESULT_FILES = [BASE_DIR / f"results_{i}.csv" for i in range(1, 8)]


def normalize_inefficiency(value) -> float:
    """Convert True/NaN/False to 1.0 or 0.0."""
    if value is True or value == 1:
        return 1.0
    return 0.0


def compute_accuracy(true_oo, true_os, true_ov, true_ineff,
                     pred_oo, pred_os, pred_ov, pred_ineff) -> dict:
    """
    Same logic as fix_accuracy.py but for the first-test format.
    Penalizes false positives; ignores true-negative pairs.
    """
    fields = [
        ('ore_ordinarie',     true_oo,    pred_oo),
        ('ore_straordinarie', true_os,    pred_os),
        ('ore_viaggio',       true_ov,    pred_ov),
        ('innefficiency',     true_ineff, pred_ineff),
    ]

    denominator = 0
    accumulator = 0
    acc = {}

    for key, true_val, pred_val in fields:
        if true_val != 0:
            match = 1 if pred_val == true_val else 0
            acc[f'{key}_acc'] = match
            accumulator += match
            denominator += 1
        elif pred_val != 0:
            # false positive
            acc[f'{key}_acc'] = 0
            denominator += 1
        else:
            acc[f'{key}_acc'] = None   # not relevant, will keep original

    if denominator == 0:
        acc = {
            'ore_ordinarie_acc':     1,
            'ore_straordinarie_acc': 1,
            'ore_viaggio_acc':       1,
            'innefficiency_acc':     1,
            'tot_acc':               1.0,
        }
    else:
        acc['tot_acc'] = accumulator / denominator

    return acc


def fix_all():
    labels = pd.read_csv(LABELS_PATH)
    print(f"Labels loaded: {len(labels)} rows")

    # Load all 7 files and track their original sizes
    dfs = []
    for path in RESULT_FILES:
        df = pd.read_csv(path)
        dfs.append(df)
        print(f"  {path.name}: {len(df)} rows")

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal concatenated rows: {len(df_all)}")

    indici_originali = [i % len(labels) for i in range(len(df_all))]

    changed = 0
    old_tot = df_all['tot_acc'].copy()

    for i, row in df_all.iterrows():
        label_row = labels.iloc[indici_originali[i]]

        # Ground truth
        true_oo    = float(label_row['ORE_ORDINARIE'])
        true_os    = float(label_row['ORE_STRAORDINARIE'])
        true_ov    = float(label_row['ORE_VIAGGIO'])
        true_ineff = normalize_inefficiency(label_row['INEFFICIENCY'])

        # Predictions from JSON
        try:
            resp = json.loads(row['model_response'])
        except (json.JSONDecodeError, TypeError):
            print(f"  Warning: could not parse model_response at row {i}, skipping")
            continue

        pred_oo    = float(resp.get('ore_ordinarie', 0) or 0)
        pred_os    = float(resp.get('ore_straordinarie', 0) or 0)
        pred_ov    = float(resp.get('ore_viaggio', 0) or 0)
        pred_ineff = normalize_inefficiency(resp.get('innefficienza', False))

        acc = compute_accuracy(true_oo, true_os, true_ov, true_ineff,
                               pred_oo, pred_os, pred_ov, pred_ineff)

        for col, val in acc.items():
            if val is not None:
                df_all.at[i, col] = val

    changed = (old_tot != df_all['tot_acc']).sum()
    print(f"\nRows with changed tot_acc: {changed}")
    print(f"Old tot_acc mean: {old_tot.mean():.4f}")
    print(f"New tot_acc mean: {df_all['tot_acc'].mean():.4f}")

    # Split back and overwrite each result file
    print()
    start = 0
    for path, df_orig in zip(RESULT_FILES, dfs):
        n = len(df_orig)
        df_fixed = df_all.iloc[start:start + n].reset_index(drop=True)
        # Restore original index column if present
        if 'Unnamed: 0' in df_orig.columns:
            df_fixed['Unnamed: 0'] = df_orig['Unnamed: 0'].values
        df_fixed.to_csv(path, index=False)
        print(f"  Saved {path.name} ({n} rows)")
        start += n

    print("\nDone!")


if __name__ == "__main__":
    fix_all()
