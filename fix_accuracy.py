"""
Script to fix accuracy calculation in CSV file.

Old measure: Only calculated accuracy for fields where true != 0, ignoring false positives.
New measure:
  - If true != 0: direct comparison (relevant field)
  - If true == 0 AND predict != 0: penalty (false positive)
  - If true == 0 AND predict == 0: ignored (not relevant, no error)
"""

import pandas as pd
import argparse
from pathlib import Path


def compute_accuracy(row):
    """
    Calcola l'accuracy per i 4 campi numerici.
    - Se true != 0: confronto diretto (campo rilevante)
    - Se true == 0 e predict != 0: penalità (falso positivo)
    - Se true == 0 e predict == 0: ignorato (non rilevante, nessun errore)
    """
    fields = [
        ('ore_ordinarie', 'ore_ordinarie_true', 'ore_ordinarie_predict'),
        ('ore_straordinarie', 'ore_straordinarie_true', 'ore_straordinarie_predict'),
        ('ore_viaggio', 'ore_viaggio_true', 'ore_viaggio_predict'),
        ('inefficiency', 'inefficiency_true', 'inefficiency_predict'),
    ]

    denominator = 0
    accumulator = 0
    acc_results = {}

    for acc_key, true_key, pred_key in fields:
        true_value = row[true_key]
        pred_value = row[pred_key]

        if true_value != 0:
            # campo rilevante: confronto diretto
            match = 1 if pred_value == true_value else 0
            acc_results[f'{acc_key}_acc'] = match
            accumulator += match
            denominator += 1

        elif pred_value != 0:
            # falso positivo: true è 0 ma il modello ha predetto qualcosa
            acc_results[f'{acc_key}_acc'] = 0
            denominator += 1
            # accumulator non incrementa -> penalità

        else:
            # true == 0 e predict == 0 -> non rilevante, keep existing or set to NaN
            acc_results[f'{acc_key}_acc'] = None

    if denominator == 0:
        # tutti i campi sono 0 sia in true che in predict (es. giornata di riposo)
        acc_results['ore_ordinarie_acc'] = 1
        acc_results['ore_straordinarie_acc'] = 1
        acc_results['ore_viaggio_acc'] = 1
        acc_results['inefficiency_acc'] = 1
        acc_results['tot_acc'] = 1.0
    else:
        acc_results['tot_acc'] = accumulator / denominator

    return acc_results


def fix_csv_accuracy(input_path: str, output_path: str = None):
    """
    Read CSV, recalculate accuracy columns, and save to output.
    """
    input_path = Path(input_path)

    if output_path is None:
        # Default: create a new file with '_fixed' suffix
        output_path = input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}"
    else:
        output_path = Path(output_path)

    print(f"Reading: {input_path}")
    df = pd.read_csv(input_path)

    print(f"Total rows: {len(df)}")

    # Store old values for comparison
    old_tot_acc = df['tot_acc'].copy()

    # Apply new accuracy calculation
    print("Recalculating accuracy...")
    for idx, row in df.iterrows():
        acc_results = compute_accuracy(row)
        for key, value in acc_results.items():
            if value is not None:
                df.at[idx, key] = value

    # Report changes
    changed_rows = (old_tot_acc != df['tot_acc']).sum()
    print(f"Rows with changed tot_acc: {changed_rows}")

    # Show some statistics
    print(f"\nOld tot_acc mean: {old_tot_acc.mean():.4f}")
    print(f"New tot_acc mean: {df['tot_acc'].mean():.4f}")

    # Save
    print(f"\nSaving to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Done!")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix accuracy calculation in CSV file")
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("-o", "--output", help="Output CSV file path (default: input_fixed.csv)")

    args = parser.parse_args()
    fix_csv_accuracy(args.input, args.output)
