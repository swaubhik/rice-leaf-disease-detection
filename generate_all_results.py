#!/usr/bin/env python3
"""Generate a consolidated all_results.csv from Ray Tune trial folders.

Usage:
  python generate_all_results.py --tune-dir ray_results --experiment-name rice_disease_tune

This looks for `progress.csv` inside each trial folder and picks the last row.
It merges them into `ray_results/<experiment>/all_results.csv` so `analyze_tuning_results.py`
and other scripts can run even if `tune_hyperparameters.py` didn't complete cleanly.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import pandas as pd


def read_last_progress_csv(p: Path):
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if df.shape[0] == 0:
            return None
        last = df.iloc[-1]
        # convert to dict
        return last.to_dict()
    except Exception:
        return None


def read_params_json(p: Path):
    if not p.exists():
        return None
    try:
        with open(p, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate all_results.csv from Ray Tune trial folders')
    parser.add_argument('--tune-dir', type=str, default='ray_results')
    parser.add_argument('--experiment-name', type=str, required=True)
    args = parser.parse_args()

    exp_dir = Path(args.tune_dir) / args.experiment_name
    if not exp_dir.exists():
        print(f"Experiment directory not found: {exp_dir}")
        return 1

    rows = []
    for child in sorted(exp_dir.iterdir()):
        if not child.is_dir():
            continue
        # look for progress.csv (prefer) or result.json
        progress = child / 'progress.csv'
        params = child / 'params.json'
        last = read_last_progress_csv(progress)
        if last is None:
            # try result.json (may contain multiple json lines)
            result_file = child / 'result.json'
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        lines = [l.strip() for l in f.readlines() if l.strip()]
                    if not lines:
                        continue
                    # try to parse last JSON object
                    last_json = json.loads(lines[-1])
                    last = last_json
                except Exception:
                    continue
            else:
                continue
        # merge with params (config)
        params_obj = read_params_json(params) or {}
        # flatten params (if nested under 'config')
        config = params_obj.get('config', params_obj) if isinstance(params_obj, dict) else {}
        # some progress.csv values are strings; keep as-is
        merged = dict(last)
        # attach trial config fields (prefix with config.)
        for k, v in config.items():
            merged_key = f"config.{k}" if not str(k).startswith('config.') else str(k)
            # only add if not present
            if merged_key not in merged:
                merged[merged_key] = v
        # add trial folder name
        merged['trial_folder'] = str(child.name)
        rows.append(merged)

    if not rows:
        print('No trial progress found to aggregate.')
        return 1

    df = pd.DataFrame(rows)
    out_path = exp_dir / 'all_results.csv'
    df.to_csv(out_path, index=False)
    print(f'Wrote consolidated results to {out_path}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
