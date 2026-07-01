#!/usr/bin/env python3
"""
Interval-based energy estimation for transient power analysis.

Reads per-cycle trace CSVs from a Scale-Sim run, bins cycles into intervals,
counts operation *activity* per interval, then distributes the known total
energy (from the original ``energy_estimation.yaml``) proportionally by
activity fraction.  This guarantees that the sum of interval energies exactly
matches the original total energy.

Outputs:
    interval_energy.csv      – energy (pJ) per interval, per component
    interval_activity.csv    – raw activity counts per interval
"""

import argparse
import configparser as cp
import csv
import os
import sys
import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_trace_csv(filepath):
    """Read a trace CSV, return ndarray (col 0 = cycle, rest = addresses)."""
    try:
        df = pd.read_csv(filepath, header=None, na_values=[''])
    except pd.errors.EmptyDataError:
        return np.zeros((0, 1))
    arr = df.to_numpy(dtype=float)
    if arr.ndim == 2 and arr.shape[1] > 1:
        valid = ~np.all(np.isnan(arr[:, 1:]), axis=1)
        arr = arr[valid]
    return arr


def count_active_in_window(trace_arr, win_start, win_end):
    """Count non-(-1) values in rows whose cycle is in [win_start, win_end).

    Returns (active_count, n_rows_in_window).
    """
    if trace_arr.size == 0:
        return 0, 0
    cycles = trace_arr[:, 0]
    addrs = trace_arr[:, 1:]
    mask = (cycles >= win_start) & (cycles < win_end)
    if not np.any(mask):
        return 0, 0
    addrs_win = addrs[mask]
    n_rows = addrs_win.shape[0]
    active = 0
    for r in range(n_rows):
        for c in range(addrs_win.shape[1]):
            v = addrs_win[r, c]
            if not np.isnan(v) and v != -1:
                active += 1
    return active, n_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_intervals(saved_folder, run_name, config_path, interval_size):
    """Run interval-based analysis."""

    # --- Config ------------------------------------------------------------
    cfg = cp.ConfigParser()
    cfg.read(config_path)
    array_h = int(cfg.get('architecture_presets', 'ArrayHeight'))
    array_w = int(cfg.get('architecture_presets', 'ArrayWidth'))
    pe_array_size = array_h * array_w

    # --- Find directories --------------------------------------------------
    run_dir = os.path.join(saved_folder, run_name)
    if not os.path.isdir(run_dir):
        run_dir = os.path.join(os.pardir, saved_folder, run_name)
    if not os.path.isdir(run_dir):
        print(f'ERROR: run dir not found: {run_dir}', file=sys.stderr)
        sys.exit(1)

    layer_dirs = sorted([d for d in os.listdir(run_dir)
                         if d.startswith('layer')
                         and os.path.isdir(os.path.join(run_dir, d))])
    print(f'Found {len(layer_dirs)} layer(s): {layer_dirs}')

    # --- Read original energy totals ---------------------------------------
    out_parent = os.path.dirname(os.path.abspath(run_dir))
    base_run = run_name.replace('scale_sim_output_', '')
    energy_path = os.path.join(out_parent,
                               f'accelergy_output_{base_run}',
                               'energy_estimation.yaml')
    if not os.path.exists(energy_path):
        # Try relative to cwd
        alt = os.path.join('output', f'accelergy_output_{base_run}',
                           'energy_estimation.yaml')
        if os.path.exists(alt):
            energy_path = alt

    if not os.path.exists(energy_path):
        print(f'ERROR: energy_estimation.yaml not found (tried {energy_path})',
              file=sys.stderr)
        sys.exit(1)

    with open(energy_path, 'r') as f:
        ee = yaml.safe_load(f)
    total_energy_by_comp = {}
    for c in ee['energy_estimation']['components']:
        total_energy_by_comp[c['name']] = float(c['energy'])
    print(f'Loaded {len(total_energy_by_comp)} component energies '
          f'from {energy_path}')

    # --- Read DETAILED_ACCESS_REPORT for windows ---------------------------
    detail_path = os.path.join(run_dir, 'DETAILED_ACCESS_REPORT.csv')
    detail_df = pd.read_csv(detail_path, sep=r'\s*,\s*', engine='python')
    detail = detail_df.to_numpy()

    # --- Determine global cycle range from DRAM traces ---------------------
    global_min = float('inf')
    global_max = float('-inf')
    for ld in layer_dirs:
        lp = os.path.join(run_dir, ld)
        for fname in ['IFMAP_DRAM_TRACE.csv', 'FILTER_DRAM_TRACE.csv',
                       'OFMAP_DRAM_TRACE.csv']:
            fp = os.path.join(lp, fname)
            if os.path.exists(fp):
                arr = read_trace_csv(fp)
                if arr.size > 0:
                    global_min = min(global_min, np.nanmin(arr[:, 0]))
                    global_max = max(global_max, np.nanmax(arr[:, 0]))
    global_min = int(global_min)
    global_max = int(global_max)
    num_intervals = int(np.ceil((global_max - global_min + 1) / interval_size))
    print(f'Cycle range: {global_min}..{global_max}, '
          f'{num_intervals} intervals of {interval_size}')

    # --- Activity tracking -------------------------------------------------
    activity_keys = [
        'ifmap_dram', 'weights_dram', 'psum_dram',
        'ifmap_glb', 'weights_glb', 'psum_glb',
        'PE_ifmap_spad', 'PE_weights_spad', 'PE_psum_spad',
        'PE_mac',
    ]

    # Per-interval accumulators
    interval_activity = [dict.fromkeys(activity_keys, 0)
                         for _ in range(num_intervals)]

    # --- Process layers ----------------------------------------------------
    for ld_idx, ld in enumerate(layer_dirs):
        lp = os.path.join(run_dir, ld)
        print(f'Processing {ld} ...')

        # Load traces
        ifmap_sram = read_trace_csv(os.path.join(lp, 'IFMAP_SRAM_TRACE.csv'))
        filter_sram = read_trace_csv(os.path.join(lp, 'FILTER_SRAM_TRACE.csv'))
        ofmap_sram = read_trace_csv(os.path.join(lp, 'OFMAP_SRAM_TRACE.csv'))
        ifmap_dram = read_trace_csv(os.path.join(lp, 'IFMAP_DRAM_TRACE.csv'))
        filter_dram = read_trace_csv(os.path.join(lp, 'FILTER_DRAM_TRACE.csv'))
        ofmap_dram = read_trace_csv(os.path.join(lp, 'OFMAP_DRAM_TRACE.csv'))

        # Windows from DETAILED_ACCESS_REPORT
        # Map our activity keys to the window dictionary keys
        row = detail[ld_idx]
        win = {
            'ifmap_sram':  (float(row[1]),  float(row[2])),
            'filter_sram': (float(row[4]),  float(row[5])),
            'ofmap_sram':  (float(row[7]),  float(row[8])),
            'ifmap_dram':  (float(row[10]), float(row[11])),
            'filter_dram': (float(row[13]), float(row[14])),
            'ofmap_dram':  (float(row[16]), float(row[17])),
        }
        # Activity key -> window key
        key_to_win = {
            'ifmap_glb': 'ifmap_sram', 'ifmap_dram': 'ifmap_dram',
            'weights_glb': 'filter_sram', 'weights_dram': 'filter_dram',
            'psum_glb': 'ofmap_sram', 'psum_dram': 'ofmap_dram',
            'PE_ifmap_spad': 'ifmap_sram',
            'PE_weights_spad': 'filter_sram',
            'PE_psum_spad': 'ofmap_sram',
        }

        for iid in range(num_intervals):
            t0 = global_min + iid * interval_size
            t1 = min(global_min + (iid + 1) * interval_size, global_max + 1)
            ia = interval_activity[iid]

            # --- SRAM activity (gated by SRAM windows) ---
            for key, trace in [
                ('ifmap_glb', ifmap_sram),
                ('weights_glb', filter_sram),
                ('psum_glb', ofmap_sram),
            ]:
                ws, we = win[key_to_win[key]]
                eff0, eff1 = max(t0, ws), min(t1, we + 1)
                if eff0 < eff1:
                    act, _ = count_active_in_window(trace, eff0, eff1)
                    ia[key] += act

            # --- DRAM activity ---
            for key, trace in [
                ('ifmap_dram', ifmap_dram),
                ('weights_dram', filter_dram),
                ('psum_dram', ofmap_dram),
            ]:
                ws, we = win[key_to_win[key]]
                eff0, eff1 = max(t0, ws), min(t1, we + 1)
                if eff0 < eff1:
                    act, _ = count_active_in_window(trace, eff0, eff1)
                    ia[key] += act

            # --- PE spad activity (from SRAM traces, same as GLB reads) ---
            for spad_key, trace in [
                ('PE_ifmap_spad', ifmap_sram),
                ('PE_weights_spad', filter_sram),
                ('PE_psum_spad', ofmap_sram),
            ]:
                ws, we = win[key_to_win[spad_key]]
                eff0, eff1 = max(t0, ws), min(t1, we + 1)
                if eff0 < eff1:
                    act, _ = count_active_in_window(trace, eff0, eff1)
                    ia[spad_key] += act

            # --- MAC activity: cycles × PE_array_size (matches original) ---
            mac0 = max(t0, win['ifmap_sram'][0], win['filter_sram'][0],
                       win['ofmap_sram'][0])
            mac1 = min(t1, win['ifmap_sram'][1] + 1,
                       win['filter_sram'][1] + 1, win['ofmap_sram'][1] + 1)
            if mac0 < mac1 and ifmap_sram.size > 0:
                cyc = ifmap_sram[:, 0]
                n_cyc = int(np.sum((cyc >= mac0) & (cyc < mac1)))
                ia['PE_mac'] += n_cyc * pe_array_size

    # --- Sum total activity across all intervals ---------------------------
    total_activity = dict.fromkeys(activity_keys, 0)
    for ia in interval_activity:
        for k in activity_keys:
            total_activity[k] += ia[k]

    # --- Aggregate PE spad & MAC energy from per-PE values -----------------
    pe_energy_totals = {}
    for name, energy in total_energy_by_comp.items():
        for suffix in ['ifmap_spad', 'weights_spad', 'psum_spad', '.mac']:
            if suffix in name:
                key = 'PE_' + suffix.lstrip('.')
                pe_energy_totals[key] = pe_energy_totals.get(key, 0.0) + energy

    # Map activity keys to original component names for GLB/DRAM
    activity_to_comp = {
        'ifmap_dram': 'systolic_array.ifmap_dram',
        'weights_dram': 'systolic_array.weights_dram',
        'psum_dram': 'systolic_array.psum_dram',
        'ifmap_glb': 'systolic_array.ifmap_glb',
        'weights_glb': 'systolic_array.weights_glb',
        'psum_glb': 'systolic_array.psum_glb',
    }

    # --- Distribute energy proportionally ----------------------------------
    print('Computing interval energies ...')
    interval_energy_rows = []

    for iid in range(num_intervals):
        t0 = global_min + iid * interval_size
        t1 = min(global_min + (iid + 1) * interval_size, global_max + 1)

        row = {
            'interval_id': iid,
            'start_cycle': int(t0),
            'end_cycle': int(t1),
            'duration_cycles': int(t1 - t0),
        }
        total_e = 0.0

        for key in activity_keys:
            ia = interval_activity[iid][key]
            ta = total_activity[key]

            if key in activity_to_comp:
                total_e_comp = total_energy_by_comp.get(
                    activity_to_comp[key], 0.0)
            else:
                total_e_comp = pe_energy_totals.get(key, 0.0)

            e = total_e_comp * (ia / ta) if ta > 0 else 0.0
            col = key.replace('PE_', '') + '_energy_pJ'
            row[col] = e
            total_e += e

        row['total_energy_pJ'] = total_e
        interval_energy_rows.append(row)

    # --- Write outputs -----------------------------------------------------
    act_path = os.path.join(run_dir, 'interval_activity.csv')
    with open(act_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['interval_id', 'start_cycle', 'end_cycle']
                         + activity_keys)
        for iid in range(num_intervals):
            t0 = global_min + iid * interval_size
            t1 = min(global_min + (iid + 1) * interval_size, global_max + 1)
            writer.writerow(
                [iid, int(t0), int(t1)]
                + [interval_activity[iid][k] for k in activity_keys])
    print(f'Wrote {act_path}')

    energy_out = os.path.join(run_dir, 'interval_energy.csv')
    with open(energy_out, 'w', newline='') as f:
        writer = csv.writer(f)
        header = list(interval_energy_rows[0].keys())
        writer.writerow(header)
        for row in interval_energy_rows:
            writer.writerow([row[k] for k in header])
    print(f'Wrote {energy_out}')

    # --- Verification ------------------------------------------------------
    sum_interval = sum(r['total_energy_pJ'] for r in interval_energy_rows)
    orig_total = sum(total_energy_by_comp.values())
    diff_pct = abs(sum_interval - orig_total) / orig_total * 100
    print(f'\n=== Verification ===')
    print(f'Original total:  {orig_total:>15.2f} pJ')
    print(f'Interval sum:    {sum_interval:>15.2f} pJ')
    print(f'Difference:      {diff_pct:.4f}%')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interval-based energy estimation from Scale-Sim traces')
    parser.add_argument('--saved_folder', required=True,
                        help='Path to Scale-Sim output dir')
    parser.add_argument('--run_name', required=True,
                        help='Name of the Scale-Sim run')
    parser.add_argument('--config', required=True,
                        help='Path to scale.cfg')
    parser.add_argument('--interval_size', type=int, default=1000,
                        help='Cycles per interval (default: 1000)')
    args = parser.parse_args()
    process_intervals(args.saved_folder, args.run_name, args.config,
                      args.interval_size)
