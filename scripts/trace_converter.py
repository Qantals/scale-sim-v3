#!/usr/bin/env python3
"""
Convert SCALE-Sim trace files between formats.

Usage:
    # Convert sparse .npy traces back to CSV (for debugging/inspection)
    python scripts/trace_converter.py layer8/ --from sparse_npy --to csv --output layer8_csv/

    # Convert CSV traces to sparse .npy
    python scripts/trace_converter.py layer8/ --from csv --to sparse_npy --output layer8_npy/

    # Convert CSV traces to compressed .npz
    python scripts/trace_converter.py layer8/ --from csv --to npz --output layer8_npz/

Supported formats: csv, npy, npz, sparse_npy
"""

import argparse
import os
import sys
import numpy as np


def _find_trace_files(dirpath, basename):
    """Find a trace file with any supported extension."""
    for ext in ['.csv', '.npy', '.npz']:
        fp = os.path.join(dirpath, basename + ext)
        if os.path.exists(fp):
            return fp
    return None


def _read_dense(trace_arr):
    """Given a trace array, return it in dense format (cycle col + address cols with -1).

    If the input is already dense (more than 2 columns), return as-is.
    If the input is sparse (2 columns), reconstruct dense with -1 padding.
    """
    if trace_arr.shape[1] > 2:
        return trace_arr  # Already dense

    # Sparse (N, 2): reconstruct dense
    cycles = np.unique(trace_arr[:, 0])
    max_addrs_per_cycle = 1
    # Find the max number of addresses per cycle by scanning
    from collections import defaultdict
    per_cycle = defaultdict(int)
    for row in trace_arr:
        cyc = int(row[0])
        per_cycle[cyc] += 1
    if per_cycle:
        max_addrs_per_cycle = max(per_cycle.values())

    n_cycles = len(cycles)
    dense = np.ones((n_cycles, 1 + max_addrs_per_cycle), dtype=np.int64) * -1
    cycle_to_row = {}
    row_idx = 0
    for cyc in cycles:
        dense[row_idx, 0] = int(cyc)
        cycle_to_row[int(cyc)] = row_idx
        row_idx += 1

    # Fill in addresses
    col_pos = {int(c): 1 for c in cycles}
    for row in trace_arr:
        cyc = int(row[0])
        addr = int(row[1])
        r = cycle_to_row[cyc]
        c = col_pos[cyc]
        dense[r, c] = addr
        col_pos[cyc] += 1

    return dense


def convert_file(src_path, dst_path, from_fmt, to_fmt):
    """Convert a single trace file from src_path to dst_path."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # --- Read ---
    ext = os.path.splitext(src_path)[1].lower()
    if ext == '.csv':
        try:
            import pandas as pd
            df = pd.read_csv(src_path, header=None, na_values=[''])
            arr = df.to_numpy(dtype=float)
            if arr.ndim == 2 and arr.shape[1] > 1:
                valid = ~np.all(np.isnan(arr[:, 1:]), axis=1)
                arr = arr[valid]
        except Exception:
            arr = np.loadtxt(src_path, delimiter=',', dtype=float)
    elif ext == '.npz':
        data = np.load(src_path)
        if 'trace' in data:
            arr = data['trace']
        else:
            arr = data[list(data.keys())[0]]
        data.close()
    else:
        arr = np.load(src_path)

    # --- Write ---
    if to_fmt == 'csv':
        dense = _read_dense(arr)
        np.savetxt(dst_path, dense, fmt='%i', delimiter=",")
    elif to_fmt == 'npz':
        # Always store dense in npz
        dense = _read_dense(arr)
        np.savez_compressed(dst_path, trace=dense)
    elif to_fmt == 'npy':
        dense = _read_dense(arr)
        np.save(dst_path, dense)
    elif to_fmt == 'sparse_npy':
        if arr.shape[1] > 2:
            # Convert dense to sparse
            cycles = arr[:, 0]
            addrs = arr[:, 1:]
            active_mask = addrs != -1
            rows, cols = np.where(active_mask)
            sparse = np.column_stack((cycles[rows], addrs[rows, cols]))
            np.save(dst_path, sparse.astype(np.int64))
        else:
            # Already sparse
            np.save(dst_path, arr.astype(np.int64))


def main():
    parser = argparse.ArgumentParser(
        description='Convert SCALE-Sim trace files between formats')
    parser.add_argument('srcdir', help='Source directory containing trace files')
    parser.add_argument('--from', dest='from_fmt', default='sparse_npy',
                        choices=['csv', 'npy', 'npz', 'sparse_npy'],
                        help='Source format (auto-detected by extension if omitted)')
    parser.add_argument('--to', dest='to_fmt', required=True,
                        choices=['csv', 'npy', 'npz', 'sparse_npy'],
                        help='Target format')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: srcdir)')
    args = parser.parse_args()

    src_dir = args.srcdir
    out_dir = args.output if args.output else src_dir
    to_fmt = args.to_fmt

    trace_basenames = [
        'IFMAP_SRAM_TRACE', 'FILTER_SRAM_TRACE', 'OFMAP_SRAM_TRACE',
        'IFMAP_DRAM_TRACE', 'FILTER_DRAM_TRACE', 'OFMAP_DRAM_TRACE',
    ]

    ext_map = {'.csv': 'csv', '.npy': 'npy', '.npz': 'npz'}
    to_ext = {'csv': '.csv', 'npy': '.npy', 'npz': '.npz', 'sparse_npy': '.npy'}[to_fmt]

    for bn in trace_basenames:
        src = _find_trace_files(src_dir, bn)
        if src is None:
            print(f'  [SKIP] {bn}: no file found in {src_dir}')
            continue

        # Auto-detect from_fmt
        ext = os.path.splitext(src)[1].lower()
        from_fmt = ext_map.get(ext, args.from_fmt)

        dst = os.path.join(out_dir, bn + to_ext)
        print(f'  {bn}: {from_fmt} → {to_fmt}  ({src} → {dst})')
        try:
            convert_file(src, dst, from_fmt, to_fmt)
        except Exception as e:
            print(f'    ERROR: {e}', file=sys.stderr)

    print('Done!')


if __name__ == '__main__':
    main()
