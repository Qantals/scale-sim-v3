#!/usr/bin/env python3
"""
Plot power-over-time from interval energy data.

Reads ``interval_energy.csv`` (produced by ``create_action_count_interval.py``)
and generates:

  1. Total power vs. time (line plot)
  2. Stacked area chart – component breakdown over time

Usage:
    python3 plot_power_timeline.py --input <interval_energy.csv>
                                   [--output <plot_dir>]
                                   [--freq_mhz 1000]
"""

import argparse
import csv
import os
import sys
import matplotlib
matplotlib.use('Agg')  # headless-safe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def load_data(csv_path):
    """Load interval_energy.csv, return lists of dicts and column names."""
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Find energy columns (end with _energy_pJ, excluding total_energy_pJ)
    energy_cols = [k for k in rows[0].keys()
                   if k.endswith('_energy_pJ') and k != 'total_energy_pJ']

    return rows, energy_cols


def plot_power_timeline(csv_path, output_dir, freq_mhz=1000.0):
    """Generate power-over-time plots."""
    rows, energy_cols = load_data(csv_path)

    n = len(rows)
    if n == 0:
        print('ERROR: no data in CSV', file=sys.stderr)
        return

    # Extract data
    start_cycles = np.array([int(r['start_cycle']) for r in rows], dtype=float)
    end_cycles = np.array([int(r['end_cycle']) for r in rows], dtype=float)
    durations = end_cycles - start_cycles  # cycles

    # Midpoint for plotting
    mid_cycles = (start_cycles + end_cycles) / 2.0

    # Convert to time (seconds): cycles / frequency
    period_ns = 1e9 / (freq_mhz * 1e6)  # nanoseconds per cycle
    time_us = mid_cycles * period_ns / 1e3  # microseconds

    # Energy per interval (pJ)
    total_energy_pJ = np.array([float(r['total_energy_pJ']) for r in rows])

    # Power (W) = energy (pJ) / time (ns) / 1000 = energy_pJ / (duration * period_ns) / 1000
    # 1 mW = 1 pJ/ns, so W = mW / 1000
    power_W = total_energy_pJ / (durations * period_ns) / 1000

    # Component energies
    comp_energy = {}
    for col in energy_cols:
        comp_energy[col] = np.array([float(r[col]) for r in rows])

    # Component power (W)
    comp_power = {}
    for col in energy_cols:
        comp_power[col] = comp_energy[col] / (durations * period_ns) / 1000

    # Pretty labels
    label_map = {
        'ifmap_dram_energy_pJ': 'IFMAP DRAM',
        'weights_dram_energy_pJ': 'Weights DRAM',
        'psum_dram_energy_pJ': 'Psum DRAM',
        'ifmap_glb_energy_pJ': 'IFMAP GLB',
        'weights_glb_energy_pJ': 'Weights GLB',
        'psum_glb_energy_pJ': 'Psum GLB',
        'ifmap_spad_energy_pJ': 'IFMAP Spad',
        'weights_spad_energy_pJ': 'Weights Spad',
        'psum_spad_energy_pJ': 'Psum Spad',
        'mac_energy_pJ': 'MAC',
    }

    os.makedirs(output_dir, exist_ok=True)

    # Colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(energy_cols)))

    # --- Plot 1: Total power vs time ---
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(time_us, power_W, 'b-', linewidth=1.0, marker='o', markersize=3)
    ax.fill_between(time_us, 0, power_W, alpha=0.15, color='blue')
    ax.set_xlabel('Time (µs)', fontsize=12)
    ax.set_ylabel('Power (W)', fontsize=12)
    ax.set_title('Total Power vs. Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'power_total.png'), dpi=150)
    plt.close(fig)
    print(f'Saved {output_dir}/power_total.png')

    # --- Plot 2: Stacked area chart ---
    fig, ax = plt.subplots(figsize=(14, 6))

    # Aggregate into major categories
    dram_cols = [c for c in energy_cols if 'dram' in c]
    glb_cols = [c for c in energy_cols if 'glb' in c]
    spad_cols = [c for c in energy_cols if 'spad' in c]
    mac_cols = [c for c in energy_cols if 'mac' in c]

    categories = [
        ('DRAM', dram_cols, 'tab:red'),
        ('GLB (SRAM)', glb_cols, 'tab:blue'),
        ('PE Spads', spad_cols, 'tab:green'),
        ('MAC', mac_cols, 'tab:orange'),
    ]

    # Stack from bottom up
    y_stack = np.zeros(n)
    for cat_name, cols, color in categories:
        cat_power = np.zeros(n)
        for c in cols:
            cat_power += comp_power[c]
        ax.fill_between(time_us, y_stack, y_stack + cat_power,
                        alpha=0.7, color=color, label=cat_name,
                        step='post')
        y_stack += cat_power

    ax.set_xlabel('Time (µs)', fontsize=12)
    ax.set_ylabel('Power (W)', fontsize=12)
    ax.set_title('Power Breakdown Over Time', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'power_breakdown.png'), dpi=150)
    plt.close(fig)
    print(f'Saved {output_dir}/power_breakdown.png')

    # --- Plot 3: Individual components ---
    fig, axes = plt.subplots(len(energy_cols), 1,
                              figsize=(14, 2.5 * len(energy_cols)),
                              sharex=True)
    if len(energy_cols) == 1:
        axes = [axes]

    for idx, col in enumerate(energy_cols):
        ax = axes[idx]
        ax.plot(time_us, comp_power[col], linewidth=1.0, marker='o',
                markersize=2, color=colors[idx])
        ax.fill_between(time_us, 0, comp_power[col], alpha=0.2,
                        color=colors[idx])
        ax.set_ylabel(label_map.get(col, col), fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (µs)', fontsize=12)
    fig.suptitle('Per-Component Power', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'power_components.png'), dpi=150)
    plt.close(fig)
    print(f'Saved {output_dir}/power_components.png')

    # Print summary
    avg_power = np.mean(power_W)
    peak_power = np.max(power_W)
    total_energy = np.sum(total_energy_pJ)
    print(f'\nSummary:')
    print(f'  Intervals:        {n}')
    print(f'  Avg power:        {avg_power:.4f} W')
    print(f'  Peak power:       {peak_power:.4f} W')
    print(f'  Total energy:     {total_energy:.2f} pJ')
    print(f'  Frequency:        {freq_mhz} MHz')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot power-over-time from interval energy data')
    parser.add_argument('--input', required=True,
                        help='Path to interval_energy.csv')
    parser.add_argument('--output', default='./plots',
                        help='Output directory for plots (default: ./plots)')
    parser.add_argument('--freq_mhz', type=float, default=1000.0,
                        help='Clock frequency in MHz (default: 1000)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'ERROR: input file not found: {args.input}', file=sys.stderr)
        sys.exit(1)

    plot_power_timeline(args.input, args.output, args.freq_mhz)
