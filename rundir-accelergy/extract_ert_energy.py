#!/usr/bin/env python3
"""
Extract per-operation energy costs from an Accelergy ERT.yaml file.

Produces a JSON lookup table keyed by (component_short_name, action_name,
address_delta, data_delta) -> energy (pJ).  The short name strips the
`systolic_array.` prefix and PE range notation, e.g.:

    ("ifmap_glb", "read", 1, 1)  ->  29.4639
    ("weights_dram", "read", None, None)  ->  256.0
    ("PE.mac", "mac_random", None, None)  ->  12.89
"""

import argparse
import json
import os
import sys
import yaml


def make_key(component_name: str, action_name: str,
             address_delta, data_delta):
    """Build a canonical lookup key."""
    return (component_name, action_name, address_delta, data_delta)


def ert_to_lookup(ert_path: str):
    """Parse ERT.yaml and return dict[lookup_key] = energy_pJ."""
    with open(ert_path, 'r') as f:
        ert = yaml.safe_load(f)

    lookup = {}

    for table in ert['ERT']['tables']:
        full_name = table['name']  # e.g. "systolic_array.ifmap_glb"

        # Normalise component name: strip prefix, collapse PE ranges
        short = full_name
        if short.startswith('systolic_array.'):
            short = short[len('systolic_array.'):]

        # PE range -> generic  "PE.ifmap_spad"
        if '.PE[' in short or short.startswith('PE['):
            # e.g. "PE[0..1023].ifmap_spad" -> "PE.ifmap_spad"
            close_bracket = short.index(']')
            dot_after = short.index('.', close_bracket)
            short = 'PE' + short[dot_after:]   # "PE.ifmap_spad"

        for action in table['actions']:
            act_name = action['name']
            args = action.get('arguments', {})
            addr_delta = args.get('address_delta', None)
            data_delta = args.get('data_delta', None)
            energy = float(action['energy'])

            key = make_key(short, act_name, addr_delta, data_delta)
            lookup[key] = energy

    return lookup


def lookup_energy(lookup, component_short, action_name,
                  address_delta=None, data_delta=None):
    """Convenience: look up energy from the dict, with sensible error."""
    key = make_key(component_short, action_name, address_delta, data_delta)
    if key not in lookup:
        # Try without deltas (DRAM-style)
        key_no_args = make_key(component_short, action_name, None, None)
        if key_no_args in lookup:
            return lookup[key_no_args]
        raise KeyError(f"Energy not found for {key}")
    return lookup[key]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract per-op energy from Accelergy ERT.yaml')
    parser.add_argument('--ert', required=True,
                        help='Path to ERT.yaml')
    parser.add_argument('--output', default='ert_lookup.json',
                        help='Output JSON file (default: ert_lookup.json)')
    args = parser.parse_args()

    if not os.path.exists(args.ert):
        print(f'ERROR: ERT file not found: {args.ert}', file=sys.stderr)
        sys.exit(1)

    lookup = ert_to_lookup(args.ert)

    # JSON requires string keys; convert tuple -> "comp|action|ad|dd"
    json_lookup = {}
    for (comp, act, ad, dd), energy in lookup.items():
        k = f"{comp}|{act}|{ad}|{dd}"
        json_lookup[k] = energy

    with open(args.output, 'w') as f:
        json.dump(json_lookup, f, indent=2)

    print(f'Wrote {len(lookup)} energy entries to {args.output}')
