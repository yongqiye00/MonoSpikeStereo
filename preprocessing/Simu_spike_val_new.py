"""Thin wrapper for the Tartanair spike-simulation preprocessing pipeline.

This module preserves the historical script path while moving the actual
implementation into `preprocessing/pipelines/tartanair_simulation.py`.

Keeping this wrapper allows existing commands such as:

    python preprocessing/Simu_spike_val_new.py --help

to continue working unchanged.
"""

from pipelines.tartanair_simulation import (
    Config,
    entry,
    main,
    parse_args,
    parse_crop,
    run_simulation_LCD,
)

__all__ = [
    "Config",
    "entry",
    "main",
    "parse_args",
    "parse_crop",
    "run_simulation_LCD",
]


if __name__ == "__main__":
    main()
