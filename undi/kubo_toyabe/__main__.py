"""Command line interface to compute the Kubo-Toyabe polarization of a structure.

    python -m undi.kubo_toyabe structure.xyz

The structure is read with ASE and must contain the muon as an H atom. The second moment of the field distribution is
computed from the nuclear environment of the muon (isotope averages, see `undi.auto_analysis.compute_second_moments`)
and the Kubo-Toyabe function is evaluated on a time grid. The result is written to ``KT.json`` (by default)::

    {"t": [...],   # time, in microseconds
     "KT": [...]}  # polarization
"""
import argparse
import json

import numpy as np
from ase.io import read

from undi.auto_analysis import compute_second_moments
from undi.kubo_toyabe.KT import kubo_toyabe


def compute_kubo_toyabe(atoms, tmax=20e-6, npoints=1000):
    """Kubo-Toyabe polarization of the muon (the H atoms) in `atoms`.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure, with the muon as an H atom.
    tmax : float
        Last time of the grid, in seconds. The grid goes from 0 to `tmax`.
    npoints : int
        Number of points of the time grid.

    Returns
    -------
    dict
        ``t`` (microseconds) and ``KT``, as lists.
    """
    t = np.linspace(0, tmax, npoints)  # seconds
    second_moments = compute_second_moments(atoms)
    polarization = kubo_toyabe(t, np.sum(list(second_moments.values())))
    return {
        "t": (t * 1e6).tolist(),  # microseconds
        "KT": np.asarray(polarization).tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Kubo-Toyabe polarization of the muon (the H atom) in a structure."
    )
    parser.add_argument("structure", type=str, help="Structure to be parsed (any format ASE can read)")
    parser.add_argument("--tmax", type=float, default=20e-6, help="Last time, in seconds. Default is 20e-6.")
    parser.add_argument("--npoints", type=int, default=1000, help="Number of points of the time grid. Default is 1000.")
    parser.add_argument("--output", type=str, default="KT.json", help="File to write the result to. Default is KT.json.")
    args = parser.parse_args(argv)

    results = compute_kubo_toyabe(read(args.structure), tmax=args.tmax, npoints=args.npoints)
    with open(args.output, "w") as f:
        json.dump(results, f)
    return results


if __name__ == "__main__":
    main()
