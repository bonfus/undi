import numpy as np
import pandas as pd
from ase import neighborlist
from importlib_resources import files

from undi import kubo_toyabe as isotopedata

file = files(isotopedata) / "isotopedata.txt"

info = pd.read_table(
    file,
    comment="%",
    sep="\s+",
    names=[
        "Z",
        "A",
        "Stable",
        "Symbol",
        "Element",
        "Spin",
        "G_factor",
        "Abundance",
        "Quadrupole",
    ],
)

munhbar = 7.622593285e6 * 2 * np.pi  # mu_N/hbar, SI
# (2/3)(μ_0/4pi)^2 (planck2pi 2pi × 135.5 MHz/T )^2 = 5.374 021 39 × 10^(−65) kg²·m^(6)·A^(−2)·s^(−4)
factor = 5.37402139e-5  # angstrom instead of m


def get_isotopes(Z):
    return info[info.Z == Z][["Abundance", "Spin", "G_factor"]].to_numpy()


def compute_second_moments(atms, cutoff_distances={}):
    """
    Compute second moments taking care of isotope averages
    """
    tot_H = np.count_nonzero(atms.get_atomic_numbers() == 1)

    species_avg = {}
    for e in np.unique(atms.get_atomic_numbers()):
        if e == 1:
            continue

        species_avg[e] = 0.0
        for a in get_isotopes(e):
            species_avg[e] += (a[0] / 100) * a[1] * (a[1] + 1) * (munhbar * a[2]) ** 2

    # compute second moments
    specie_contribs = {}
    for e in np.unique(atms.get_atomic_numbers()):
        if e == 1:
            continue
        sum = 0.5 * np.sum(
            neighborlist.neighbor_list(
                "d", atms, cutoff={(1, e): cutoff_distances.get(e, 40)}
            )
            ** -6
        )
        specie_contribs[e] = species_avg[e] * sum * factor / tot_H

    return specie_contribs


def kubo_toyabe(tlist, Gmu_S2):
    """Calculates the Kubo-Toyabe polarization for the nuclear arrangement
    provided in input.

    Parameters
    ----------
    tlist : numpy.array
        List of times at which the muon polarization is observed.

    Returns
    -------
    numpy.array
        Kubo-Toyabe function, for a powder in zero field.
    """
    # this is gamma_mu times sigma^2
    return 0.333333333333 + 0.6666666666 * (1 - Gmu_S2 * np.power(tlist, 2)) * np.exp(
        -0.5 * Gmu_S2 * np.power(tlist, 2)
    )


#### end for KT
