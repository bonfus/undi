"""
Kubo-Toyabe second moments from the nuclear environment of the muon.
"""
import numpy as np

munhbar = 7.622593285e6 * 2 * np.pi  # mu_N/hbar, SI

def zero_field_distribution_powder(atoms):
    """Calculates gamma_mu ^2 DeltaG ^2, where gamma_mu is the
        gyromagnetic ratio of the muon and Delta^2 is the variance of
        a Gaussian field distribution, using the secular approximation
        for the dipolar interaction. Powder averaging and zero external
        field are intended.

    Returns
    -------
    float
        gamma_mu ^2 DeltaG ^2. See above.
    """


    plank2pi = 1.0545718E-34 #joule second
    mu_0 = 0.0000012566371 # (kilogram meter) ∕ (ampere^2 × second^2)
    r = 0.
    gamma_mu = 0.
    pos_mu = None
    for atom in atoms:
        if atom['Label'] == 'mu':
            gamma_mu = atom['Gamma']
            pos_mu   = atom['Position']
            break
    else:
        raise RuntimeWarning('Multiple muons?! Only using last one in list')


    for atom in atoms:
        if atom['Label'] == 'mu':
            continue
        I = atom['Spin']
        gamma = atom['Gamma']
        r3 = np.linalg.norm(atom['Position'] - pos_mu)**3

        r += 2. * (mu_0/(4*np.pi))**2    * \
            ((gamma * plank2pi) / r3)**2 * \
            0.33333333333 * (I * (I+1))

    return r * (gamma_mu**2)

def kubo_toyabe(tlist, Gmu_S2):
    """Calculates the Kubo-Toyabe polarization for the nuclear arrangement
    provided in input.

    Parameters
    ----------
    tlist : numpy.array
        List of times at which the muon polarization is observed.
    Gmu_S2 : float
        Gamma_mu squared times sigma squared (second moment).

    Returns
    -------
    numpy.array
        Kubo-Toyabe function, for a powder in zero field.
    """
    # this is gamma_mu times sigma^2
    return 0.333333333333 + 0.6666666666 * (1 - Gmu_S2 * np.power(tlist, 2)) * np.exp(
        -0.5 * Gmu_S2 * np.power(tlist, 2)
    )
