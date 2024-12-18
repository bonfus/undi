import sys, json, argparse
from itertools import product
import numpy as np

from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list

from undi import MuonNuclearInteraction
from .isotopes import Element

from tqdm import tqdm

angtom = 1e-10

def gen_neighbouring_atomic_structure(atoms, isotopes, spins, hdim_max):

    ai,aj,d,D = neighbor_list('ijdD',atoms, 10.) # very large cutoff to get all possible interactions.
                                                 # Actual selection is done below
    try:
        EFGs = atoms.get_array('efg') # EFG should be stored in the atoms object. Done via read_magres_file() function (see below).
    except:
        EFGs = np.array([0])

    # sort
    srt = np.argsort(d)
    ai = ai[srt]
    aj = aj[srt]
    d = d[srt]
    D = D[srt]

    data = []
    muon_pos = np.array([0,0,0])
    hdim = 2 # the muon Hilbert space

    for i in range(len(D)):

        if not (ai[i] == len(atoms)-1):
            continue

        atom_symbol = atoms[aj[i]].symbol

        isotope_spin = spins[atom_symbol]
        # skip nuclei with no spin
        if isotope_spin < 0.49:
            continue

        # get specific isotope
        symb = str( isotopes.get(atom_symbol) ) + atom_symbol

        # increase Hilbert space
        hdim *= (2*isotope_spin+1)

        if hdim > hdim_max:
            break

        pos = D[i] * angtom
        print('Adding atom ', symb , ' with position', pos, ' and distance ', np.linalg.norm(pos))

        # if EFG too small, neglect it.
        if np.max(np.abs(EFGs)) < 1e-9:
            data.append({'Position': pos, 'Label': symb })
        else:
            print('EFG tensor is', EFGs)
            data.append({'Position': pos, 'Label': symb, 'EFGTensor':  EFGs})

    data.insert(0,
                    {'Position': muon_pos,
                     'Label': 'mu'},
                )

    return data, hdim

def execute_undi_analysis(
        structure: Atoms,
        B_mod: float = 0.0,
        atom_as_muon: str = 'H',
        max_hdim: int = 10000,
        convergence_check: bool = False,
        algorithm: str = 'fast',
        angular_integration_steps: int = 7
    ):
    """
        Execute UNDI (mUon Nuclear Dipolar Interaction) analysis on a given atomic structure.
        Parameters:
        structure (Atoms): The atomic structure to analyze. Can contain `efg` for EFG to be used for quadrupolar interaction.
        B_mod (float, optional): The external magnetic field magnitude. Default is 0.0.
        atom_as_muon (str, optional): The symbol of the atom to be treated as a muon. Default is 'H'.
        max_hdim (int, optional): The maximum Hilbert space dimension. Default is 1000.
        convergence_check (bool, optional): If True, perform convergence check. Default is False.
        algorithm (str, optional): The algorithm to use for the analysis. Default is 'fast'.
        angular_integration_steps (int, optional):  Number of step in theta and phi for powder average. Default is 7.
                                                    Should be converged for accurate results.
        Returns:
        list: A list of dictionaries containing the analysis results for each cluster of isotopes.
        Raises:
        RuntimeError: If the muon does not appear as the last atom in the structure.
        RuntimeError: If an appropriate Hilbert space could not be found.
        Notes:
        - The function performs safety checks, collects relevant isotopes, and computes results for each cluster.
        - It defines longitudinal field (LF) and transverse field (TF) directions and computes signals for each direction.
        - If convergence_check is True, it performs an average over multiple iterations.
        - The function also computes powder averages for the signals.
        - Results are saved to 'results.json' if convergence_check is True.
        """

    # This is just to set a limit to what we consider zero-field.
    B_mod_earth = 31e-6 # T

    # safety check
    if structure[-1].symbol.lower() != atom_as_muon.lower():
        raise RuntimeError("Muon should appear as last atom")

    elements = np.unique(structure.get_chemical_symbols()[:-1])

    # Collect relevant isotopes (i.e. abundance > 5%)
    info = {}
    for element in elements:
        isotopes = Element(element).isotopes
        info[element] = [ i for i in isotopes if i.abundance > 5.]

    # Time interval
    t = np.linspace(0, 20e-6, 200)

    # Compute results for each cluster
    results = []
    for cluster in product(*info.values()):
        cluster_spins = {}
        cluster_isotopes = {}

        # stores probability for this cluster
        abd_prob = 1.


        # for isotope in cluster
        for i in cluster:
            print(i.info)
            e = i.symbol

            abd_prob *= (i.abundance/100.)

            cluster_spins[e] = i.spin
            cluster_isotopes[e] = i.mass_number


        undi_input, hdim = gen_neighbouring_atomic_structure(structure, cluster_isotopes, cluster_spins, max_hdim)
        if hdim == 2:
            raise RuntimeError("Could not find appropriate Hilbert space! Either too many spins or none found!")

        print(cluster_isotopes)

        # Definition of LF and TF
        B_lf = B_mod * np.array([0.,0.,1.])
        B_tf = B_mod * np.array([1.,0.,0.])

        if convergence_check:
            # just return the z signal to be analyzed.
            print("Doing sample along Z, LF, convergence check")
            signal_z_lf = np.zeros_like(t)
            for iteration in range(1,6):
                # we do an average
                NS = MuonNuclearInteraction(undi_input,external_field=B_lf)
                #NS.translate_rotate_sample_vec(direction) # translate_rotate_sample_by_vec
                signal_z_lf += NS.celio_on_steroids(t,  k=4, algorithm=algorithm)
                del NS
            signal_z_lf /=6

            results.append(
                {
                    "cluster_isotopes": cluster_isotopes,
                    "spins": cluster_spins,
                    "B_ext":B_mod,
                    "t": t.tolist(),
                    #"signal_x_lf": signal_x_lf.tolist(),
                    #"signal_y_lf": signal_y_lf.tolist(),
                    "signal_z_lf": signal_z_lf.tolist(),
                    #"signal_x_tf": signal_x_tf.tolist(),
                    #"signal_y_tf": signal_y_tf.tolist(),
                    #"signal_z_tf": signal_z_tf.tolist(),
                    "probability" : abd_prob,
                    #"signal_powder_tf": powder_signal_tf.tolist(),
                    #"signal_powder_lf": powder_signal_lf.tolist()
                }
            )

            return results

        # Single crystal signals

        # Written explicitely, to make it more clear.

        ## signal z
        print("Doing sample along Z, LF")
        direction = np.array([0.,0.,1.])
        ### LF
        NS = MuonNuclearInteraction(undi_input,external_field=B_lf)
        NS.translate_rotate_sample_vec(direction) # translate_rotate_sample_by_vec
        signal_z_lf = NS.celio_on_steroids(t,  k=2, algorithm=algorithm)
        del NS

        ## signal x
        direction = np.array([1.,0.,0.])
        ### LF
        print("Doing sample along X, LF")
        NS = MuonNuclearInteraction(undi_input,external_field=B_lf)
        NS.translate_rotate_sample_vec(direction) # translate_rotate_sample_by_vec
        signal_x_lf = NS.celio_on_steroids(t,  k=2, algorithm=algorithm)
        del NS

        ## signal y
        direction = np.array([0.,1.,0.])
        ### LF
        print("Doing sample along Y, LF")
        NS = MuonNuclearInteraction(undi_input,external_field=B_lf)
        NS.translate_rotate_sample_vec(direction) # translate_rotate_sample_by_vec
        signal_y_lf = NS.celio_on_steroids(t,  k=2, algorithm=algorithm)
        del NS

        if B_mod > B_mod_earth:
            ### TF ###
            # Z
            direction = np.array([0.,0.,1.])
            print("Doing sample along Z, TF")
            NS = MuonNuclearInteraction(undi_input,external_field=B_tf)
            NS.translate_rotate_sample_vec(direction) # translate_rotate_sample_by_vec
            signal_z_tf = NS.celio_on_steroids(t,  k=2, algorithm=algorithm)
            del NS

            # X
            direction = np.array([1.,0.,0.])
            print("Doing sample along X, TF")
            NS = MuonNuclearInteraction(undi_input,external_field=B_tf)
            NS.translate_rotate_sample_vec(direction) # translate_rotate_sample_by_vec
            signal_x_tf = NS.celio_on_steroids(t,  k=2, algorithm=algorithm)
            del NS

            # Y
            direction = np.array([0.,1.,0.])
            print("Doing sample along Y, TF")
            NS = MuonNuclearInteraction(undi_input,external_field=B_tf)
            NS.translate_rotate_sample_vec(direction) # translate_rotate_sample_by_vec
            signal_y_tf = NS.celio_on_steroids(t,  k=2, algorithm=algorithm)
            del NS
        else:
            signal_z_tf = signal_z_lf
            signal_x_tf = signal_x_lf
            signal_y_tf = signal_y_lf

        # powder averages: if ZF, just isotropic spatial average.
        # if LF or TF, we randomly rotate the sample 1000 times and we average.
        print("Doing sample powder average")


        powder_signal_lf = powder_signal_tf = np.zeros_like(t)
        if B_mod < B_mod_earth:
            normalisation_factor = 3
            powder_signal_lf = signal_z_lf+signal_y_lf+signal_x_lf
            powder_signal_tf = signal_z_tf+signal_y_tf+signal_x_tf
        else:
            n = angular_integration_steps # should the at least 7

            # Powder avg
            d_theta = np.pi / n
            d_phi = np.pi / n
            N_theta = np.pi / d_theta
            N_phi = 2 * np.pi / d_phi
            normalisation_factor = N_phi * np.sin(N_theta*d_theta/2) * \
                                    np.sin((N_theta-1)*d_theta/2) / np.sin(d_theta/2)

            pbartheta = tqdm(np.arange(d_theta, np.pi, d_theta))
            pbartheta.set_description('θ: ')
            for theta in pbartheta:
                pbarphi = tqdm(np.arange(0, 2*np.pi, d_phi), leave=False)
                pbarphi.set_description('φ: ')
                for phi in pbarphi:
                    direction = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

                    NS = MuonNuclearInteraction(undi_input,external_field=B_lf)
                    NS.translate_rotate_sample_vec(direction)
                    powder_signal_lf += np.sin(theta) * NS.celio_on_steroids(t,  k=2, algorithm=algorithm, progress = False)
                    del NS

                    NS = MuonNuclearInteraction(undi_input,external_field=B_tf)
                    NS.translate_rotate_sample_vec(direction)
                    powder_signal_tf += np.sin(theta) * NS.celio_on_steroids(t,  k=2, algorithm=algorithm, progress = False)
                    del NS


        powder_signal_lf /= normalisation_factor
        powder_signal_tf /= normalisation_factor

        results.append(
            {
                "cluster_isotopes": cluster_isotopes,
                "spins": cluster_spins,
                "B_ext":B_mod,
                "t": t.tolist(),
                "signal_x_lf": signal_x_lf.tolist(),
                "signal_y_lf": signal_y_lf.tolist(),
                "signal_z_lf": signal_z_lf.tolist(),
                "signal_x_tf": signal_x_tf.tolist(),
                "signal_y_tf": signal_y_tf.tolist(),
                "signal_z_tf": signal_z_tf.tolist(),
                "probability" : abd_prob,
                "signal_powder_tf": powder_signal_tf.tolist(),
                "signal_powder_lf": powder_signal_lf.tolist()
            }
        )

    return results
