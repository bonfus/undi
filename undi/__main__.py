import sys, json, argparse
from itertools import product
import numpy as np

from ase.io import read
from ase.neighborlist import neighbor_list

from undi import MuonNuclearInteraction
from .isotopes import Element

angtom = 1e-10


def gen_neighbouring_atomic_structure(atoms, isotopes, spins, hdim_max):

    ai,aj,d,D = neighbor_list('ijdD',atoms, 10.) # very large cutoff to get all possible interactions.
                                              # Actual selection is done below

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
        data.append({'Position': pos, 'Label': symb })



    data.insert(0,
                    {'Position': muon_pos,
                     'Label': 'mu'},
                )
    return data, hdim


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-hdim', type=int, default=1e5)
    parser.add_argument('--atom-as-muon', type=str, default="H")
    parser.add_argument('structure', type=str, help='Structure to be parsed')
    args = parser.parse_args()

    atms = read(args.structure)

    # safety check
    if atms[-1].symbol.lower() != args.atom_as_muon.lower():
        raise RuntimeError("Muon should appear as last atom")

    elements = np.unique(atms.get_chemical_symbols()[:-1])

    # Collect relevant isotopes (i.e. abundance > 5%)
    info = {}
    for element in elements:
        isotopes = Element(element).isotopes
        info[element] = [ i for i in isotopes if i.abundance > 5.]



    # Time interval
    t = np.linspace(0,10e-6,100)

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


        undi_input, hdim = gen_neighbouring_atomic_structure(atms, cluster_isotopes, cluster_spins, args.max_hdim)
        if hdim == 2:
            raise RuntimeError("Could not find appropriate Hilbert space! Either too many spins or none found!")

        print(cluster_isotopes)

        NS = MuonNuclearInteraction(undi_input)
        NS.translate_rotate_sample_vec(np.array([1.,0.,0.]))
        signal_x = NS.celio_on_steroids(t,  k=1, algorithm='light')
        del NS

        NS = MuonNuclearInteraction(undi_input)
        NS.translate_rotate_sample_vec(np.array([0.,1.,0.]))
        signal_y = NS.celio_on_steroids(t,  k=1, algorithm='light')
        del NS

        NS = MuonNuclearInteraction(undi_input)
        NS.translate_rotate_sample_vec(np.array([0.,0.,1.]))
        signal_z = NS.celio_on_steroids(t,  k=1, algorithm='light')
        del NS

        results.append(
            {
                "cluster_isotopes": cluster_isotopes,
                "spins": cluster_spins,
                "t": t.tolist(),
                "signal_x": signal_x.tolist(),
                "signal_y": signal_y.tolist(),
                "signal_z": signal_z.tolist(),
                "probability" : abd_prob
            }
        )

    with open('results.json','w') as f:
        json.dump(results, f)


main()
