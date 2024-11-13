import sys, json, argparse
from itertools import product
import numpy as np

from ase.io import read
from ase.neighborlist import neighbor_list

from undi import MuonNuclearInteraction
from .isotopes import Element

angtom = 1e-10
efg_au = 9.7173624424e21

def gen_neighbouring_atomic_structure(atoms, isotopes, spins, hdim_max):

    ai,aj,d,D = neighbor_list('ijdD',atoms, 10.) # very large cutoff to get all possible interactions.
                                              # Actual selection is done below

    EFGs = atoms.get_array('efg') # EFG should be stored in the atoms object. Done via read_magres_file() function (see below).

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

def read_magres_file(filename):
    i = 0

    atoms_type = []
    atoms_pos = []
    with open(filename, 'r') as f:
        for line in f:
            if 'units' in line:
                if 'lattice' in line:
                    assert('Angstrom' in line)
                if 'efg' in line:
                    EFGs = np.zeros([len(atoms_type), 3, 3])
                    assert('au' in line)
                continue
            if 'lattice' in line:
                lattice = [float(x)  for x in line.split()[1:]]
            if re.search(r'\batom\b', line):
                info = line.split()
                print(info)
                atoms_type.append(info[1])
                atoms_pos.append( [float(x) for x in info[4:]] )
            if 'efg' in line:
                efg_values = line.split()[3:]
                if len(efg_values) == 9:
                    efg= [float(x) * efg_au for x in efg_values]
                    EFGs[i] = np.array(efg).reshape(3,3)
                    i += 1
                else:
                    print("Error parsing magres file")
    atms = Atoms(symbols=atoms_type, positions=atoms_pos,
                 pbc=True,cell=np.array(lattice).reshape(3,3))
    atms.set_array('efg', EFGs)
    return atms

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-hdim', type=int, default=1e5)
    parser.add_argument('--atom-as-muon', type=str, default="H")
    parser.add_argument('--Bmod', type=float, default=0.)
    parser.add_argument('--convergence-check', type=bool, default=False)
    parser.add_argument('--algorithm', type=str, default="fast")
    parser.add_argument('structure', type=str, help='Structure to be parsed') 
    """
    The parsing of the structure should be optimized: should we be able to provide also directly and ASE? 
    XYZ file better than CIF.
    """
    
    args = parser.parse_args()

    if 'magres' in args.structure:
        # Read structure from magres file
        atms = read_magres_file(args.structure)
    else:
        atms = read(args.structure)
        atms.set_array('efg', np.zeros([len(atms), 3, 3]))
        
    Bmod = args.Bmod
    convergence_check = args.convergence_check

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
    t = np.linspace(0,20e-6,500) # time_step=0.04 micro seconds

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
        
        # Definition of LF and TF
        B_lf = Bmod * np.array([0.,0.,1.])
        B_tf = Bmod * np.array([1.,0.,0.])
        
        # Single crystal signals
        
        # Written explicitely, to make it more clear.
        
        ## signal z
        print("Doing sample along Z, LF")
        direction = np.array([0.,0.,1.])
        ### LF
        NS = MuonNuclearInteraction(undi_input,external_field=B_lf)
        NS.translate_rotate_sample_vec(direction) # translate_rotate_sample_by_vec
        signal_z_lf = NS.celio_on_steroids(t,  k=4, algorithm=args.algorithm)
        del NS
        
        if convergence_check:
            # just return the z signal to be analyzed. 
            
            for iteration in range(1,6):
                # we do an average
                NS = MuonNuclearInteraction(undi_input,external_field=B_lf)
                NS.translate_rotate_sample_vec(direction) # translate_rotate_sample_by_vec
                signal_z_lf += NS.celio_on_steroids(t,  k=4, algorithm=args.algorithm)
                del NS
            signal_z_lf /=6
                
            results.append(
                {
                    "cluster_isotopes": cluster_isotopes,
                    "spins": cluster_spins,
                    "B_ext":Bmod,
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
            with open('results.json','w') as f:
                json.dump(results, f)
                
            return
        
        ### TF
        print("Doing sample along Z, TF")
        NS = MuonNuclearInteraction(undi_input,external_field=B_tf)
        NS.translate_rotate_sample_vec(direction) # translate_rotate_sample_by_vec
        signal_z_tf = NS.celio_on_steroids(t,  k=4, algorithm=args.algorithm)
        del NS
        
        ## signal x
        direction = np.array([1.,0.,0.])
        ### LF
        print("Doing sample along X, LF")
        NS = MuonNuclearInteraction(undi_input,external_field=B_lf)
        NS.translate_rotate_sample_vec(direction) # translate_rotate_sample_by_vec
        signal_x_lf = NS.celio_on_steroids(t,  k=4, algorithm=args.algorithm)
        del NS
        ### TF
        print("Doing sample along X, TF")
        NS = MuonNuclearInteraction(undi_input,external_field=B_tf)
        NS.translate_rotate_sample_vec(direction) # translate_rotate_sample_by_vec
        signal_x_tf = NS.celio_on_steroids(t,  k=4, algorithm=args.algorithm)
        del NS

        ## signal y
        direction = np.array([0.,1.,0.])
        ### LF
        print("Doing sample along Y, LF")
        NS = MuonNuclearInteraction(undi_input,external_field=B_lf)
        NS.translate_rotate_sample_vec(direction) # translate_rotate_sample_by_vec
        signal_y_lf = NS.celio_on_steroids(t,  k=4, algorithm=args.algorithm)
        del NS
        ### TF
        print("Doing sample along Y, TF")
        NS = MuonNuclearInteraction(undi_input,external_field=B_tf)
        NS.translate_rotate_sample_vec(direction) # translate_rotate_sample_by_vec
        signal_y_tf = NS.celio_on_steroids(t,  k=4, algorithm=args.algorithm)
        del NS
        
        # powder averages: if ZF, just isotropic spatial average.
        # if LF or TF, we randomly rotate the sample 1000 times and we average.
        print("Doing sample powder average")
        if Bmod<1e-5:
            powder_signal_lf = (signal_z_lf+signal_y_lf+signal_x_lf)/3
            powder_signal_tf = (signal_z_tf+signal_y_tf+signal_x_tf)/3 
        else:
            n = int(1000) # should the at least 1e3 ! 10 is for testing.
            signal_lf = np.zeros_like(t)
            signal_tf = np.zeros_like(t)

            for i in range(n):
                direction = np.random.rand(3)
                direction /= np.linalg.norm(direction)
                NS = MuonNuclearInteraction(undi_input,external_field=B_lf)
                NS.translate_rotate_sample_vec(direction)
                signal_lf += NS.celio_on_steroids(t,  k=4, algorithm=args.algorithm)
                del NS
            powder_signal_lf = signal_lf/n 
                
            for i in range(n):
                direction = np.random.rand(3)
                direction /= np.linalg.norm(direction)
                NS = MuonNuclearInteraction(undi_input,external_field=B_tf)
                NS.translate_rotate_sample_vec(direction)
                signal_tf += NS.celio_on_steroids(t,  k=4, algorithm=args.algorithm)
                del NS
            powder_signal_tf = signal_tf/n
                    

        results.append(
            {
                "cluster_isotopes": cluster_isotopes,
                "spins": cluster_spins,
                "B_ext":Bmod,
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

    with open('results.json','w') as f:
        json.dump(results, f)


main()
