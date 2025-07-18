import sys, json, argparse, re
from itertools import product
import numpy as np

from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list

from undi import MuonNuclearInteraction
from .isotopes import Element

from undi.auto_analysis import execute_auto_analysis

efg_au = 9.7173624424e21

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
    """
    Main function to parse arguments and execute the UNDI analysis.
    Arguments:
    - structure (str): Structure to be parsed.
    - --max-hdim (int, optional): Dimension of the Hilbert space. Increase to have better results. Default is 1000.
    - --atom-as-muon (str, optional): Atom to be considered as muon. Default is "H".
    - --Bmod (float, optional): Magnetic field modulus. Default is 0.0.
    - --convergence-check (bool, optional): Flag to check for convergence. Default is False.
    - --powder-average (bool, optional): Performs powder average. Default is True.
    - --algorithm (str, optional): Algorithm to be used for the analysis. Default is "fast".
    - --sample_size_average (int, optional): Number of random samples to for powder average. Default is 1000.
    - --dump (bool, optional): Flag to dump the results. Default is False.
    Returns:
    - results (dict): Dictionary with the results of the UNDI analysis.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('structure', type=str, help='Structure to be parsed')
    parser.add_argument('--max-hdim', type=int, default=1e3, help='Dimension of the Hilbert space. Increase to have better results. Default is 1000.')
    parser.add_argument('--atom-as-muon', type=str, default="H", help='Atom to be considered as muon. Default is "H".')
    parser.add_argument('--Bmod', type=float, default=0., help='Magnetic field modulus. Default is 0.0.')
    parser.add_argument('--convergence-check', type=bool, default=False, help='Flag to check for convergence. Default is False.')
    parser.add_argument('--TF', type=bool, default=True, help='Perform simulation with transverse field. Default is True.', action=argparse.BooleanOptionalAction)
    parser.add_argument('--LF', type=bool, default=True, help='Perform simulation with longitudinal field. Default is True.', action=argparse.BooleanOptionalAction)
    parser.add_argument('--powder-average', type=bool, default=True, help='Perform powder average. Default is True.', action=argparse.BooleanOptionalAction)
    parser.add_argument('--algorithm', type=str, default="fast", help='Algorithm to be used for the analysis. Default is "fast".')
    parser.add_argument('--angular-integration-steps', type=int, default=7, help='Angular grid for powder average. Default is 7.')
    parser.add_argument('--dump', type=bool, default=True, help='Flag to dump the results. Default is True.')
     
    args = parser.parse_args()
    structure = args.structure
    Bmod = args.Bmod
    convergence_check = args.convergence_check
    atom_as_muon = args.atom_as_muon
    max_hdim = args.max_hdim
    algorithm = args.algorithm
    dump = args.dump
    powder_average = args.powder_average
    angular_integration_steps = args.angular_integration_steps
    lf=args.LF
    tf=args.TF
    
    if 'magres' in structure:
        # Read structure from magres file
        atms = read_magres_file(structure)
    else:
        atms = read(structure)
        atms.set_array('efg', np.zeros([len(atms), 3, 3]))
    
    results = execute_auto_analysis(
        structure=atms,
        B_mod=Bmod,
        atom_as_muon=atom_as_muon,
        max_hdim=max_hdim,
        convergence_check=convergence_check,
        algorithm=algorithm,
        lf=lf,
        tf=tf,
        powder_average=powder_average,
        angular_integration_steps=angular_integration_steps
    )
    
    if dump:
        with open('results.json','w') as f:
            json.dump(results, f)
            
    return results

main()
