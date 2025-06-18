#!/usr/bin/env python
# coding: utf-8
import logging
import numpy as np
from numpy import pi
import functools as ft
from scipy.linalg import expm, ishermitian

from copy import deepcopy
from fractions import Fraction

from undi.isotopes import Element as element


qdot = lambda x,y : np.dot(x[0],y[0]) + np.dot(x[1],y[1]) + np.dot(x[2],y[2])
qMdotV = lambda x,y : (x[0,0] * y[0] + x[0,1]*y[1] + x[0,2]*y[2],
                        x[1,0]*y[0] + x[1,1]*y[1] + x[1,2]*y[2],
                        x[2,0]*y[0] + x[2,1]*y[1] + x[2,2]*y[2])

# Constants
angtom=1.0e-10 # m
h=6.6260693e-34 # Js
hbar=h/(2*np.pi) # Js
mu_0=(4e-7)*np.pi # Tm A-1
elementary_charge=1.6021766E-19 # Coulomb = ampere ⋅ second

# Conversions
J_to_neV = 6.241508e27 # 1 J = 6.241508e27 neV
planck2pi_neVs = 6.582117e-7 #planck2pi [neV*s]
one_over_plank2pi_neVs = 1. / planck2pi_neVs

# Simple spin class, slightly adapted from Johnny Wilkinson's MuSR Decoherence
class Spin(object):
    def __init__(self, I):
        self.I = I
        self.II = II = int(2*I)

        # pauli z
        self.Jz = -.5*np.array(np.diag([m for m in range(-II, II+2, 2)]), dtype=complex) # the self.II+2 is due to Python using exclusive ranges
        # pauli +
        self.Jp = np.diag([np.sqrt(self.I*(self.I+1) - .5*m*(.5*m+1)) for m in range(-II, II, 2)], 1) + 0.j
        # pauli -
        self.Jm = np.diag([np.sqrt(self.I * (self.I + 1) - .5 * m * (.5 * m + 1)) for m in range(-II, II, 2)], -1) + 0.j

        # calculate pauli x and y
        self.Jx = .5*(self.Jp + self.Jm)
        self.Jy = .5/1j * (self.Jp - self.Jm)

sigmax = lambda: np.array([[0.+0.j, 1.+0.j], [1.+0.j, 0.+0.j]])
sigmay = lambda: np.array([[0.+0.j, 0.-1.j], [0.+1.j, 0.+0.j]])
sigmaz = lambda: np.array([[ 1.+0.j,  0.+0.j], [ 0.+0.j, -1.+0.j]])

qeye = lambda dim: np.diag(np.ones(dim))
tensor = lambda *x: ft.reduce(np.kron, x)



class MuonNuclearInteraction(object):
    """The main UNDI class"""

    # Collection of gammas, never really used apart from muon's one.
    gammas = {'mu': 2*np.pi*135.5e6,
              'F' : 2*np.pi*40.053e6,
              'H' : 2*np.pi*42.577e6,
              'V' : 2*np.pi*11.212944e6} # (sT)^-1


    @staticmethod
    def splitIsotope(s):
        """This function separates the isotope number and the element name.

        Parameters
        ----------
        s : str
            input string, e.g. "63Cu" becomes ('63', 'Cu')

        Returns
        -------
        tuple
            (isotope number, element name)
        """
        return (''.join(filter(str.isdigit, s)) or None,
                ''.join(filter(str.isalpha, s)) or None)

    @staticmethod
    def dipolar_interaction(a_i, a_j):
        """Dipolar interaction between atom a_i and atom a_j.

        Parameters
        ----------
        a_i :
            first atom interacting with ...
        a_j :
            second atom.

        Returns
        -------
        qutip.QObj
            The dipolar contribution to the Hamiltoninan for the given couple of atoms.
        """
        gamma_i, p_i, s_i = a_i['Gamma'], a_i['Position'], a_i['Spin']
        gamma_j, p_j, s_j = a_j['Gamma'], a_j['Position'], a_j['Spin']

        d = p_j-p_i
        n_of_d = np.linalg.norm(d)
        u = d/n_of_d

        # dipolar interaction
        Bd=(mu_0*gamma_i*gamma_j*(hbar**2))/(4*np.pi*(n_of_d**3))
        # print('\n\n\n d, omega_di = ', n_of_d, Bd/hbar)
        Bd*=J_to_neV # to neV

        I = a_i['Operators']
        J = a_j['Operators']

        return -Bd * (3*qdot(I,u) @ qdot(J,u) - qdot(I,J))

    @staticmethod
    def quadrupolar_interaction(a_i):
        """Quadrupolar interaction for atom a_i in the electric field gradient
        described by 'EFGTensor'.

        Parameters
        ----------
        a_i :
            Atom considered

        Returns
        -------
        qutip.QObj
            The Quadrupolar contribution to the Hamiltoninan for the given atom.
        """
        l = a_i['Spin']

        # no quadrupolar interaction for spin 1/2
        if (l < 0.5001):
            raise RuntimeError("Ivalid spin")

        EFG = a_i['EFGTensor']
        Q = a_i['ElectricQuadrupoleMoment']
        I  = a_i['Operators']

        # PAS
        ee, ev = np.linalg.eig(EFG)
        Vxx,Vyy,Vzz = ee[np.argsort(np.abs(ee))]

        # declare just in case EFG is zero
        eta = 0
        if abs(Vzz) >= 0.000001:
            eta = (Vxx-Vyy)/Vzz

        E_q = J_to_neV * ( elementary_charge * Q * Vzz / (4*l *(2*l -1)) )

        omega_q = E_q * one_over_plank2pi_neVs
        # print('omega_q: ',omega_q )

        # Quadrupole
        cost = J_to_neV * (elementary_charge * Q /(2*l * (2*l -1)))
        return( \
                cost * (qdot(I, qMdotV(EFG,I))) , \
                (omega_q, eta) \
                )


    @staticmethod
    def muon_induced_efg(a_i, mu, eta=0):
        r"""Operator for EFG directed along muon-atom direction.
        The contribution is defined as

        $$
        \begin{aligned}
        \mathcal{H}_{\mathrm{Q}, i} &=\frac{e^{2} q_{i} Q_{i}}{4 I_{i}\left(2 I_{i}-1\right)}\left\{\left[3\left(I_{i}^{z}\right)^{2}-\left(I_{i}\right)^{2}\right]+\eta_{i}\left[\left(I_{i}^{x}\right)^{2}-\left(I_{i}^{y}\right)^{2}\right]\right\} \\
        & \equiv \hbar \omega_{\mathrm{E}}\left\{\left[3\left(I_{i}^{z}\right)^{2}-\left(I_{i}\right)^{2}\right]+\eta_{i}\left[\left(I_{i}^{x}\right)^{2}-\left(I_{i}^{y}\right)^{2}\right]\right\}
        \end{aligned}
        $$



        Parameters
        ----------
        a_i :
            Atoms affected by electric field gradient
        mu :
            muon, only used to calculate distance.

        Returns
        -------
        qutip.QObj
            The Quadrupolar contribution to the Hamiltoninan for the given atom.
        """
        l = a_i['Spin']

        if (l < 0.5001):
            raise RuntimeError("Invalid spin")

        I = a_i['Operators']

        n = a_i['Position'] - mu['Position']
        n /= np.linalg.norm(n)

        E_q = planck2pi_neVs * a_i['OmegaQmu']

        # Quadrupole
        return E_q * ( qdot(n, I) @ qdot(n, I) - 0.33333333333*(l * (l+1)))

    @staticmethod
    def custom_term(a_i):
        """Adds a custom Hamiltonian term according to the expression given in
        a_i['CustomHamiltonianTerm']

        Parameters
        ----------
        a_i :
            atom subject to CustomHamiltonianTerm

        Returns
        -------
        qutip.QObj
            The Custom contribution to the Hamiltoninan for the given atom.
        """
        I = a_i['Operators']
        Ix, Iy, Iz = I
        p = a_i['Position']
        L = a_i['Spin']
        expression = a_i['CustomHamiltonianTerm']
        return eval(expression)

    @staticmethod
    def external_field(atom, H):
        """Lorentz term for atom in the external magnetic field H

        Parameters
        ----------
        atom :
            the atoms experiencing the external field
        H :
            the external field.

        Returns
        -------
        qutip.QObj
            The Lorents contribution to the Hamiltoninan for the given atom.
        """
        return - planck2pi_neVs * atom['Gamma'] * qdot(atom['Operators'], H)

    @staticmethod
    def create_hilbert_space(atoms):
        """Generates various operators in the Hilbert space defined by atoms.

        Parameters
        ----------
        atoms :
            dictionary with atoms information.

        Returns
        -------
        list
            Hilbert space dimensions of the subspaces.
        """
        n_nuclei = len(atoms)
        dims = []
        # check spins
        for a in atoms:
            v = 2*a['Spin']+1.
            if not v.is_integer():
                raise RuntimeError('Spin {} of atom number {} is strange!'.format(a['Spin'],i))
            dims.append(int(v))

        for i in range(n_nuclei):

            Lx = tensor(* [Spin(a_j['Spin']).Jx if j == i else qeye(int(2*a_j['Spin']+1)) for j,a_j in enumerate(atoms)] )
            Ly = tensor(* [Spin(a_j['Spin']).Jy if j == i else qeye(int(2*a_j['Spin']+1)) for j,a_j in enumerate(atoms)] )
            Lz = tensor(* [Spin(a_j['Spin']).Jz if j == i else qeye(int(2*a_j['Spin']+1)) for j,a_j in enumerate(atoms)] )
            atoms[i]['Operators'] = (Lx, Ly, Lz)

            # add also observables in the case of muon
            if atoms[i]['Label'] == 'mu':
                Ox = tensor(* [sigmax() if j == i else qeye(int(2*a_j['Spin']+1)) for j,a_j in enumerate(atoms)] )
                Oy = tensor(* [sigmay() if j == i else qeye(int(2*a_j['Spin']+1)) for j,a_j in enumerate(atoms)] )
                Oz = tensor(* [sigmaz() if j == i else qeye(int(2*a_j['Spin']+1)) for j,a_j in enumerate(atoms)] )
                atoms[i]['Observables'] = (Ox, Oy, Oz)

        return dims

    def __init__(self, atoms, external_field = [0.,0.,0.], logger = None, log_level = ''):

        # Make own copy to avoid overwriting of internal elements
        atoms = deepcopy(atoms)

        self._ext_field = np.array(external_field)

        self.logger = logger or logging.getLogger(__name__)

        if log_level:
            try:
                self.logger.setLevel(getattr(logging, log_level.upper()))
            except:
                self.logger.warning("Invalid logging level")

        self.Hdim = 1

        for i, atom in enumerate(atoms):
            spin  = atom.get('Spin', None)
            label = atom.get('Label', None)
            pos   = atom.get('Position', None)
            gamma = atom.get('Gamma', None)
            quadrupole_moment = atom.get('ElectricQuadrupoleMoment', None)

            # validation
            if pos is None:
                raise ValueError('Position needed for atom {}'.format(i))

            # assign values
            if label == 'mu':
                if spin:
                    if spin != 0.5:
                        self.logger.warning("Warning, muon spin already set differs from 0.5!!")
                else:
                    atoms[i]['Spin'] = 0.5

                atoms[i]['Gamma'] = self.gammas[label]
            else:
                A, Symbol = self.splitIsotope(label)
                e = element(Symbol)
                if A:
                    A = int(A)
                    for isotope in e.isotopes:
                        if isotope.mass_number == A:
                            break
                    else:
                        raise ValueError('Isotope {} for atom {} not found.'.format(A, Symbol))
                else:

                    max_ab = -1.
                    l = -1
                    for is_n, isotope in enumerate(e.isotopes):
                        if isotope.abundance is None:
                            continue
                        if isotope.abundance > max_ab:
                            l = is_n
                            max_ab = isotope.abundance
                    # Select isotope with highest abundance
                    isotope = e.isotopes[l]

                    level = logging.WARNING if max_ab < 0.99 else logging.INFO
                    self.logger.log(level, 'Using most abundand isotope for {}, i.e. {}{}, {} abundance'.format(label, isotope.mass_number, e.symbol, max_ab))

                # check if overriding spin
                mendeelev_spin = float(Fraction(isotope.spin))
                if spin:
                    if spin != mendeelev_spin:
                        self.logger.warning("Warning, overriding spin for {}".format(label))
                else:
                    if mendeelev_spin == 0:
                        raise RuntimeError("Isotope with " + str(isotope) + " with spin 0 not allowed. Specify a different isotope or remove this nucleus.")
                    atoms[i]['Spin'] = mendeelev_spin

                # check if overriding gamma
                if gamma:
                    self.logger.warning("Warning, overriding gamma for {}".format(label))
                else:
                    if isotope.g_factor:
                        atoms[i]['Gamma'] = isotope.g_factor * 7.622593285e6 * 2. * pi  #  \mu_N /h, is 7.622593285(47) MHz/T that in turn is equal to  γ_n / (2 π g_n)
                    else:
                        self.logger.error("g_factor missing in mendeleev library, you need to specify it by hand")
                        raise RuntimeError("Missing gammma.")

                if quadrupole_moment is None:
                    atoms[i]['ElectricQuadrupoleMoment'] = isotope.quadrupole_moment * 1e-28 # m^2
                else:
                    atoms[i]['ElectricQuadrupoleMoment'] = quadrupole_moment
                    self.logger.warning("Warning, overriding quadrupole moment for {}".format(label))

            # increase Hilbert space dimension
            self.Hdim *= (2*atoms[i]['Spin']+1)

        # Check Hilber space dimension makes sense
        if np.abs(self.Hdim - np.rint(self.Hdim)) > 1e-10:
            raise RuntimeError("Something is very bad in the setup")
        else:
            self.Hdim = int(self.Hdim)

        self.logger.info("Hilbert space is {} dimensional".format(self.Hdim))
        self.atoms = atoms

    def set_extfield(self, external_field):
        """Sets an external field

        Parameters
        ----------
        external_field : numpy.array
            A 3D vector with the external field in T.

        Returns
        -------
        None
        """
        self._ext_field = np.array(external_field)

    def translate_rotate_sample_vec(self, bring_this_to_z):
        """This function translates all positions in order to put the muon at
        the origin of the Cartesian axis system and rotates the atomic position
        in order to align the vector given in `bring_this_to_z` to the z Cartesian axis.

        Parameters
        ----------
        bring_this_to_z :
            3d vector.

        Returns
        -------
        None
        """
        natoms = len(self.atoms)

        # Bring muon to origin
        for a in self.atoms:
            if a['Label'] == 'mu':
                mup = a['Position']
        for i in range(natoms):
            self.atoms[i]['Position'] = self.atoms[i]['Position'] - mup

        def rotation_matrix_from_vectors(vec1, vec2):
            """Find the rotation matrix that aligns vec1 to vec2
            https://stackoverflow.com/a/59204638

            Parameters
            ----------
            vec1 :
                A 3d "source" vector
            vec2 :
                A 3d "destination" vector

            Returns
            -------
            numpy.array
                A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
            """
            a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
            v = np.cross(a, b)
            c = np.dot(a, b)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
            return rotation_matrix

        bring_this_to_z = np.array(bring_this_to_z, dtype=np.float64)
        bring_this_to_z /= np.linalg.norm(bring_this_to_z)

        if np.allclose(bring_this_to_z, np.array([0,0,1.])):
            rmat = np.eye(3)
        elif np.allclose(bring_this_to_z, np.array([0,0,-1.])):
            rmat = np.diag([1.,-1.,-1.])
        else:
            rmat = rotation_matrix_from_vectors(bring_this_to_z, np.array([0,0,1.]))

        self.translate_rotate_sample(rmat)

    def translate_rotate_sample(self, rmat):
        """This function translates all positions in order to put the muon at
        the origin of the Cartesian axis system and rotates the atomic position
        according to rmat. EFG tensors are rotated as well.

        Parameters
        ----------
        rmat :
            3D rotation matrix.

        Returns
        -------
        None
        """
        natoms = len(self.atoms)

        # Bring muon to origin
        for a in self.atoms:
            if a['Label'] == 'mu':
                mup = a['Position']
        for i in range(natoms):
            self.atoms[i]['Position'] = self.atoms[i]['Position'] - mup

        irmat = np.linalg.inv(rmat)

        for i in range(natoms):
            self.atoms[i]['Position'] = rmat.dot(self.atoms[i]['Position'])
            if 'EFGTensor' in self.atoms[i].keys():
                self.atoms[i]['EFGTensor'] = np.dot(rmat, np.dot(self.atoms[i]['EFGTensor'], irmat))

    def _create_H(self, cutoff = 10.0E-10):
        """Generates the Hamiltonian

        Parameters
        ----------
        cutoff : float, optional
            maximum distance between atoms to be considered (Default value = 10.0E-10)

        Returns
        -------
        None
        """

        atoms = self.atoms

        # Generate Hilber space and operators
        dims = self.create_hilbert_space(atoms)

        # Empty Hamiltonian
        H = np.diag(np.zeros(np.prod(dims), dtype=complex))

        # Find the muon, used later...
        mu = None
        for a in atoms:
            if a['Label'] == 'mu':
                mu = a
        assert(mu)

        # Dipolar interaction
        for i, a_i in enumerate(atoms):
            for j, a_j in enumerate(atoms):

                if j <= i:
                    continue

                p_i = a_i['Position']
                p_j = a_j['Position']
                d = p_j-p_i
                n_of_d = np.linalg.norm(d)

                # cutoff for the interaction
                if n_of_d > cutoff:
                    self.logger.info("Skipped interaction between {} and {} with distance {}".format( a_i['Label'], a_j['Label'], n_of_d ) )
                    continue

                self.logger.info("Adding interaction between {} and {} with distance {}".format( a_i['Label'], a_j['Label'], n_of_d ) )


                H += self.dipolar_interaction(a_i, a_j)

                self.logger.info('Dipolar contribution between {}<->{}, r={}'.format(i,j, n_of_d))

        # Quadrupolar interaction
        for i, a_i in enumerate(atoms):

            if (a_i['Spin'] > 0.5):
                EFG = a_i.get('EFGTensor', None)
                Q = a_i.get('ElectricQuadrupoleMoment', None)

                if (not EFG is None) and (not Q is None):
                    Q, strengh = self.quadrupolar_interaction(a_i)
                    self.logger.info('Quadrupolar coupling for atom {} {} is {}'.format(i, a_i['Label'], strengh))
                    H += Q

                # Muon induced quadrupolar interaction
                if 'OmegaQmu' in a_i.keys():
                    H += self.muon_induced_efg(a_i, mu)

        # Custom interaction
        for i, a_i in enumerate(atoms):
            if 'CustomHamiltonianTerm' in a_i.keys():
                H += self.custom_term(a_i)


        # External field
        if np.linalg.norm(self._ext_field) > 0.000001:
            for a_i in atoms:
                H += self.external_field(a_i, self._ext_field)

        self.H = H

    def celio_on_steroids(self, tlist, k=4, direction=[0,0,1.], single_precision=False, progress=True, algorithm='light'):
        """This reimplements celio with C++.

        Parameters
        ----------
        tlist : list or numpy.array
            list of times
        k : int
            factor for Trotter approximation (Default value = 4)
        direction : list
            unused! Don't touch it. The code will complain if you touch it (Default value = [0, 0, 1])
        single_precision : bool
            use single precision algorithms (float32). Reduces memory footprint.
        progress: bool
            shows a progress bar (requires tqdm)
        algorithm: str
            C++ implementation to be used. Default is good compromise between speed and memort footprint

        Returns
        -------
        numpy.array
            Muon polarization function along z.
        """
        if algorithm == 'fat':
            from fast_quantum_blas  import initialize, finalize, evolve, measure, Handler
        elif algorithm == 'light':
            from fast_quantum_light import initialize, finalize, evolve, measure, Handler
        elif algorithm == 'fast':
            from fast_quantum       import initialize, finalize, evolve, measure, Handler
        elif algorithm == 'python':
            # python version, should never be used
            raise RuntimeError("This implementation is for debugging only!" +
                               "If you need it change the code.")
            Handler = lambda _: None

            def setsecondlast(dims, idx):
                """
                Moves nucleus idx to second last position (remember, muon must always be last!)
                """
                s = np.prod(dims)
                return np.swapaxes(np.arange(s,dtype=np.uint64).reshape(dims),idx,-2).flatten()

            def initialize(dims, idxswap):
                for ui in range(len(dims)-1):
                    idxswap[ui] = setsecondlast(dims, ui)
            def finalize(idxswap):
                del idxswap

            def measure(o,psi, dummy, dummy_):
                r = 0.
                for j in range(int(psi.shape[0]/o.shape[0])):
                    s = j*o.shape[0]
                    e = (j+1)*o.shape[0]
                    r += np.dot(np.transpose(np.conjugate(psid[s:e])), np.dot(o,psid[s:e]).T)
                return r
            def evolve(o, psi, dummy, dummy_, idxswap):
                for j in range(int(psi.shape[0]/o.shape[0])):
                    s = j*o.shape[0]
                    e = (j+1)*o.shape[0]
                    psi[idxswap[s:e]] = np.dot(o , psi[idxswap[s:e]])
        else:
            raise RuntimeError("Unknown algorithm!")

        # Import optional progress bar
        try:
            if progress:
                from tqdm import tqdm
            else:
                tqdm = lambda x: x
        except ImportError:
            tqdm = lambda x: x

        # Import fast random number generator (possibly)
        try:
            if single_precision:
                from omprnd import unif01 as uniform01
            else:
                from omprnd import unid01 as uniform01
        except ImportError:
            from numpy.random import rand as uniform01

        # Sanity checks
        if k < 1:
            raise ValueError("Invalid value for Trotter expansion.")
        if (np.abs(np.diff(tlist,2)) > 1e-14).any():
            raise ValueError("Please provide a uniformly spaced sequence of times.")

        # define types
        if single_precision:
            ftype = np.float32
            ctype = np.complex64
        else:
            ftype = np.float64
            ctype = np.complex128

        # internal copy
        atoms = self.atoms
        n_atoms = len(atoms)

        mu_idx = -1
        for l, atom in enumerate(atoms):
            # record muon position in list. To be used to insert polarized state
            if atom['Label'] == 'mu':
                mu_idx = l
                continue
        if mu_idx < 0:
            raise RuntimeError("Where is the muon!?!")

        Subspaces = []
        for l, atom in enumerate(atoms):
            if l == mu_idx:
                continue

            couple = [atoms[l].copy(),  atoms[mu_idx].copy()]

            dims = self.create_hilbert_space(couple)

            H = self.dipolar_interaction(*couple)
            d = np.linalg.norm( atoms[mu_idx]['Position'] - atoms[l]['Position'] )
            self.logger.info("Adding interaction between {} and {} with distance {}".format( atoms[mu_idx]['Label'], atom['Label'], d ) )

            if (couple[0]['Spin'] > 0.5 and 'EFGTensor' in couple[0].keys()):
                Q, info = self.quadrupolar_interaction(couple[0])
                H += Q
            if ( 'OmegaQmu' in couple[0].keys()):
                H += self.muon_induced_efg(couple[0], couple[1])

            if np.linalg.norm(self._ext_field) > 0.000001:
                # Add field to atom
                H += self.external_field(couple[0], self._ext_field)
                # Add 1/Nth field to muon
                H += self.external_field(couple[1], self._ext_field/(n_atoms-1))

            # generate maximally mixed state for nuclei (all states populated with random phase)
            NucHdim = int(2*atom['Spin']+1)
            #NuclearPsi = Qobj( np.exp(2.j * np.pi * np.random.rand(NucHdim)), type='ket')

            Subspaces.append({'H': H, 'NucHdim': NucHdim, 'Distance': d})


        # Convert list of dict to dict of list
        SubspacesInfo = {u: [dic[u] for dic in Subspaces] for u in Subspaces[0]} # https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists


        #
        # Notes for future myself.
        #
        # Here's the deal: the muon must always be the last particle
        # in the Hilbert space. This allows to work with evolution operators
        # and observables that are block diagonal, i.e. something like
        #
        #   A B 0 0 0 0 0 0
        #   C D 0 0 0 0 0 0
        #   0 0 A B 0 0 0 0
        #   0 0 C D 0 0 0 0
        #   0 0 0 0 A B 0 0
        #   0 0 0 0 C D 0 0
        #   0 0 0 0 0 0 A B
        #   0 0 0 0 0 0 C D
        #
        # thus allowing to work and/or accumulate results on small blocks
        # of the same size of the observable or the evolution operator.
        # The bad news is that you have to swap the position of the nuclei
        # in the wavefunction, making all the algorithm memory bound.

        def computeU(tt, k):
            """ Computes time evolution operators

            Parameters
            ----------
            tt :
                time step
            k :
                factor used in Trotter expansion

            Returns
            -------
            list
                Hamiltonians acting on the various subspaces.
            """
            # Computer time evolution operator.
            #  we will put the muon as the first particle
            Us = []
            for i, subspace in enumerate(Subspaces):

                # get the Hamiltonian
                hh = subspace['H']

                # evolution operator on the small matrix
                uu = expm(-1j * hh * one_over_plank2pi_neVs * tt / k)

                Us.append( uu.astype(ctype, order='C') )
            return Us


        r = np.zeros_like(tlist, dtype=np.complex128)

        # observe along direction
        direction /= np.linalg.norm(direction)
        if not np.allclose(direction,[0,0,1]):
            self.logger.log(logging.WARNING, "Polarization different from z not yet fully implemented (but it's easy to implement)")
            o = qdot((sigmax(), sigmay(), sigmaz()), direction ).astype(ctype, order='C')
        else:
            o = sigmaz().astype(ctype, order='C')


        # Insert muon polarized along positive quantization direction
        if not np.allclose(direction,[0,0,1]):
            e, v  = (o+qeye(2)).eigenstates()
            mu_psi = v[1] if e[1] > 0.1 else v[0]
        else:
            mu_psi = np.array([1.,0.j]) # basis(2,0)

        # Dimension of the nucler subspace
        HdimHalf = np.prod(SubspacesInfo['NucHdim'])

        # Full initial state, nuclei and muon (at the end!!)
        dims=SubspacesInfo['NucHdim'] + [2]

        psid = np.kron( np.exp(2.j * np.pi * uniform01(HdimHalf).astype(ftype, order='C',copy=False)) , # Initial (random) state for all nuclei
                        np.array(mu_psi).flatten().astype(ctype, order='C',copy=False)) # And muon


        self.logger.info("Size of wavefunction: {} MB".format( psid.nbytes/1024/1024 ) )

        # Normalize
        Normalization = 1./np.sqrt(HdimHalf, dtype=ctype)
        np.multiply(psid , Normalization, casting='no')

        dUs = computeU(tlist[1]-tlist[0], k)

        h = Handler()
        aux = np.empty(1, dtype=ctype) # dummy
        if algorithm == 'python': h = np.empty([len(dUs), np.prod(dims)], dtype=np.uint64)
        elif algorithm == 'fat':  aux = np.empty_like(psid, dtype=ctype)


        initialize(dims, h)

        for i, t in enumerate(tqdm(tlist)):
            # measure
            r[i] = measure(o, psid, aux, h)

            # safety check (for 99% of users)
            if ((i == 0) and (abs(r[0] - 1.0)>1e-7)): self.logger.log(logging.WARNING, "Initial polarization {}?!".format(r[0]))

            # Evolve psi
            for _ in range(k):
                for ui, dU in enumerate(dUs):
                    # compute psi evolution for nucleus ui
                    evolve(dU, psid, aux, ui, h)
        finalize(h)
        del h
        return np.real_if_close(r)

    def celio_with_milu(self, tlist, k=1, direction=[0,0,1.], outdir=''):
        """This reimplements celio with C++ using Cyclope Tensor Framework.
        Only useful for extremely large simulations.

        TODO: this function has a lot of duplicated code. Must be cleaned.

        Parameters
        ----------
        tlist : list or numpy.array
            list of times
        k : int
            factor for Trotter approximation (Default value = 4)
        direction : list
            unused! Don't touch it. The code will complain if you touch it (Default value = [0, 0, 1])
        outdir : list
            output directory where data files are stored

        Returns
        -------
        None
        """
        import json, os

        # Sanity checks
        if k < 1:
            raise ValueError("Invalid value for Trotter expansion.")
        if (np.abs(np.diff(tlist,2)) > 1e-14).any():
            raise ValueError("Please provide a uniformly spaced sequence of times.")

        # internal copy
        atoms = self.atoms
        n_atoms = len(atoms)

        mu_idx = -1
        for l, atom in enumerate(atoms):
            # record muon position in list. To be used to insert polarized state
            if atom['Label'] == 'mu':
                mu_idx = l
                continue
        if mu_idx < 0:
            raise RuntimeError("Where is the muon!?!")

        Subspaces = []
        for l, atom in enumerate(atoms):
            if l == mu_idx:
                continue

            couple = [atoms[l].copy(),  atoms[mu_idx].copy()]

            dims = self.create_hilbert_space(couple)

            d = np.linalg.norm( atoms[mu_idx]['Position'] - atoms[l]['Position'] )

            H = self.dipolar_interaction(*couple)
            self.logger.info("Adding interaction between {} and {} with distance {}".format( atoms[mu_idx]['Label'], atom['Label'], d ) )

            if (couple[0]['Spin'] > 0.5 and 'EFGTensor' in couple[0].keys()):
                Q, info = self.quadrupolar_interaction(couple[0])
                H += Q
            if ( 'OmegaQmu' in couple[0].keys()):
                H += self.muon_induced_efg(couple[0], couple[1])

            if np.linalg.norm(self._ext_field) > 0.000001:
                # Add field to atom
                H += self.external_field(couple[0], self._ext_field)
                # Add 1/Nth field to muon
                H += self.external_field(couple[1], self._ext_field/(n_atoms-1))

            # generate maximally mixed state for nuclei (all states populated with random phase)
            NucHdim = int(2*atom['Spin']+1)
            #NuclearPsi = Qobj( np.exp(2.j * np.pi * np.random.rand(NucHdim)), type='ket')

            Subspaces.append({'H': H, 'NucHdim': NucHdim, 'Distance': d})

        # Sort according to distance
        Subspaces = sorted(Subspaces, key=lambda d: d['Distance'])

        # Convert list of dict to dict of list
        SubspacesInfo = {u: [dic[u] for dic in Subspaces] for u in Subspaces[0]} # https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists

        def computeU(tt, k):
            """ Computes time evolution operators

            Parameters
            ----------
            tt :
                time step
            k :
                factor used in Trotter expansion

            Returns
            -------
            list
                Hamiltonians acting on the various subspaces.
            """
            # Computer time evolution operator.
            #  we will put the muon as the first particle
            Us = []
            for i, subspace in enumerate(Subspaces):

                # get the Hamiltonian
                hh = subspace['H']

                # evolution operator on the small matrix
                uu = expm(-1j * hh * one_over_plank2pi_neVs * tt / k)

                Us.append( uu.data.to_array() )
            return Us

        # observe along direction
        direction /= np.linalg.norm(direction)
        if not np.allclose(direction,[0,0,1]):
            self.logger.log(logging.WARNING, "Polarization different from z not yet fully implemented (but it's easy to implement)")
            o = qdot((sigmax(), sigmay(), sigmaz()), direction ).data.to_array()
        else:
            o = sigmaz().data.to_array()


        # Insert muon polarized along positive quantization direction
        if not np.allclose(direction,[0,0,1]):
            e, v  = (o+qeye(2)).eigenstates()
            mu_psi = v[1] if e[1] > 0.1 else v[0]
        else:
            mu_psi = basis(2,0)

        # Dimension of the nucler subspace
        HdimHalf = np.prod(SubspacesInfo['NucHdim'])

        # Full initial state, nuclei and muon (at the end!!)
        dims=SubspacesInfo['NucHdim'] + [2]

        dUs = computeU(tlist[1]-tlist[0], k)

        # write milu input
        milu_input = {}
        milu_input['dims'] = dims
        milu_input['dt'] = tlist[1]-tlist[0]
        milu_input['k'] = k

        if outdir: os.mkdir(outdir)

        for ui, dU in enumerate(dUs):
            np.save(os.path.join(outdir, 'OP{}'.format(ui)),
                    np.asfortranarray(np.array(dU).T.reshape([dims[ui],2,dims[ui],2])),
                    allow_pickle=False)
        np.save(os.path.join(outdir,'m'), o, allow_pickle=False)

        with open(os.path.join(outdir, 'milu.json'),'w') as f:
            json.dump(milu_input, f)



    def compute(self, cutoff = 10.0E-10):
        """This generates the Hamiltonian and finds eigenstates

        Parameters
        ----------
        cutoff : float
            maximum distance between interacting atoms (Default value = 10.0E-10)

        Returns
        -------
        None
        """
        # generate Hamiltonian
        self._create_H(cutoff)

        # assert matrix is hermitian
        ishermitian(self.H)

        # find the energy eigenvalues of the composite system
        self.evals, self.ekets = np.linalg.eigh(self.H)


    def matrix_elements(self, direction=[0,0,1]):
        """Same as above, but with numpy vectorized operations.

        Parameters
        ----------
        direction : list
            Not used. Don't touch it. The code will complain if you do (Default value = [0, 0, 1]).

        Returns
        -------
        numpy.array
            Square of matrix elements
        """

        atoms = self.atoms
        ekets = self.ekets

        for atom in atoms:
            if atom['Label'] == 'mu':
                Ox, Oy, Oz = atom['Observables']

        self.logger.info('Storing kets in dense matrices')


        #w = np.matrix(np.zeros((len(ekets),len(ekets)), dtype=np.float64))

        if direction == [0,0,1]:
            O = Oz
        elif direction == [0,1,0]:
            O = Oy
        elif direction == [1,0,0]:
            O = Ox
        else:
            raise NotImplemented

        w = np.square( np.abs( ekets.conjugate().T @ O @ ekets ) ) # AAx = allkets.T*Ox.data.to_array()*allkets

        #
        # This is what is done above...
        #for idx in range(len(ekets)):
        #    for jdx in range(len(ekets)):
        #        AA[idx,jdx]=np.abs(Ox.matrix_element(ekets[idx],ekets[jdx]))**2
        return w


    def polarization(self, tlist, cutoff = 10.0E-10, approximated=False):
        """This function computes the depolarization function for a muon with
        spin initially polarized along z and observed along the same direction.

        cutoff: can be used to limit the maximum distance considered for dipolar
        interactions. Units: Angstrom. Default: 10 Ang.

        approximated: skips evaluation of contributions that slowly depend
        on time (assumed as flat) and contributions that are nearly zero.
        Can slightly accelerate the computation.

        Parameters
        ----------
        tlist : list or numpy.array
            List of times used to compute muon polarization function.

        cutoff : float
            Maximum distance between interacting atoms.
            (Default value = 10.0E-10)

        approximated : bool
            Wether to avoid computing the contribution from matrix elements
            close to 0 and consider low frequency signals as flat.
            (Default value = False)


        Returns
        -------
        numpy.array
            Muon polarization function along z.
        """

        self.compute(cutoff=cutoff)

        w=self.matrix_elements()

        if approximated:
            return self._generate_approximated_signal(tlist, w)
        else:
            return self._generate_signal(tlist, w)

    def _generate_signal(self, tlist, w):
        r""" This function evaluates the time evolution operator at the
        times provided in tlist.
        It is later used to compute:

        .. math::

            \signa(t) = \exp \left( i\frac{H}{\hbar} t \right) \sigma \exp \left( -i\frac{H}{\hbar} t \right)

        It eventually compute the trace assuming the initial spin direction
        to be paralllel to the measurement direction and both parallel to z.
        This assumption makes the code considerably faster and does not
        really represent a limit for the user.

        Parameters
        ----------
        tlist : numpy.array
            List of times at which the muon polarization is observed.

        w : numpy.array
            Matrix elements of the operator defining the direction of observation.

        Returns
        -------
        numpy.array
            Muon polarization function along z.
        """
        signal = np.zeros_like(tlist, dtype=np.complex128)

        evals = self.evals

        # this does e_i - e_j for all eigenvalues
        ediffs  = np.subtract.outer(evals, evals)
        ediffs *= one_over_plank2pi_neVs

        # The code below is faster for small systems, but longer for large ones.
        # A better implementation should exploit Hermiticity.
        #
        # for idx in range(len(evals)):
        #     self.logger.info('Adding signal {}...'.format(idx))
        #     for jdx in range(len(evals)):
        #         signal += np.exp( 1.j*ediffs[idx,jdx]*tlist ) * w[idx,jdx] # 6.582117e-7 =planck2pi [neV*s]
        #
        for i, t in enumerate(tlist):
            signal[i] = np.sum( np.multiply( np.exp( 1.j*ediffs*t ), w ) )

        return ( np.real_if_close(signal / self.Hdim ) )


    def zero_field_distribution_powder(self, atoms=None):
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

        if atoms:
            self.atoms = atoms

        plank2pi = 1.0545718E-34 #joule second
        mu_0 = 0.0000012566371 # (kilogram meter) ∕ (ampere^2 × second^2)
        r = 0.
        gamma_mu = 0.
        pos_mu = None
        for atom in self.atoms:
            if atom['Label'] == 'mu':
                gamma_mu = atom['Gamma']
                pos_mu   = atom['Position']
                break
        else:
            self.logger.warning('Multiple muons?! Only using last one in list')


        for atom in self.atoms:
            if atom['Label'] == 'mu':
                continue
            I = atom['Spin']
            gamma = atom['Gamma']
            r3 = np.linalg.norm(atom['Position'] - pos_mu)**3

            r += 2. * (mu_0/(4*np.pi))**2    * \
                ((gamma * plank2pi) / r3)**2 * \
                0.33333333333 * (I * (I+1))

        return r * (gamma_mu**2)

    def kubo_toyabe(self, tlist):
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
        Gmu_S2 = self.zero_field_distribution_powder()
        return 0.333333333333 + 0.6666666666 * \
                (1- Gmu_S2  *  np.power(tlist,2)) * \
                np.exp( - 0.5 * Gmu_S2 * np.power(tlist,2))

if __name__ == '__main__':
    """
    Minimal example showing how to obtain FmuF polarization function.
    TODO: remove.
    """
    import matplotlib.pyplot as plt
    SAVE_REFERENCE=False
    COMPARE_REFERENCE=False

    angtom=1.0e-10 # m

    # This is a linear F-mu-F along z
    r=1.17 * angtom
    atoms = [
                {'Position': np.array([0.000000  ,  0.  ,  0]),
                'Label': 'F',
                },

                {'Position': np.array([0.000000  ,  0.  ,  r ]),
                'Label': 'mu'
                },

                {'Position': np.array([0.000000  ,  0.  ,  2*r]),
                'Label': 'F',
                }
            ]
    # Time values, in seconds
    tlist = np.linspace(0, 10e-6, 100)

    # Define main class
    NS = MuonNuclearInteraction(atoms, log_level='info')
    # Rotate sample such that axis z used to define the atomic positions
    # is aligned with quantization axis which also happens to be z.
    # Basically the next call will do nothing
    NS.translate_rotate_sample_vec([0,0,1])

    # cutoff the dipolar interaction in order to avoid F-F term
    signal_FmuF = NS.polarization(tlist, cutoff=1.2 * angtom)

    NS = MuonNuclearInteraction(atoms, log_level='info')
    NS.translate_rotate_sample_vec([0,1,0])
    signal_FmuF += NS.polarization(tlist, cutoff=1.2 * angtom)

    NS = MuonNuclearInteraction(atoms, log_level='info')
    NS.translate_rotate_sample_vec([1,0,0])
    signal_FmuF += NS.polarization(tlist, cutoff=1.2 * angtom)

    signal_FmuF /= 3.

    # no cutoff this time
    NS = MuonNuclearInteraction(atoms, log_level='info')
    NS.translate_rotate_sample_vec([0,0,1])
    signal_FmuF_with_Fdip = NS.polarization(tlist)

    NS = MuonNuclearInteraction(atoms, log_level='info')
    NS.translate_rotate_sample_vec([0,1,0])
    signal_FmuF_with_Fdip += NS.polarization(tlist)

    NS = MuonNuclearInteraction(atoms, log_level='info')
    NS.translate_rotate_sample_vec([1,0,0])
    signal_FmuF_with_Fdip += NS.polarization(tlist)

    signal_FmuF_with_Fdip /= 3.

    # no cutoff, with Celio (just for testing purposes...)
    NS = MuonNuclearInteraction(atoms, log_level='info')
    NS.translate_rotate_sample_vec([0,0,1])

    signal_FmuF_Celio = np.zeros_like(tlist)
    for _ in range(12):
        signal_FmuF_Celio += NS.celio_on_steroids(tlist, k=4)

    NS = MuonNuclearInteraction(atoms, log_level='info')
    NS.translate_rotate_sample_vec([0,1,0])
    for _ in range(12):
        signal_FmuF_Celio += NS.celio_on_steroids(tlist, k=4)


    NS = MuonNuclearInteraction(atoms, log_level='info')
    NS.translate_rotate_sample_vec([1,0,0])
    for _ in range(12):
        signal_FmuF_Celio += NS.celio_on_steroids(tlist, k=4)


    signal_FmuF_Celio /= 3.*12.

    fig, axes = plt.subplots(1,1)
    axes.plot(tlist, signal_FmuF, label='Computed', linestyle='-')
    axes.plot(tlist, signal_FmuF_with_Fdip, label='Computed, with F-F interaction', linestyle='-.')
    axes.plot(tlist, signal_FmuF_Celio, label='Computed with Celio method', linestyle='--')

    # Generate and plot analytical version for comparison
    def plot_brewer(interval,r):
        from numpy import cos, sin, sqrt
        omegad = (mu_0*NS.gammas['mu']*NS.gammas['F']*(hbar))
        omegad /=(4*np.pi*((r)**3))

        tomegad=interval*omegad
        y = (1./6.)*(3+cos(sqrt(3)*tomegad)+ \
                    (1-1/sqrt(3))*cos(((3-sqrt(3))/2)*tomegad)+ \
                    (1+1/sqrt(3))*cos(((3+sqrt(3))/2)*tomegad))
        return y

    axes.plot(tlist, plot_brewer(tlist, r), label='F-mu-F Brewer', linestyle=':')
    axes.plot(tlist, NS.kubo_toyabe(tlist), label='Kubo-Toyabe', linestyle=':')

    ticks = np.round(axes.get_xticks()*10.**6)
    axes.set_xticklabels(ticks)
    axes.set_xlabel(r'$t (\mu s)$', fontsize=20)
    axes.set_ylabel(r'$\left<P_z\right>$', fontsize=20);
    axes.grid()
    fig.legend()
    plt.show()

    if SAVE_REFERENCE:
        np.savez('reference.npz', signal_FmuF=signal_FmuF,
                                  signal_FmuF_with_Fdip=signal_FmuF_with_Fdip,
                                  signal_FmuF_Celio=signal_FmuF_Celio)
    if COMPARE_REFERENCE:
        reference_data = np.load('reference.npz')

        fig, axes = plt.subplots(1,1)
        axes.plot(tlist, signal_FmuF, label='Computed', linestyle='-')
        axes.plot(tlist, reference_data['signal_FmuF'], label='Reference', linestyle='--')
        fig.legend()
        plt.show()

        fig, axes = plt.subplots(1,1)
        axes.plot(tlist, signal_FmuF_with_Fdip, label='Computed, with F-F interaction', linestyle='-')
        axes.plot(tlist, reference_data['signal_FmuF_with_Fdip'], label='Reference', linestyle='--')
        fig.legend()
        plt.show()

        fig, axes = plt.subplots(1,1)
        axes.plot(tlist, signal_FmuF_Celio, label='Computed with Celio method', linestyle='-')
        axes.plot(tlist, reference_data['signal_FmuF_Celio'], label='Reference', linestyle='--')
        fig.legend()
        plt.show()
