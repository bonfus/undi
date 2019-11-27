#!/usr/bin/env python
# coding: utf-8
import logging
from qutip import *
import numpy as np
from numpy import pi
from mendeleev import element
from copy import deepcopy

qdot = lambda x,y : x[0]*y[0] + x[1]*y[1] + x[2]*y[2]
qMdotV = lambda x,y : (x[0,0]*y[0] + x[0,1]*y[1] + x[0,2]*y[2],
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
planck2pi_alt = 6.582117e-7 #planck2pi [neV*s]

def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, completely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


class MuonNuclearInteraction(object):

    # Collection of gammas, never really used apart from muon's one.
    gammas = {'mu': 2*np.pi*135.5e6,
              'F' : 2*np.pi*40.053e6,
              'H' : 2*np.pi*42.577e6,
              'V' : 2*np.pi*11.212944e6} # (sT)^-1

    one_over_plank2pi = 1. / 6.582117e-7   # 6.582117e-7 =planck2pi [neV*s]

    @staticmethod
    def splitIsotope(s):
        return (''.join(filter(str.isdigit, s)) or None,
                ''.join(filter(str.isalpha, s)) or None)

    def __init__(self, atoms, logger = None, log_level = ''):

        # Make own copy to avoid overwriting of internal elements
        atoms = deepcopy(atoms)

        self.logger = logger or logging.getLogger(__name__)

        if log_level:
            level_config = {'debug': logging.DEBUG, 'info': logging.INFO}
            try:
                self.logger.setLevel(level_config[log_level.lower()])
            except:
                self.logger.warning("Invalid logging level")

        self.Hdim = 1

        for i, atom in enumerate(atoms):
            spin  = atom.get('Spin', None)
            label = atom.get('Label', None)
            pos   = atom.get('Position', None)

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
                if spin:
                    if spin != isotope.spin:
                        self.logger.warning("Warning, overriding spin for {}".format(label))
                else:
                    atoms[i]['Spin'] = isotope.spin

                atoms[i]['Gamma'] = isotope.g_factor * 7.622593285e6 * 2. * pi  #  \mu_N /h, is 7.622593285(47) MHz/T that in turn is equal to  γ_n / (2 π g_n)

            # increase Hilbert space dimension
            self.Hdim *= (2*atoms[i]['Spin']+1)

        # Check Hilber space dimension makes sense
        if np.abs(self.Hdim - np.rint(self.Hdim)) > 1e-10:
            raise RuntimeError("Something is very bad in the setup")
        else:
            self.Hdim = int(self.Hdim)

        self.logger.info("Hilbert space is {} dimensional".format(self.Hdim))
        self.atoms = atoms

    def create_H(self, cutoff = 10.0E-10):
        """
        cutoff is in Angstrom
        """

        atoms = self.atoms
        n_nuclei = len(atoms)

        for i in range(n_nuclei):
            Lx = tensor(* [spin_Jx(a_j['Spin']) if j == i else qeye(2*a_j['Spin']+1) for j,a_j in enumerate(atoms)] )
            Ly = tensor(* [spin_Jy(a_j['Spin']) if j == i else qeye(2*a_j['Spin']+1) for j,a_j in enumerate(atoms)] )
            Lz = tensor(* [spin_Jz(a_j['Spin']) if j == i else qeye(2*a_j['Spin']+1) for j,a_j in enumerate(atoms)] )
            atoms[i]['Operators'] = (Lx, Ly, Lz)

            # add also observables in the case of muon
            if atoms[i]['Label'] == 'mu':
                Ox = tensor(* [sigmax() if j == i else qeye(2*a_j['Spin']+1) for j,a_j in enumerate(atoms)] )
                Oy = tensor(* [sigmay() if j == i else qeye(2*a_j['Spin']+1) for j,a_j in enumerate(atoms)] )
                Oz = tensor(* [sigmaz() if j == i else qeye(2*a_j['Spin']+1) for j,a_j in enumerate(atoms)] )
                atoms[i]['Observables'] = (Ox, Oy, Oz)

        # Use last operator to get dimensions...not nice but does the job.
        H = Qobj(dims=Lx.dims)
        self.Hs = []
        # Two body dipolar interaction
        for i, a_i in enumerate(atoms):
            for j, a_j in enumerate(atoms):

                if j <= i:
                    continue

                gamma_i, r_i, s_i = a_i['Gamma'], a_i['Position'], a_i['Spin']
                gamma_j, r_j, s_j = a_j['Gamma'], a_j['Position'], a_j['Spin']

                d = r_j-r_i
                n_of_d = np.linalg.norm(d)
                u = d/n_of_d

                # cutoff for the interaction
                if n_of_d > cutoff:
                    continue

                self.logger.info("Adding interaction between {} and {} with distance {}".format( a_i['Label'], a_j['Label'], n_of_d ) )

                # dipolar interaction
                Bd=(mu_0*gamma_i*gamma_j*(hbar**2))/(4*np.pi*(n_of_d**3))
                Bd*=6.241508e27 # to neV

                I = a_i['Operators']
                J = a_j['Operators']

                H += -Bd * (3*qdot(I,u)*qdot(J,u) - qdot(I,J))
                self.Hs.append(-Bd * (3*qdot(I,u)*qdot(J,u) - qdot(I,J)))

                self.logger.info('Added Dipolar contribution: {}'.format(Bd))

        # Quadrupolar atoms
        for i, a_i in enumerate(atoms):
            l = a_i['Spin']
            if (l > 0.5):
                EFG = a_i.get('EFGTensor', None)
                Q = a_i.get('ElectricQuadrupoleMoment', None)
                if (EFG is None) or (Q is None):
                    self.logger.info('Skipped quadrupolar coupling for atom {} {}'.format(i, a_i['Label']))
                    continue
                I  = a_i['Operators']
                # Quadrupole
                self.logger.info('Adding quadrupolar contribution: {}'.format(J_to_neV * (elementary_charge * Q /(2*l *(2*l -1))) * EFG[0,0]))
                H += J_to_neV * (elementary_charge * Q /(2*l *(2*l -1))) * (qdot(I, qMdotV(EFG,I)))

        self.H = H

    def time_evolve_qutip(self, dt, steps):

        atoms = self.atoms
        for atom in atoms:
            if atom['Label'] == 'mu':
                Ox, Oy, Oz = atom['Observables']

        rhox = (1./self.Hdim) * Ox
        rhoy = (1./self.Hdim) * Oy
        rhoz = (1./self.Hdim) * Oz

        dU = (-1j * self.H * self.one_over_plank2pi * dt).expm()

        r = np.zeros(steps, dtype=np.complex)
        for i in range(steps):
            U = dU ** i
            r[i] += ( rhox * U.dag() * Ox * U ).tr()
            r[i] += ( rhoy * U.dag() * Oy * U ).tr()
            r[i] += ( rhoz * U.dag() * Oz * U ).tr()

        return np.real_if_close(r/3.)

    def time_evolve_trotter(self, dt, steps, k=3.):
        """
        This subroutine does Trotter expnsion of the matrix
        """
        from time import time
        atoms = self.atoms
        for atom in atoms:
            if atom['Label'] == 'mu':
                Ox, Oy, Oz = atom['Observables']

        rhox = (1./self.Hdim) * Ox
        rhoy = (1./self.Hdim) * Oy
        rhoz = (1./self.Hdim) * Oz

        # compose U operator
        dU = qeye(Oz.dims[0])
        for h in self.Hs:
            dU *= (-1j * h * self.one_over_plank2pi * dt / k).expm()
        dU = dU**k

        r = np.zeros(steps, dtype=np.complex)
        for i in range(steps):
            U = dU**i
            r[i] += ( rhox * U.dag() * Ox * U).tr()
            r[i] += ( rhoy * U.dag() * Oy * U).tr()
            r[i] += ( rhoz * U.dag() * Oz * U).tr()

        return np.real_if_close(r/3.)

    def celio(self, dt, steps, k=3.):
        """
        This implements Celio's approximation as in Phys. Rev. Lett. 56 2720
        """

        # internal copy
        atoms = self.atoms

        # generate maximally mixed state for nuclei (all states populated with random phase)
        # also record muon index to later add polarized state
        mu_idx = -1
        nuclear_states = []
        for l, atom in enumerate(atoms):
            # record muon position in list. To be used to insert polarized state
            if atom['Label'] == 'mu':
                Ox, Oy, Oz = atom['Observables']
                mu_idx = l
                continue

            # Create dephased states for current nucleus
            S = int(2*atom['Spin']+1)
            psi = Qobj( np.exp(-2.j * np.pi * np.random.rand(S)), type='ket')
            nuclear_states.append(psi)

        if mu_idx < 0:
            print('error')

        # Insert muon along +z
        nuclear_states.insert(mu_idx, basis(2,0))
        psiz = tensor(nuclear_states)

        # Insert muon along +x (after having removed the previous state)
        nuclear_states.pop(mu_idx) ; nuclear_states.insert(mu_idx, 0.7071067811865475*(basis(2,0)+basis(2,1)))
        psix = tensor(nuclear_states)

        # Insert muon along +y (after having removed the previous state)
        nuclear_states.pop(mu_idx) ; nuclear_states.insert(mu_idx, 0.7071067811865475*(basis(2,0)+1.j*basis(2,1)))
        psiy = tensor(nuclear_states)

        # Normalize
        psix = psix * np.sqrt(2./self.Hdim)
        psiy = psiy * np.sqrt(2./self.Hdim)
        psiz = psiz * np.sqrt(2./self.Hdim)


        # Computer time evolution operator
        #
        # Exact H
        #   dU = (-1j * self.H * self.one_over_plank2pi * dt).expm()
        #
        # Trotter:
        dU = qeye(Oz.dims[0])
        for h in self.Hs:
            dU *= (-1j * h * self.one_over_plank2pi * dt / k).expm()
        dU = dU**k

        r = np.zeros(steps, dtype=np.complex)
        for i in range(steps):
            # set spin along x and measure along x (we don't want to rotate the system!)
            r[i] += (psix.dag() * Ox * psix)[0,0]
            # same as above for y
            r[i] += (psiy.dag() * Oy * psiy)[0,0]
            # same as above for z
            r[i] += (psiz.dag() * Oz * psiz)[0,0]

            # Evolve psi
            psix = dU * psix
            psiy = dU * psiy
            psiz = dU * psiz

        return np.real_if_close(r/3.)

    def compute(self, cutoff = 10.0E-10):
        """
        This generates the Hamiltonian and finds eigenstates
        """
        # generate Hamiltonian
        self.create_H(cutoff)

        # find the energy eigenvalues of the composite system
        self.evals, self.ekets = self.H.eigenstates()


    def load_or_solve_H(self, cutoff = 10.0e-10, load_eigenpairs = False, eigenpairs_file=''):
        """
        This is a helper function to solve or load previous results.
        """

        if load_eigenpairs == False:
            self.logger.info("Diagonalizing matrix...")
            self.compute(cutoff)
            self.logger.info("done...")
            if eigenpairs_file:
                np.savez(save_eigenpairs, evals = self.evals, ekets = self.ekets)
        else:
            data = np.load(eigenpairs_file)

            self.evals = data['evals']
            self.ekets = data['ekets']

    def sample_spherical(self):
        """
        This computes the elements to be later traced. Simple and slow implementation.
        """

        atoms = self.atoms
        for atom in atoms:
            if atom['Label'] == 'mu':
                Ox, Oy, Oz = atom['Observables']

        ekets = self.ekets

        AA = np.zeros([len(ekets),len(ekets),3], dtype=np.complex)
        for idx in range(len(ekets)):
            for jdx in range(len(ekets)):
                AA[idx,jdx,0]=np.abs(Ox.matrix_element(ekets[idx],ekets[jdx]))**2
                AA[idx,jdx,1]=np.abs(Oy.matrix_element(ekets[idx],ekets[jdx]))**2
                AA[idx,jdx,2]=np.abs(Oz.matrix_element(ekets[idx],ekets[jdx]))**2

        return (AA[:,:,0]+AA[:,:,1]+AA[:,:,2])*0.3333333333

    def fast_sample_spherical(self):
        """
        Same as above, but with numpy vectorized operations.
        """

        atoms = self.atoms
        ekets = self.ekets

        for atom in atoms:
            if atom['Label'] == 'mu':
                Ox, Oy, Oz = atom['Observables']

        self.logger.info('Storing kets in dense matrices')
        allkets = np.matrix(np.zeros((len(ekets),len(ekets)), dtype=np.complex))
        for idx in range(len(ekets)):
            allkets[:,idx] = ekets[idx].data.toarray()[:,0].reshape((len(ekets),1))

        w = np.matrix(np.zeros((len(ekets),len(ekets)), dtype=np.float))
        self.logger.info('Apply operators x')
        w += np.square(  np.abs(   allkets.T*Ox.data.toarray()*allkets  )   ) # AAx = allkets.T*Ox.data.toarray()*allkets
        self.logger.info('Apply operators y')
        w += np.square(  np.abs(   allkets.T*Oy.data.toarray()*allkets  )   ) # AAy = allkets.T*Oy.data.toarray()*allkets
        self.logger.info('Apply operators z')
        w += np.square(  np.abs(   allkets.T*Oz.data.toarray()*allkets  )   ) # AAz = allkets.T*Oz.data.toarray()*allkets
        #
        # This is what is done above...
        #for idx in range(len(ekets)):
        #    for jdx in range(len(ekets)):
        #        AA[idx,jdx,0]=np.abs(Ox.matrix_element(ekets[idx],ekets[jdx]))**2
        #        AA[idx,jdx,1]=np.abs(Oy.matrix_element(ekets[idx],ekets[jdx]))**2
        #        AA[idx,jdx,2]=np.abs(Oz.matrix_element(ekets[idx],ekets[jdx]))**2
        #
        return w*0.3333333333333 #(np.square(np.abs(AAx)) + np.square(np.abs(AAy)) + np.square(np.abs(AAz)))/3.


    def generate_signal(self, tlist, approximated=False):

        w=self.fast_sample_spherical()

        if approximated:
            return self._generate_approximated_signal(tlist, w)
        else:
            return self._generate_signal(tlist, w)

    def _generate_signal(self, tlist, w):
        signal = np.zeros_like(tlist, dtype=np.complex)

        evals = self.evals

        # this does e_i - e_j for all eigenvalues
        ediffs  = np.subtract.outer(evals, evals)
        ediffs *= self.one_over_plank2pi

        for idx in range(len(evals)):
            self.logger.info('Adding signal {}...'.format(idx))
            for jdx in range(len(evals)):
                signal += np.exp( 1.j*ediffs[idx,jdx]*tlist ) * w[idx,jdx] # 6.582117e-7 =planck2pi [neV*s]

        return ( np.real_if_close(signal / self.Hdim ) )

    def _generate_approximated_signal(self, tlist, w, weps=1e-18, feps=1e-14):

        evals = self.evals

        factor = 4.0/len(evals)
        weps *= factor
        tmax = np.max(tlist)

        signal = np.zeros_like(tlist, dtype=np.complex)

        # makes the difference of all eigenvalues
        ediffs  = np.subtract.outer(evals, evals)
        ediffs *= self.one_over_plank2pi

        order_w = False
        if order_w:
            self.logger.info("Ordering weights")

            idx = np.unravel_index(np.argsort(w, axis=None)[::-1], w.shape)

            _x, _y = idx[0][0], idx[1][0]
        else:
            _x = range(0,len(evals))
            _y = range(0,len(evals))

        for idx in _x:
            self.logger.info('Adding signal...{}'.format(idx))
            for jdx in _y:
                if w[idx,jdx] > weps:
                    if (np.abs(ediffs[idx,jdx]*tmax) > feps ):
                        signal += np.exp( 1.j*ediffs[idx,jdx]*tlist ) * w[idx,jdx] # 6.582117e-7 =planck2pi [neV*s]
                    else:
                        signal += w[idx,jdx]
                        self.logger.info('Skipped (freq) {} {} {}'.format(idx, jdx, ediffs[idx,jdx]*tmax) )
                else:
                    self.logger.info('Skipped (weight) {} {} {}'.format(idx, jdx, w[idx,jdx]) )
        return ( np.real_if_close(signal / self.Hdim ) )


if __name__ == '__main__':
    """
    Here we always use SI in input.
    """
    import matplotlib.pyplot as plt
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
    # cutoff the dipolar interaction in order to avoid F-F term
    NS.load_or_solve_H(cutoff=1.2 * angtom)

    signal_FmuF = NS.generate_signal(tlist)

    # no cutoff this time
    NS.load_or_solve_H()

    signal_FmuF_with_Fdip = NS.generate_signal(tlist)

    fig, axes = plt.subplots(1,1)
    axes.plot(tlist, signal_FmuF, label='Computed', linestyle='-')
    axes.plot(tlist, signal_FmuF_with_Fdip, label='Computed, with F-F interaction', linestyle='-.')

    # Generate and plot analytical version for comparison
    def plot_brewer(interval,r):
        from numpy import cos, sin, sqrt
        omegad = (mu_0*NS.gammas['mu']*NS.gammas['F']*(hbar))
        omegad /=(4*np.pi*((r)**3))

        tomegad=interval*omegad
        y = (1./6.)*(3+cos(sqrt(3)*tomegad)+ \
                    (1-1/sqrt(3))*cos(((3-sqrt(3))/2)*tomegad)+ \
                    (1+1/sqrt(3))*cos(((3+sqrt(3))/2)*tomegad))#+0.05*(exp(-x/2.5))**1.5
        return y

    axes.plot(tlist, plot_brewer(tlist, r), label='F-mu-F Brewer', linestyle=':')


    ticks = np.round(axes.get_xticks()*10.**6)
    axes.set_xticklabels(ticks)
    axes.set_xlabel(r'$t (\mu s)$', fontsize=20)
    axes.set_ylabel(r'$\left<P_z\right>$', fontsize=20);
    axes.grid()
    fig.legend()
    plt.show()
