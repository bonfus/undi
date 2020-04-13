# Importing stuff...
try:
    from undi import MuonNuclearInteraction
except (ImportError, ModuleNotFoundError):
    import sys
    sys.path.append('../undi')
    from undi import MuonNuclearInteraction
import matplotlib.pyplot as plt
import numpy as np



#| # LaCuO4
#| 
#| This notebook tries to reproduce the results in the papers https://arxiv.org/pdf/1201.5406.pdf or https://doi.org/10.1016/j.phpro.2012.04.056
#| Some additional information can be found here:
#| 
#|  * https://journals.aps.org/prb/abstract/10.1103/PhysRevB.49.9879
#|  * https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.72.760
#|  * https://doi.org/10.1103/PhysRevB.64.134525
#| 
#| ![](images/Fig1.png)
#| ![](images/Fig5a.png)
#| ![](images/Fig6.png)


# Input data and constants

angtom=1.0e-10 # m

Quadrupole_moment = {
'Cu' :  -0.211e-28 ,  # m^2 
'La' : 0.22e-28       # m^2
}

Omega = {
'Cu': 2*np.pi*34.0e6,          #  213.6e6 or 194.8e6 # Hz
'La': 2*np.pi*6.4e6            #  40.2e6
}

eta={
'Cu':0.0, #0.02                          # eta, extract form paper
'La':0.0 # 0.03
}


#| From "Nuclear Quadrupole Resonance Spectroscopy" by Hand and Das, Solid State Physics Supplement 1, the NQR transition frequency $\omega$ for a spin $S$ with electric field gradient principal component $V_{zz}$ and quadrupole moment $Q$ is
#| 
#| 
#| $\begin{align}
#|     A &= \frac{eV_{zz}Q}{4S(2S-1)}\\
#|     \omega &= 3 A (2 |m| + 1)/\hbar
#| \end{align}$
#| 
#| for all available transitions $m \rightarrow m+1$ where $-S \le m \le S$.



def EFG_from_omegaq_PAS(omegaq, eta, m, I, Q):
    #
    # (2/3) (planck2pi × 1s^−1 )/(elementary_charge × 1 m^2) = ((2 ∕ 3) × (planck2pi × (1 × (second^−1)))) ∕ (elementary_charge × (1 × (meter^2))) to volt × (meter^−2)
    # 4.3880797E-16 volt ∕ meter^2
    # the following is equivalent to the value provided below
    #  Vzz = omegaq * (I*(2*I-1))  * 4.3880797E-16 / Q
    #
    plank2pi = 1.0545718E-34 #joule second
    elementary_charge=1.6021766E-19     # Coulomb = ampere ⋅ second
    
    A = omegaq / (3 * (2*np.abs(m)+1)/plank2pi)
    Vzz = A * (4 * I * (2 * I -1))/(Q * elementary_charge)
    
    
    Vxx = (Vzz/2.0) * (eta-1)
    Vyy = -(Vzz/2.0) * (eta+1)
    return np.diag([Vxx, Vyy,Vzz])



#| ## Calculate Electric Field Gradient from the muon


elementary_charge=1.6021766E-19 # Coulomb = ampere ⋅ second

def Vzz_for_unit_charge_at_distance(r):
    epsilon0 = 8.8541878E-12 # ampere^2 ⋅ kilogram^−1 ⋅ meter^−3 ⋅ second^4
    elementary_charge=1.6021766E-19 # Coulomb = ampere ⋅ second
    Vzz = (2./(4 * np.pi * epsilon0)) * (elementary_charge / (r**3))
    return Vzz

def gen_radial_EFG(p_mu, p_N, Vzz=None):
    x=p_N-p_mu
    n = np.linalg.norm(x)

    if Vzz is None:
        Vzz = Vzz_for_unit_charge_at_distance(n)

    x /= n; r = 1. # keeping formula below for clarity
    return -Vzz * ( (3.*np.outer(x,x)-np.eye(3)*(r**2))/r**5 ) * 0.5


def gen_neighbouring_atomic_structure(muon_position, cutoffs):
    from ase.io import read
    from ase.atom import Atom
    from ase.neighborlist import neighbor_list
    
    
    atoms = read('./structures/1008481.cif')
    atoms.extend(Atom('H', [0,0,0]))
        
    # update muon position
    pos = atoms.get_scaled_positions()
    pos[-1] = muon_position
    atoms.set_scaled_positions(pos)
    
    ai,aj,D = neighbor_list('ijD',atoms, 14.) # very large cutoff to get everything. Selection is done below
    
    data = []
    muon_pos = np.array([0,0,0])
    for i in range(len(D)):
        if not (ai[i] == len(atoms)-1):
            continue
        
        symb = atoms[aj[i]].symbol
            
        if np.linalg.norm(D[i]) > cutoffs.get(symb, 0):
            continue
        
        if symb in cutoffs.keys():
            
            pos = D[i] * angtom
            print('Adding atom ', symb , ' with position', pos, ' and distance ', np.linalg.norm(pos))
            
            
            # Nuclear part of EFG
            spin = 7./2. if symb == 'La' else 3./2.
            EFG_tensor = EFG_from_omegaq_PAS(Omega[symb], eta[symb], 1./2., spin, Quadrupole_moment[symb])
            # Muon part of EFG (the muon is always at 0)
            # EFG_tensor += gen_radial_EFG(muon_pos, pos)
            
            data.append({'Position': pos,
                         'Label': symb,
                         'ElectricQuadrupoleMoment': Quadrupole_moment[symb],
                         'EFGTensor': EFG_tensor,
                         }
                        )
    data.insert(0, 
                    {'Position': muon_pos,
                     'Label': 'mu'},

                )
    return data


def gen_signal(atoms, pol_direction, k=2, nrep=1):
    ttime=20
    steps = 100
    tlist = np.linspace(0, ttime*1e-6, steps)
    
        
    NS = MuonNuclearInteraction(atoms, log_level='info')
    NS.translate_rotate_sample_vec(pol_direction)
    
    print("Computing signal...", end='', flush=True)
    signal = NS.celio(tlist,  k=k)
    if nrep > 1:
        for i in range(nrep):
            signal += NS.celio(tlist,  k=k)
        signal /= nrep+1.
    print('done!')
    del NS
    
    return signal



#| ### Site D
#| 
#| Muon site $D$ (0.120, 0, 0.219)


ttime=20
steps = 100
tlist = np.linspace(0, ttime*1e-6, steps)

atoms = gen_neighbouring_atomic_structure([0.12,  0.,  0.219], cutoffs={'Cu': 3.2, 'La': 3.2})

signal_D_Pa = gen_signal(atoms, np.array([1.,0.,0.]))
signal_D_Pc = gen_signal(atoms, np.array([0.,0.,1.]))

fig, axes = plt.subplots(1,1, figsize=(12,12))
axes.plot(tlist*1e6,signal_D_Pc,'b--',marker='o',label='P//c Calc. ', zorder=1)
axes.plot(tlist*1e6,signal_D_Pa,'y--',marker='o',label='P _|_ c Calc. ', zorder=2)

imdata = plt.imread('images/Fig6.png')
axes.imshow(imdata, zorder=0, extent=[0., 20.0, -0.3, 1.1], aspect=20/(1.1+0.3), resample=True)

axes.set_ylim([-0.3,1.1])
axes.set_xlim([0,20])
axes.set_xlabel(r'$t (\mu s)$', fontsize=20)
axes.set_ylabel(r'Asymmetry', fontsize=20);

plt.legend(loc=2, fontsize=20)
plt.savefig("D.png")
plt.show()


#| ### Site M



atoms = gen_neighbouring_atomic_structure([0.5,  0.,  0.096], cutoffs={'Cu':3.2, 'La':3.2})

print("Computing signal...", end='', flush=True)
signal_M_Pab = gen_signal(atoms, np.array([1.,1.,0.]))
print('done!')


imdata = plt.imread('images/Fig5a.png')

fig, axes = plt.subplots(1,1, figsize=(12,12))

axes.plot(tlist*1e6,signal_M_Pab,'k--',marker='o',label='M Calc. ', zorder=1)
axes.imshow(imdata, zorder=0, extent=[0., 20.0, -0.3, 1.1], aspect=20/(1.1+0.3), resample=True)


axes.set_ylim([-0.3,1.1])
axes.set_xlim([0,20])
axes.set_xlabel(r'$t (\mu s)$', fontsize=20)
axes.set_ylabel(r'Asymmetry', fontsize=20);

plt.legend(loc=2, fontsize=20)
plt.savefig("M.png")
plt.show()



#| ### Site T1 and T2


atoms = gen_neighbouring_atomic_structure([0.2,  0.,  0.15], cutoffs={'Cu':4., 'La':3.}) 

print("Computing signal...", end='', flush=True)
signal_T1_Pab = gen_signal(atoms, np.array([1.,1.,0.]), k=4)
print('done!')

atoms = gen_neighbouring_atomic_structure([0.225,  0.,  0.225], cutoffs={'Cu':4.2, 'La':3.}) 

print("Computing signal...", end='', flush=True)
signal_T2_Pab = gen_signal(atoms, np.array([1.,1.,0.]), k=4)
print('done!')



fig, axes = plt.subplots(1,1, figsize=(12,12))

axes.plot(tlist*1e6,signal_T1_Pab,'r--',marker='o',label='T1 Calc. ', zorder=1)
axes.plot(tlist*1e6,signal_T2_Pab,'g--',marker='o',label='T2_Calc. ', zorder=2)

imdata = plt.imread('images/Fig1.png')
axes.imshow(imdata, zorder=0, extent=[0., 20.0, -0.3, 1.05], aspect=20/(1.05+0.3), resample=True)


axes.set_ylim([-0.3,1.05])
axes.set_xlim([0,20])
axes.set_xlabel(r'$t (\mu s)$', fontsize=20)
axes.set_ylabel(r'Asymmetry', fontsize=20);

plt.legend(loc=2, fontsize=20)
plt.savefig("T1_and_T2.png")
plt.show()


#| ### Site Ba (not really)


atoms = gen_neighbouring_atomic_structure([0.1,  0.,  0.1], cutoffs={'Cu':2., 'La':3.}) 

print("Computing signal...", end='', flush=True)
signal_Ba_Pab = gen_signal(atoms, np.array([1.,1.,0.]))
print('done!')

imdata = plt.imread('images/Fig5a.png')

fig, axes = plt.subplots(1,1, figsize=(12,12))

axes.plot(tlist*1e6,signal_Ba_Pab,'m--',marker='o',label='Ba Calc. ', zorder=1)
axes.imshow(imdata, zorder=0, extent=[0., 20.0, -0.3, 1.1], aspect=20/(1.1+0.3), resample=True)


axes.set_ylim([-0.3,1.1])
axes.set_xlim([0,20])
axes.set_xlabel(r'$t (\mu s)$', fontsize=20)
axes.set_ylabel(r'Asymmetry', fontsize=20);

plt.legend(loc=2, fontsize=20)
plt.savefig("Ba.png")
plt.show()


#|  ### Site U1 (not really)


atoms = gen_neighbouring_atomic_structure([0.12,  0.,  0.11], cutoffs={'Cu':3.7, 'La':3.})

print("Computing signal...", end='', flush=True)
signal_U1_Pab = gen_signal(atoms, np.array([1.,1.,0.]), nrep=4, k=4)
print('done!')

imdata = plt.imread('images/Fig5a.png')

fig, axes = plt.subplots(1,1, figsize=(12,12))

axes.plot(tlist*1e6,signal_U1_Pab,'b--',marker='o',label='U1 Calc. ', zorder=1)
axes.imshow(imdata, zorder=0, extent=[0., 20.0, -0.28, 1.1], aspect=20/(1.1+0.28), resample=True)


axes.set_ylim([-0.3,1.1])
axes.set_xlim([0,20])
axes.set_xlabel(r'$t (\mu s)$', fontsize=20)
axes.set_ylabel(r'Asymmetry', fontsize=20);


plt.legend(loc=2, fontsize=20)
plt.savefig("U1.png")
plt.show()



#| ### Site H


atoms = gen_neighbouring_atomic_structure([0.253,  0.,  0.152], cutoffs={'Cu':4., 'La':3.5}) 

print("Computing signal...", end='', flush=True)
signal_H_Pab = gen_signal(atoms, np.array([1.,1.,0.]), k=1)
signal_H_Pc = gen_signal(atoms, np.array([0.,0.,1.]), k=1)
print('done!')

fig, axes = plt.subplots(1,1, figsize=(12,12))

axes.plot(tlist*1e6,signal_H_Pab,'--', color='darkgreen', marker='o',label='H Calc. ', zorder=1)
axes.plot(tlist*1e6,signal_H_Pc, '--', color='lightgreen', marker='o',label='H Calc. P//c', zorder=2)

imdata = plt.imread('images/Fig1.png')
axes.imshow(imdata, zorder=0, extent=[0., 20.0, -0.3, 1.05], aspect=20/(1.05+0.3), resample=True)


axes.set_ylim([-0.3,1.05])
axes.set_xlim([0,20])
axes.set_xlabel(r'$t (\mu s)$', fontsize=20)
axes.set_ylabel(r'Asymmetry', fontsize=20);

plt.legend(loc=2, fontsize=20)
plt.savefig("H.png")
plt.show()

