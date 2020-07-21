mUon Nuclear Dipolar Interaction (UNDI)
=======================================

UNDI is a small python library to compute the expected signal from muon-nuclear dipolar interaction.


Code status
-----------

It's in a very early stage of development, don't use it yet

Requirements
------------

The following are needed for undi to work 

+--------------+------------+------------------------------------------+-------------------------------------------------------------+
| package      | version    |                  url                     |                   description                               |
+==============+============+==========================================+=============================================================+
| python       |   3.5+     |  https://www.python.org/                 |   Program langauge environment                              |
+--------------+------------+------------------------------------------+-------------------------------------------------------------+
| numpy        |   1.8+     |  https://numpy.org/                      | A numerical computing library                               |
+--------------+------------+------------------------------------------+-------------------------------------------------------------+
| Qutip        |    4.5     |  https://qutip.org/                      | A Quantum toolbox for Hamiltonian simulation                | 
+--------------+------------+------------------------------------------+-------------------------------------------------------------+
|mendelevev    |   0.5.2    | https://github.com/lmmentel/mendeleev    | Gives various property of nuclei in periodic table          |
+--------------+------------+------------------------------------------+-------------------------------------------------------------+


We recommend to check the dependency needed to install Qutip before starting installation of undi. 

Installation
------------

Installing undi is fairly easy by pip command 

::

 $ pip install undi 

or directly from the repository source

::

 $ git clone https://github.com/bonfus/undi.git undi
 $ cd undi
 $ python setup.py --user install
  
Working with virtualenv
-----------------------

Make sure you install and  run all the package on python3 environment if you are working on virtualenv.



=========================
Getting started with undi
=========================

The directory :math:`example` contain the example highlighted below.

You can start with the following test examples.

You can run the following example using a :math:`jupyter-notebook`


***************
F-:math:`\mu`-F
***************

.. code:: ipython3

    angtom=1.0e-10 # m
    h=6.6260693e-34 # Js
    hbar=h/(2*np.pi) # Js
    mu_0=(4e-7)*np.pi # Tm A-1
    
    # This is a linear F-mu-F along z
    r=1.17 * angtom
    atoms = [
                {'Position': np.array([0., 0., 0.]),
                'Label': 'F'},
                
                {'Position': np.array([0., 0., r ]),
                'Label': 'mu'},
                
                {'Position': np.array([0., 0., 2*r]),
                'Label': 'F'}
            ]


Define the basic undi class. It contains all the basic information need to generated a signal 

.. code:: ipython3

    tlist = np.linspace(0, 10e-6, 100)   # Time values, in seconds
    
    NS = MuonNuclearInteraction(atoms)     # Define main class
    
    
Rotate the sample in x, y and z direction such that the atomic position are always align along the quantization axis z.

In This case, the signal is calculated without contritbution of F-F dipolar interaction by defining a cutoff.

.. code:: ipython3

    NS.translate_rotate_sample_vec([0,0,1])        # rotate along z
    
    # cutoff the dipolar interaction in order to avoid F-F term
    signal_FmuF = NS.polarization(tlist, cutoff=1.2 * angtom)
    
    NS = MuonNuclearInteraction(atoms, log_level='info')
    NS.translate_rotate_sample_vec([0,1,0])            # rotate along y
    signal_FmuF += NS.polarization(tlist, cutoff=1.2 * angtom)
    
    NS = MuonNuclearInteraction(atoms, log_level='info')
    NS.translate_rotate_sample_vec([1,0,0])             # rotate along x
    signal_FmuF += NS.polarization(tlist, cutoff=1.2 * angtom)
    
    signal_FmuF /= 3.


The outputs of the calculation and information of the nuclei uses in the simulation

.. parsed-literal::

    INFO:undi:Using most abundand isotope for F, i.e. 19F, 1.0 abundance
    INFO:undi:Using most abundand isotope for F, i.e. 19F, 1.0 abundance
    INFO:undi:Hilbert space is 8 dimensional
    INFO:undi:Adding interaction between F and mu with distance 1.17e-10
    INFO:undi:Dipolar contribution between 0<->1, r=1.17e-10
    INFO:undi:Skipped interaction between F and F with distance 2.34e-10
    INFO:undi:Adding interaction between mu and F with distance 1.17e-10
    INFO:undi:Dipolar contribution between 1<->2, r=1.17e-10
    INFO:undi:Storing kets in dense matrices
    INFO:undi:Adding signal 0...
    INFO:undi:Adding signal 1...
    INFO:undi:Adding signal 2...
    INFO:undi:Adding signal 3...
    INFO:undi:Adding signal 4...
    INFO:undi:Adding signal 5...
    INFO:undi:Adding signal 6...
    INFO:undi:Adding signal 7...
    INFO:undi:Using most abundand isotope for F, i.e. 19F, 1.0 abundance
    INFO:undi:Using most abundand isotope for F, i.e. 19F, 1.0 abundance
    INFO:undi:Hilbert space is 8 dimensional
    INFO:undi:Adding interaction between F and mu with distance 1.17e-10
    INFO:undi:Dipolar contribution between 0<->1, r=1.17e-10
    INFO:undi:Skipped interaction between F and F with distance 2.34e-10
    INFO:undi:Adding interaction between mu and F with distance 1.17e-10
    INFO:undi:Dipolar contribution between 1<->2, r=1.17e-10
    INFO:undi:Storing kets in dense matrices
    INFO:undi:Adding signal 0...
    INFO:undi:Adding signal 1...
    INFO:undi:Adding signal 2...
    INFO:undi:Adding signal 3...
    INFO:undi:Adding signal 4...
    INFO:undi:Adding signal 5...
    INFO:undi:Adding signal 6...
    INFO:undi:Adding signal 7...


We calculate signal as above but with F-F Dipolar Interaction

.. code:: ipython3
    
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


The infos...

.. parsed-literal::

    INFO:undi:Using most abundand isotope for F, i.e. 19F, 1.0 abundance
    INFO:undi:Using most abundand isotope for F, i.e. 19F, 1.0 abundance
    INFO:undi:Hilbert space is 8 dimensional
    INFO:undi:Adding interaction between F and mu with distance 1.17e-10
    INFO:undi:Dipolar contribution between 0<->1, r=1.17e-10
    INFO:undi:Adding interaction between F and F with distance 2.34e-10
    INFO:undi:Dipolar contribution between 0<->2, r=2.34e-10
    INFO:undi:Adding interaction between mu and F with distance 1.17e-10
    INFO:undi:Dipolar contribution between 1<->2, r=1.17e-10
    INFO:undi:Storing kets in dense matrices
    INFO:undi:Adding signal 0...
    INFO:undi:Adding signal 1...
    INFO:undi:Adding signal 2...
    INFO:undi:Adding signal 3...
    INFO:undi:Adding signal 4...
    INFO:undi:Adding signal 5...
    INFO:undi:Adding signal 6...
    INFO:undi:Adding signal 7...
    INFO:undi:Using most abundand isotope for F, i.e. 19F, 1.0 abundance
    INFO:undi:Using most abundand isotope for F, i.e. 19F, 1.0 abundance
    INFO:undi:Hilbert space is 8 dimensional
    INFO:undi:Adding interaction between F and mu with distance 1.17e-10
    INFO:undi:Dipolar contribution between 0<->1, r=1.17e-10
    INFO:undi:Adding interaction between F and F with distance 2.34e-10
    INFO:undi:Dipolar contribution between 0<->2, r=2.34e-10
    INFO:undi:Adding interaction between mu and F with distance 1.17e-10
    INFO:undi:Dipolar contribution between 1<->2, r=1.17e-10
    INFO:undi:Storing kets in dense matrices
    INFO:undi:Adding signal 0...
    INFO:undi:Adding signal 1...
    INFO:undi:Adding signal 2...
    INFO:undi:Adding signal 3...
    INFO:undi:Adding signal 4...
    INFO:undi:Adding signal 5...
    INFO:undi:Adding signal 6...
    INFO:undi:Adding signal 7...
    INFO:undi:Using most abundand isotope for F, i.e. 19F, 1.0 abundance
    INFO:undi:Using most abundand isotope for F, i.e. 19F, 1.0 abundance
    INFO:undi:Hilbert space is 8 dimensional
    INFO:undi:Adding interaction between F and mu with distance 1.17e-10
    INFO:undi:Dipolar contribution between 0<->1, r=1.17e-10
    INFO:undi:Adding interaction between F and F with distance 2.34e-10
    INFO:undi:Dipolar contribution between 0<->2, r=2.34e-10
    INFO:undi:Adding interaction between mu and F with distance 1.17e-10
    INFO:undi:Dipolar contribution between 1<->2, r=1.17e-10
    INFO:undi:Storing kets in dense matrices
    INFO:undi:Adding signal 0...
    INFO:undi:Adding signal 1...
    INFO:undi:Adding signal 2...
    INFO:undi:Adding signal 3...
    INFO:undi:Adding signal 4...
    INFO:undi:Adding signal 5...
    INFO:undi:Adding signal 6...
    INFO:undi:Adding signal 7...


Generate an analytical function (:math:`G(t)`) for comparison . All relevant information can be found in J.H.Brewer et al. PRB 33 11 (1986).

:math:`G(t)=\frac{1}{6}\left[3+\cos(\sqrt{3} \omega_\text{D} t)+\left(1-\frac{1}{\sqrt{3}}\right)\cos(\frac{3-\sqrt{3}}{2}\omega_\text{D} t)+\left(1+\frac{1}{\sqrt{3}}\right)\cos(\frac{3+\sqrt{3}}{2}\omega_\text{D} t)\right]`

.. code:: ipython3
    
    def plot_brewer(interval,r):
        from numpy import cos, sin, sqrt
        omegad = (mu_0*NS.gammas['mu']*NS.gammas['F']*(hbar))
        omegad /=(4*np.pi*((r)**3))
        
        tomegad=interval*omegad
        y = (1./6.)*(3+cos(sqrt(3)*tomegad)+ \
                    (1-1/sqrt(3))*cos(((3-sqrt(3))/2)*tomegad)+ \
                    (1+1/sqrt(3))*cos(((3+sqrt(3))/2)*tomegad))#+0.05*(exp(-x/2.5))**1.5
        return y

Plot the signals calculated

.. code:: ipython3

    fig, axes = plt.subplots(1,1)
    axes.plot(tlist, signal_FmuF, label='Computed', linestyle='-')
    axes.plot(tlist, signal_FmuF_with_Fdip, label='Computed, with F-F interaction', linestyle='-.')
    axes.plot(tlist, plot_brewer(tlist, r), label='F-mu-F Brewer', linestyle=':')
    
    ticks = np.round(axes.get_xticks()*10.**6)
    axes.set_xticklabels(ticks)
    axes.set_xlabel(r'$t (\mu s)$', fontsize=20)
    axes.set_ylabel(r'$\left<P_z\right>$', fontsize=20);
    axes.grid()
    fig.legend()
    plt.show()

.. figure:: output_8_0.png
   :alt: fmuf signal
   :align: center
   :width: 400px

   F-:math:`\mu`-F Signal in LiF


In above example due to the dimensionality of Hamiltonian (:math:`8\times 8`) we used :math:`NS.polarization(tlist)`. 

The next examples explore how to use the program to deal with huge dimensional matrix.


*********************
Copper (:math:`63Cu`)
*********************

This is simulation that follws the results published in M. Celio Phys. Rev. Lett. 56 2720 (1986).

The muon and Nuclei positions are define as 

.. code:: ipython3

    angtom=1.0e-10 # m
    a=3.6212625504 # Cu lattice constant, in Angstrom
    
    Cu_Quadrupole_moment =  (-0.211) * (10**-28) # m^2
    atoms = [
        
        {'Position': np.array([0.5, 0.5, 0.5]) * a * angtom,
         'Label': 'mu'},
    
        {'Position': np.array([0.0    ,    0.5   ,     0.5])*angtom*a,
         'Label': '63Cu',
         'ElectricQuadrupoleMoment': Cu_Quadrupole_moment,
        },
        
        {'Position': np.array([0.5    ,    0.0   ,     0.5])*angtom*a,
         'Label': '63Cu',
         'ElectricQuadrupoleMoment': Cu_Quadrupole_moment,
        },
        
        {'Position': np.array([1.0    ,   0.5   ,     0.5])*angtom*a,
         'Label': '63Cu',
         'ElectricQuadrupoleMoment': Cu_Quadrupole_moment,
        },
        
        {'Position': np.array([0.5    ,    1.0   ,     0.5])*angtom*a,
         'Label': '63Cu',
         'ElectricQuadrupoleMoment': Cu_Quadrupole_moment,
        },
        
        {'Position': np.array([0.5    ,    0.5   ,     0.0])*angtom*a,
        'Label': '63Cu',
        'ElectricQuadrupoleMoment': Cu_Quadrupole_moment,
        },
        
        {'Position': np.array([0.5    ,    0.5   ,     1.0])*angtom*a,
         'Label': '63Cu',
         'ElectricQuadrupoleMoment': Cu_Quadrupole_moment,
        }
    ]


Electric field gradient generated by muon.
For more information check M. Camani et al  Phys. Rev. Lett. 39, 836 (1977) and   M. Celio Phys. Rev. Lett. 56 2720 (1986)
    
.. code:: ipython3
    
    elementary_charge=1.6021766E-19 # Coulomb 
    
    def Vzz_from_Celio_PRL():
        # 0.27 angstrom^−3 is from PRL 39 836
        # (4 pi epsilon_0)^−1 (0.27 angstrom^−3) elementary_charge = 3.8879043E20 meter^−2 ⋅ volts
        Vzz = 1.02702 * 3.8879043E20 
        # the factor 1.02702 gives exactly 3.2e6 s^-1 for omega_q quadrupole interaction strength
        # in Phys. Rev. Lett. 56 2720 (1986)
        return Vzz


A function to define the radial Electric Field Gradient (EFG) tensor that depends on the Cu-mu distance.
    
.. code:: ipython3
    
    def gen_radial_EFG(p_mu, p_N, Vzz):
        x=p_N-p_mu
        n = np.linalg.norm(x)
        x /= n; r = 1. # keeping formula below for clarity
        return -Vzz * ( (3.*np.outer(x,x)-np.eye(3)*(r**2))/r**5 ) * 0.5
    
    # add the EFG to the dictionary list of atom description
    for idx, atom in enumerate(atoms):
        if atom['Label'] == '63Cu':
            atoms[idx]['EFGTensor'] = gen_radial_EFG(atoms[0]['Position'], atom['Position'], Vzz_from_Celio_PRL())


Generating signals for varous external longitudinal fields (LF) as reported in M. Celio Phys. Rev. Lett. 56 2720 (1986)

.. code:: ipython3
 
    
    steps = 200
    tlist = np.linspace(0, 16e-6, steps)
    signals = np.zeros([6,steps], dtype=np.float)
    
    LongitudinalFields = (0.0, 0.001, 0.003, 0.007, 0.008, 0.01) # LF field in Tesla
    for idx, Bmod in enumerate(LongitudinalFields):
    
        # Put field along muon polarization, that is always z
        B = Bmod * np.array([0,0.,1.])
        NS = MuonNuclearInteraction(atoms, external_field=B, log_level='info')
    
        NS.translate_rotate_sample_vec(np.array([1.,1.,1.])) # along [111] direction
    
        print("Computing signal 4 times with LF {} T...".format(Bmod), end='', flush=True)
        signal_Cu = NS.celio(tlist,  k=2)
        for i in range(3):
            print('{}...'.format(i+1), end='', flush=True)
            signal_Cu += NS.celio(tlist, k=2)
        print('done!')
        signal_Cu /= float(i+1+1)
        del NS
    
        signals[idx]=signal_Cu


The infos. from the calculation


.. parsed-literal::

    INFO:undi.undi:Hilbert space is 8192 dimensional


.. parsed-literal::

    Computing signal 4 times with LF 0.0 T...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10


.. parsed-literal::

    1...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10


.. parsed-literal::

    2...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10


.. parsed-literal::

    3...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Hilbert space is 8192 dimensional


.. parsed-literal::

    done!
    Computing signal 4 times with LF 0.001 T...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10


.. parsed-literal::

    1...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10


.. parsed-literal::

    2...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10


.. parsed-literal::

    3...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Hilbert space is 8192 dimensional


.. parsed-literal::

    done!
    Computing signal 4 times with LF 0.003 T...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10


.. parsed-literal::

    1...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10


.. parsed-literal::

    2...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10


.. parsed-literal::

    3...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Hilbert space is 8192 dimensional


.. parsed-literal::

    done!
    Computing signal 4 times with LF 0.007 T...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10


.. parsed-literal::

    1...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10


.. parsed-literal::

    2...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10


.. parsed-literal::

    3...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Hilbert space is 8192 dimensional


.. parsed-literal::

    done!
    Computing signal 4 times with LF 0.008 T...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10


.. parsed-literal::

    1...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10


.. parsed-literal::

    2...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10


.. parsed-literal::

    3...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Hilbert space is 8192 dimensional


.. parsed-literal::

    done!
    Computing signal 4 times with LF 0.01 T...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10


.. parsed-literal::

    1...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10


.. parsed-literal::

    2...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10


.. parsed-literal::

    3...

.. parsed-literal::

    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10
    INFO:undi.undi:Adding interaction between mu and 63Cu with distance 1.8106312752000003e-10


.. parsed-literal::

    done!

Ploting the results...
        
.. code:: ipython3

    fig, axes = plt.subplots(1,1)
    for i, Bmod in enumerate(LongitudinalFields):
        color = list(np.random.choice(range(256), size=3)/256)
        axes.plot(1e6*tlist, signals[i], label='{} mT'.format(Bmod*1e3), linestyle='-', color=color)
    axes.set_ylim((-0.1,1.1))
    axes.set_xlabel(r'$t (\mu s)$', fontsize=20)
    axes.set_ylabel(r'$P_z(t)$', fontsize=20);
    axes.grid()
    plt.legend()
    plt.show()


.. figure:: output_15_0.png
   :alt: Cu signal
   :align: center
   :width: 400px
   
   Muon spin polarization along the [111] direction as a function of time for various applied longitudinal fields.
   The exact result (dash-dotted line) is compared with the approximated one (continuous lines).
   


`Issues <https://github.com/bonfus/undi/issues>`_
-------------------------------------------------

submit `issues`_ regarding the code and examples.
   


