.. UNDI documentation master file, created by
   sphinx-quickstart on Wed Jan 15 07:48:14 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to UNDI's documentation!
================================

mUon Nuclear Dipolar Interaction is a small package to obtain the time evolution
of the muon spin polarization originating from its interaction with nuclear magnetic dipoles
in standard experimental conditions
(i.e. when thermal energy is much larger that nuclear interactions).



Muon-nuclei interaction
-----------------------

The interaction of the muon spin :math:`\mathbf{I}_{\mu}` and the :math:`N` nuclear spin :math:`\mathbf{I}_{i}`,  with the effect Electric Field Gradient (EFG) is described by the hamiltonian

.. math::
   :label: total_hamiltonian0

   \mathcal{H} = \mathcal{H}_{Z,\mu} + \sum_{i}^{N} \left[ \mathcal{H}_{Z,i} + \mathcal{H}_{dip,i} + \mathcal{H}_{Q,i} \right]

where

.. math::
   :label: total_hamiltonianmuEXT

   \mathcal{H}_{Z,\mu} = -\hbar \gamma_{\mu} \mathbf{I}_{\mu}\cdot\mathbf{B}_{ext}

.. math::
   :label: total_hamiltonianIEXT

   \mathcal{H}_{Z,i} = -\hbar \gamma_{i} \mathbf{I}_{i}\cdot\mathbf{B}_{ext}

.. math::
   :label: total_hamiltonianDIP

   \mathcal{H}_{dip,i} = \frac{\mu_0}{4\pi}\gamma_{i}\gamma_{\mu} \left( \frac{\mathbf{I}_{i}\cdot\mathbf{I}_{\mu}}{r^3} -  \frac{3(\mathbf{I}_{i}\cdot\mathbf{r})(\mathbf{I}_{\mu}\cdot\mathbf{r})}{r^5} \right)

.. math::
   :label: total_hamiltonianQUAD

   \mathcal{H}_{Q,i} = \frac{eQ_{i}}{6I_{i}(2I_{i}-1)} \sum_{{\alpha},{\beta}{\in}\{{x,y,z}\}} V_{i}^{{\alpha}{\beta}} \left[ \frac{3}{2}\left( I_{i}^{\alpha}I_{i}^{\beta} - I_{i}^{\beta} I_{i}^{\alpha}  \right) - \delta_{{\alpha}{\beta}}I_{i}^{2} \right]


:math:`\mathcal{H}_{Z,j} ~ j \in \{i,\mu\}` is the Zeeman interaction for each nucleus :math:`i`  or the muon :math:`\mu` subject to the external field :math:`\mathbf{B}_{ext}`.
:math:`\mathcal{H}_{dip,i}` is the dipolar interaction between the muon and the nuclei and :math:`\mathcal{H}_{Q,i}` is the quadrupolar interaction,
with :math:`V_i` being the EFG at nuclear site i.


Installalation
--------------
UNDI requires both NumPy and QuTip_.
At the moment you can install the first version using

::

  pip install undi

or using the repository with

::

  git clone https://github.com/bonfus/undi.git undi
  cd undi
  python setup.py --user install



Input
-----

The input is defined by the following data:

- **Position**  : 3D vector position of the nuclei, muon expressed in Cartesian Coordinates.

- **Label**  :  A string use to define the type of the nuclei, it is Isotope. A default label of 'mu' is always used to define muon.

- **Spin**  : Nuclear spin (optional), A python mendelev package is used to define the spin value if **Label** of the nuclei is correctly define (see example below),

- **Gamma** :  Gyromagnetic ratio (optional). Same as above,

- **ElectricQuadrupoleMoment [Optional]**  : Nuclear quadrupole moment for nuclei spin :math:`\mathbf{I}_{i}` , **Spin** > :math:`1/2`

- **EFGTensor [Optional]**  : 3D EFG tensor expressed in Cartesian coordinates

All the above parameters are accepted as input list of python dictionaries as shown in the examples.

Some important conventions
--------------------------

Before you start playing with UNDI through examples, a few important information:

* UNDI uses SI units, which might seem an unfortunate choice
  (you end up writing tiny numbers) but does not hurt too much.

* UNDI assumes that the spin polarization is along :math:`z` and
  that observation is done along the same direction.
  You must rotate the sample definition accordingly (there is
  a function to do that). Initial polarization along arbitraty
  directions is presently implemented only in Celio's method.

* As a consequence, in genrale, an external field applied along :math:`z`
  is a Longitudinal Field (LF) while external fields in the plane
  perpendicular to :math:`z` are Transverse Fields (TF).

Using UNDI
----------

The best way to start with UNDI is to explore some illustrative examples
that reproduce some results already published in literature.


.. toctree::
   :maxdepth: 2

   examples/examples


.. _QuTip: http://qutip.org
