Welcome to UNDI's documentation!
================================

mUon Nuclear Dipolar Interaction is a small package to obtain the time evolution
of the muon spin polarization originating from its interaction with nuclear magnetic dipoles
in standard experimental conditions
(i.e. when thermal energy is much larger that nuclear interactions).



Muon-nuclei interaction
-----------------------

The interaction of the muon spin :math:`\mathbf{I}_{\mu}` and the :math:`N` nuclear spin :math:`\mathbf{I}_{i}`,  with the effect Electric Field Gradient (EFG) is described by the Hamiltonian

.. math::
   :label: total_hamiltonian0

   \mathcal{H} = \mathcal{H}_{Z,\mu} + \sum_{i}^{N}  \mathcal{H}_{Z,i} + \mathcal{H}_{dip,i} + \mathcal{H}_{Q,i} + \sum_{i,j>i}^N \mathcal{H}_{dip,ij}

where

.. math::
   :label: total_hamiltonianmuEXT

   \mathcal{H}_{Z,\mu} = -\hbar \gamma_{\mu} \mathbf{I}_{\mu}\cdot\mathbf{B}_{ext}

.. math::
   :label: total_hamiltonianIEXT

   \mathcal{H}_{Z,i} = -\hbar \gamma_{i} \mathbf{I}_{i}\cdot\mathbf{B}_{ext}

.. math::
   :label: total_hamiltonianDIP

   \mathcal{H}_{dip,i} = \frac{\mu_0 \hbar^2 }{4\pi}\gamma_{i}\gamma_{\mu} \left( \frac{\mathbf{I}_{i}\cdot\mathbf{I}_{\mu}}{r^3} -  \frac{3(\mathbf{I}_{i}\cdot\mathbf{r})(\mathbf{I}_{\mu}\cdot\mathbf{r})}{r^5} \right)

.. math::
   :label: total_hamiltonianQUAD

   \mathcal{H}_{Q,i} = \frac{eQ_{i}}{6I_{i}(2I_{i}-1)} \sum_{{\alpha},{\beta}{\in}\{{x,y,z}\}} V_{i}^{{\alpha}{\beta}} \left[ \frac{3}{2}\left( I_{i}^{\alpha}I_{i}^{\beta} - I_{i}^{\beta} I_{i}^{\alpha}  \right) - \delta_{{\alpha}{\beta}}I_{i}^{2} \right]


.. math::
   :label: total_hamiltonianDIPIJ

   \mathcal{H}_{dip,ij} = \frac{\mu_0 \hbar^2 }{4\pi} \gamma_{i} \gamma_{j} \left(\frac{\mathbf{I}_{i}\cdot\mathbf{I}_{j}}{r_{ij}^3} - \frac{3(\mathbf{I}_{i}\cdot\mathbf{r}_{ij})(\mathbf{I}_{j}\cdot\mathbf{r}_{ij})}{r_{ij}^5} \right)



:math:`\mathcal{H}_{Z,j} ~ j \in \{i,\mu\}` is the Zeeman interaction for each nucleus :math:`i`  or the muon :math:`\mu` subject to the external field :math:`\mathbf{B}_{ext}`.
:math:`\mathcal{H}_{dip,i}` is the dipolar interaction between the muon and the nuclei and :math:`\mathcal{H}_{Q,i}` is the quadrupolar interaction,
with :math:`V_i` being the EFG at nuclear site i.


Installalation
--------------
UNDI requires NumPy, QuTip_ and Mendelev_ Python packages.
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

The input is defined by a list of dictionaries. The details for the
muon and the nuclei can be set with the following keys:

- **Position**  : 3D vector position of the nuclei, muon expressed in Cartesian Coordinates.

- **Label**  :  A string defining the nucleus or the muon. Can be either the element or the isotope name. The label 'mu' must always be introduced to define the muon details.

- **Spin**  : Nuclear spin (optional), the mendelev package is used to obtain the spin value using the information in **Label**,

- **Gamma** :  Gyromagnetic ratio (optional). Same as above,

- **ElectricQuadrupoleMoment**  : Nuclear quadrupole moment for the nucleus (optional for :math:`\mathbf{I} > 1/2`),

- **EFGTensor**  : 3D array with the EFG tensor expressed in Cartesian coordinates (optional).


Celio's approximated approach
-----------------------------

The description provided below follows the discussion originally presented by Celio_.
In general the muon depolarization is given by:

.. math::
   :label: density_matrix

   \boldsymbol{\mathcal{P}}_{\mu}(t) = \textrm{Tr}\left[ \rho e^{(i\mathcal{H}t/\hbar)} \boldsymbol{{\sigma}}_{\mu} e^{(-i\mathcal{H}t/\hbar)} \right]

where  :math:`\boldsymbol{{\sigma}}_{\mu} = 2\mathbf{I}_{\mu}` are the Pauli operators and :math:`\rho` is the density matrix.

In the Hamiltonian :math:`\mathcal{H}` of Eq. :eq:`total_hamiltonian0` the operators :math:`\mathcal{H}_{dip,i}` do not commute with each other and an analytical solution for :math:`\boldsymbol{\mathcal{P}}_{\mu}(t)` is, in general, not possible.

However, one can expand the exponentials in Eq. :Eq:`density_matrix` in to sums of simple block diagonal terms by using Trotter formula for bounded operators.
For example, for a set of two operators :math:`\mathcal{H}_1` and :math:`\mathcal{H}_2` a Trotter expansion can be written as

.. math::
   :label: trotter_twoOP

   e^{ \mathcal{H}_1 + \mathcal{H}_2 } = \lim_{n \to \infty}\left( e^{\frac{\mathcal{H}_1}{n}} e^{\frac{\mathcal{H}_2}{n}} \right)^n

Eq. :Eq:`density_matrix` can be rewritten as:

.. math::
   :label: density_matrix_2

   \boldsymbol{\mathcal{P}}_{\mu}(t) = \textrm{Tr}\left[ \rho\boldsymbol{{\sigma}}_{\mu}(t)\right]

where

.. math::
   :label: sigma_matrix

   \boldsymbol{{\sigma}}_{\mu}(t) =  e^{(i\mathcal{H}t/\hbar)} \boldsymbol{{\sigma}}_{\mu}e^{(-i\mathcal{H}t/\hbar)}

Choosing a representation where :math:`\sigma_{\mu}^z` is diagonal, Eq. :Eq:`density_matrix_2` can be written in Schrodinger picture as:

.. math::
   :label: polarization

   \boldsymbol{\mathcal{P}}(t) = \sum_{n}^{d_{\mathcal{H}}} w_{n}  \left\langle \psi_{n}(t)|\boldsymbol{{\sigma}}_{\mu}|\psi_{n}(t)\right\rangle

where

.. math::
   :label: psi_t

   | {\psi_{n}(t)} \rangle= e^{-i\mathcal{H}t/\hbar} | {\psi_{n}(0)} \rangle

The coefficient :math:`w_n` can be described as the probability of finding the spin system, in the pure state :math:`| {\psi_{n}(0)} \rangle` at time :math:`t=0` and,
for a fully spin polarized muon in standard experimental conditions (nuclear magnetic moments not ordered), they are

.. math::
   :label: w_n

    w_n = \frac{2}{d_{\mathcal{H}}}  & \quad \textrm{if} \quad \boldsymbol{{\sigma}}_{\mu}^{z} | {\psi_{n}(0)} \rangle = + | {\psi_{n}(0)} \rangle \\
    w_n =      0       & \quad \textrm{if} \quad  \boldsymbol{{\sigma}}_{\mu}^{z} | {\psi_{n}(0)} \rangle = - | {\psi_{n}(0)} \rangle \\


Eq. :Eq:`total_hamiltonian0` can also be rewritten as

.. math::
   :label: H_celio

    \mathcal{H} =  \sum _{i} ^{N} \frac{\mathcal {H}_{\mathrm Z, \mu}}{N} + \mathcal {H}_{\mathrm Z, i} + \mathcal {H}_{\mathrm dip, i} + \mathcal {H}_{\mathrm Q, i}  = \sum _{i} ^{N} \mathcal {H}_{i} \label{eq:HCelio}

if the dipolar interaction between the nuclei can be neglected.

:math:`\mathcal{H}_i` acts in a small subspace of dimension :math:`2(2I+1)` thus the problem of diagonalizing a huge :math:`d_{\mathcal{H}}\times d_{\mathcal{H}}` matrix has been circumvented.
The Trotter expansion of the evolution operator is

.. math::
   :label: trotter_all

   e^{-i\mathcal{H}t/\hbar} = \lim_{k \to \infty} \left[ \prod_{i=1}^{N}e^{\left(-i\frac{\mathcal{H}_i}{k}\frac{t}{\hbar}\right)} \right]^{k}

and Eq. :Eq:`psi_t` becomes

.. math::
   :label: psi_with_trotter

   | {\psi_{n}(t)} \rangle & = e^{-i\mathcal{H}t/\hbar} | {\psi_{n}(0)} \rangle  \\

                           & = \lim_{k \to \infty} \left[ \prod_{i=1}^{N}e^{\left(-i\frac{\mathcal{H}_i}{k}\frac{t}{\hbar}\right)} \right]^{k} | {\psi_{n}(0)} \rangle  \\


Where the value of :math:`k` is chosen to make the depolarization independent of the expansion parameter in the experimental time window.

The matrix elements :math:`\left\langle \psi_{n}(t)|\boldsymbol{{\sigma}}_{\mu}|\psi_{n}(t)\right\rangle` in Eq. :Eq:`polarization` can be calculated, but this would be a time consuming operation.

Let's consider instead the random superposition of states at :math:`t=0` given by

.. math::
   :label:

   | {\phi(0)} \rangle = \sum_{l=1}^{d_{\mathcal{H}}/2}  \sqrt{\frac{2}{d_{\mathcal{H}}}} e^{\lambda_l} | {\psi_{l}(0)} \rangle

where :math:`\lambda_l` is taken at random in the interval :math:`[0, 2 \pi)`.

At a later time :math:`t` ,  :math:`|\phi(t)\rangle` is:

.. math::
   :label: random_phase

    | {\phi(t)} \rangle = \sum_{l=1}^{d_{\mathcal{H}}/2} \sqrt{\frac{2}{d_{\mathcal{H}}}} e^{i\lambda_{l}} | {\psi_{l}(t)} \rangle

and the expactation value of :math:`\boldsymbol{{\sigma}}_{\mu}` is:


.. math::
   :label: random_phase2

   \langle {\phi(t)} | \boldsymbol{{\sigma}}_{\mu} | {\phi(t)} \rangle = \sum_{l=1}^{d_{\mathcal{H}}/2} \left( \frac{2}{d_{\mathcal{H}}} \right) \langle {\psi_{l}(t)} | \boldsymbol{{\sigma}}_{\mu} | {\psi_{l}(t)} \rangle +\sum_{l,n=1, l \ne n}^{d_{\mathcal{H}}/2} \left( \frac{2}{d_{\mathcal{H}}} \right) e^{i(\lambda_{l}-\lambda_{n})} \langle {\psi_{l}(t)} | \boldsymbol{{\sigma}}_{\mu} | {\psi_{n}(t)} \rangle


The first term on the right-hand side of Eq. :Eq:`random_phase2` is the exact solution of Eq. :Eq:`polarization` while second term tends to zero in a limit of large :math:`d_{\mathcal{H}}`. In order to obtain an exact signal :math:`\boldsymbol{\mathcal{P}}(t)` , one has to calculate the above matrix element over various random initials states and average out the unwanted additional contribution.

The function `MuonNuclearInteraction.celio()` implements the method described above and
its usage is explained with a number of examples discussed below.


Some important conventions
--------------------------

Before you start playing with UNDI through examples, a few important information:

* UNDI uses SI units, which might seem an unfortunate choice
  (you end up writing tiny numbers) but does not hurt too much.

* UNDI assumes that the spin polarization is along :math:`z` and
  that observation is done along the same direction.
  You must rotate the sample definition accordingly (there is
  a function to do that). Initial polarization along arbitrary
  directions is presently implemented only in Celio's method.

* As a consequence, in general, an external field applied along :math:`z`
  is a Longitudinal Field (LF) while external fields in the plane
  perpendicular to :math:`z` are Transverse Fields (TF).

Using UNDI
----------

The best way to start with UNDI is to explore some illustrative examples
that reproduce some results already published in literature.
Alternatively you can check the detailed function documentation.

.. toctree::
   :maxdepth: 1

   examples/examples
   code


.. _QuTip: http://qutip.org
.. _Mendelev: https://github.com/lmmentel/mendeleev
.. _Celio: https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.56.2720
