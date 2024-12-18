Automatic analysis
================================

This example shows how use the automatic analysis implemented in undi as a python module.

```
python -m undi --max-hdim 10000000 structure-or-magres-file
```

where `--max-hdim` specifies the maximum allowed dimension for the Hilbert space to be considered.
The code automatically collects the information on the nuclear magnetic and
quadrupolar moments and computes the list of neighbors that can fit in the
specified Hilbert space. The isotope average according to natural abundances
is also performed.

A json file named `results.json` is produced by the code and contains
both zero-field, longitudinal and transverse field results.


Approximations
--------------

In this section we describe the details of the approximations performed in the
automatic analysis.
In this approach the Celio approximation is used and reasonable defaults
for the algorithm are selected assuming a Hilbert space dimension at least :math:`10^6` large.
The set of neighboring atoms of the muon, called cluster,
is selected by considering all the isotopes
with nuclear spin grater than 0 at growing distance until
the specified Hilbert space dimensions is smaller than the user required value.


The isotope average is also approximated. For a given cluster, the number
of possible combinations for the isotopes appearing in the various
positions and their probability is easily obtained with
symbolic expansion of a polynomial function.
For example, when considering the cluster Rb2Sb7, the polynomial takes the form

.. math::
   :label: total_isotope_avg

    ({}^{85}\text{Rb} + {}^{87}\text{Rb})^{2} \times ({}^{121}\text{Sb} + {}^{123}\text{Sb})^{7}

The symbolic expansion of this equation yields numerous product terms,
each representing distinct isotopic variants of the cluster.
Substituting the isotopic abundances for Rb and Sb
into each term allows for the determination of the probability of each cluster.
The accuracy of the isotope average can now be progressively improved
by including different structures in decreasing order of probability.
This is however very time consuming in general.

For this reason, a simpler approximation is considered. Instead of considering distinct atoms
in the previous formula, only distinct species are taken into account.
This means that, in each cluster, all atoms appear with the same isotope, as in

.. math::
   :label: isotope_avg_approx

    ({}^{85}\text{Rb} + {}^{87}\text{Rb}) \times ({}^{121}\text{Sb} + {}^{123}\text{Sb})


This neglects the dependence on the geometrical distribution of different isotopes for
the same atom, but is general a good approximation to the complete isotope average.


The powder average in zero-field is performed as [#f1]_

.. math::
   :label: powder_avg

    P_{powder} = (P_x + P_y + P_z)/3

while for longitudinal and transverse field the a regular sampling of
the polar angle and the azimuthal angles is performed.
The default sampling step is :math:`\pi/7`.


.. [#f1] For more details see the Ph.D thesis of John Wilkinson, https://ora.ox.ac.uk/objects/uuid:b7c1fc6c-70f8-44f4-a7e0-6893ea7a6f61/files/dnc580n113
