.. UNDI documentation master file, created by
   sphinx-quickstart on Wed Jan 15 07:48:14 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to UNDI's documentation!
================================

mUon Nucleai Dipolar Interaction is a small package to obtain muon
depolarization originating from nuclear dipoles in standard experimental
conditions (i.e. when the nuclear density matrix can be assumed to be the
identity matrix).

Before you start
----------------

Before you start playing with UNDI through examples, a few important information:

* UNDI uses SI units, which soetimes might be strange 
  (you end up adding many 1e-10) but does not hurt too much.

* UNDI assumes that the spin polarization is along :math:`z` and that observation is done
  along the same direction. You must rotate the same according to match this
  condition with your experimental setting.

* As a consequence, an external field applied along :math:`z` is a Longitudinal Field (LF)
  while external fields in the plane perpendicular to :math:`z` are Transverse Fields (TF).


Using UNDI
----------

The documentation, for the time being, is provided as a set of examples
that reproduce results published in literature.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   examples/examples


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
