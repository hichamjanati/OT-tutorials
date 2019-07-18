Optimal transport tutorials
===========================


This repo presents tutorials of Optimal transport applications using
the POT_ library.


* OT_neuroimaging: Averaging neuroimaging data on a triangulated surface using
  Optimal transport. The creation of a low resolution triangulated mesh
  is done with MNE_ software. To run this notebook, please install the latest
  version of POT (for unbalanced OT) and the dependencies:

::

    pip install git+https://github.com/rflamary/POT.git
    pip install mne nilearn numba


.. _POT: http://pot.readthedocs.io
.. _MNE: http://martinos.org/mne
