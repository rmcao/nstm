Quick start on differential phase contrast microscopy (DPC)
===========================================================

The example notebook for DPC can be found in ``examples/notebook-DPC.ipynb`` with dense microbead data.
After :doc:`installing the dependencies<installation>`, you can run the notebook to go over the reconstruction procedure step-by-step.

    .. code-block:: bash

        $ jupyter lab --notebook-dir=./nstm/examples

This notebook is also available to run on `Google Colab <https://colab.research.google.com/drive/1rxRBrBgQgedR4DW7wITcdJVFVCqC0dcQ?usp=sharing>`_. Please note that env setup is slightly different on Colab due to the pre-installed dependencies, and you need to follow the instruction in the Colab notebook.

Using your own data
-------------------
You can simply replace the data and illumination patterns ``s`` used in the notebook with your own raw data and source patterns.
This NSTM reconstruction assumes images *after background normalization* (so that pixel values are centered around zero).
You may test with the conventional Tikhonov reconstruction (included in the notebook) first as a sanity check, and then proceed with the space-time reconstruction.