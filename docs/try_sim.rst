Quick start on structured illumination microscopy (SIM)
=======================================================

Option 1: Step-by-step Jupyter notebook
---------------------------------------
The example notebook can be found in ``examples/notebook-SIM.ipynb`` with dense microbead data. After :doc:`installing the dependencies<installation>`, you can run the notebook to go over the reconstruction procedure step-by-step.

This notebook is also available to run on `Google Colab <https://colab.research.google.com/drive/1rxRBrBgQgedR4DW7wITcdJVFVCqC0dcQ?usp=sharing>`_.
Please note that env setup is slightly different on Colab due to the pre-installed dependencies, and you need to follow the instruction in the Colab notebook.

Option 2: Run the python script
-------------------------------

1. Download additional data from `this Google Drive <https://drive.google.com/drive/folders/1GkjU4gFv-DswJnui4WiVChe6Lz5RBau1>`_ and place .npz files in ``examples`` folder.
2. Start running endoplasmic reticulum (ER)-labeled cell reconstruction in commandline. Replace ``er_cell`` with ``mito_cell`` for mitochondria-labeled cell data.

   .. code-block:: bash

        $ python nstm/sim3d_main.py --config er_cell

.. note::
    The ``mito_cell`` reconstruction takes ~40 minutes (slightly faster for ``er_cell``) on a single NVIDIA A6000 GPU (48GB). ``er_cell`` is also runnable on a single NVIDIA RTX 3090 GPU (24GB) when ``batch_size`` is set to 1 in the .yaml file. ``mito_cell`` requires close to 40GB GPU memory to run, as it has more image planes.

3. The reconstruction results will be saved in ``examples/checkpoint/`` folder. The 3D reconstruction volume with three timepoints (each corresponding to an illumination orientation) will be saved as ``recon_filtered.tif``, and can be viewed using Fiji_. The recovered motion map will be saved as ``motion_dense_t.npy``.

.. _Fiji: https://imagej.net/Fiji/Downloads

4. Additional reconstruction parameters are stored in ``examples/configs/er_cell.yaml`` and ``examples/configs/mito_cell.yaml``. To print the full parameter descriptions, run:

    .. code-block:: bash

        $ python nstm/sim3d_main.py --helpfull
