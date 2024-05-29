Using your own 3D SIM data
==========================

Prerequisites
-------------
- 3D SIM raw images
- Timestamp for the acquisition of each raw image
- OTF (optical transfer function) for the 3D SIM system
- System parameters (e.g. pixel size, z-step size, etc.)

.. note::
    Only three-beam SIM data is supported at this time. Other SIM configurations can be supported by modifying the forward model in :py:class:`nstm.sim3d_flow.FluoSIM3D`.

Data preparation
----------------
The simplest way to use your own data with NSTM reconstruction is to save your raw images, OTF, timestamps into a single .npz file. The .npz file should contain the following keys:

- 'img': 5D numpy array of raw images (rotation, phase, z, y, x).
- 'OTF': 4D numpy array of OTF (rotation, fr, fz, band). Here we assume radially averaged OTF, which can be generated through `makeotf script`_ in cudasirecon software from experimentially measured PSF. OTF for each rotation angle can be specified through the first dimension (or just repeat the same OTF on each rotation).
- 'timestamp_phase': 3D numpy array of timestamps for raw images. In fast mode (default), it is saved as (rotation, phase, 1). The fast mode assumes that raw images with the same sinusodial rotation and phase at different depth are taken at the same timepoint. In non-fast mode (not recommended), it is saved as (rotation, phase, z).

.. note::
    The timestamps have to be normalized into [-1, 1].
.. _`makeotf script`: https://github.com/scopetools/cudasirecon/tree/master?tab=readme-ov-file#formakeotf

.. warning::
    Our script assumes that the raw images have already been oversampled by ``zoomfact`` [#]_ in the x and y directions, and the OTF has NOT been oversampled.

We provide an example script to convert the raw images, OTF, and timestamps into a .npz file. The script can be found in ``nstm/commands/process_raw_images``. The script can be run as follows:

    .. code-block:: bash

        $ python nstm/commands/process_raw_images.py --raw_path path/to/raw_images --otf_path path/to/otf --timestamp_path path/to/timestamp --save_path path/to/save

.. [#] zoomfact is specified in the .yaml config file. Oversampling is to make sure the super-resolved reconstruction not pixel-limited.

Create .yaml config file
------------------------
System parameters and other settings can be specified in a .yaml file for each reconstruction. The data.npy file should be specified in the ``raw_path`` field. The full arguments can be found in :py:meth:`nstm.sim3d_utils.define_flags`.

An example of the .yaml file is shown below::

    raw_path: 'path/to/data.npz'
    save_path: 'path/to/save'
    line_spacing: [0.419, 0.419, 0.419] # grating line spacing in um
    k0angles: [1.843, 2.960, -2.315]    # k0 angles in rad
    na: 1.28
    ri_medium: 1.33
    wavelength: 0.515                   # emission wavelength in um
    ps: 0.040625                        # pixel size in um (before oversampling by zoomfact)
    ps_otf: 0.040625                    # pixel size in um for OTF
    dz: 0.15                            # z-step size in um
    dz_otf: 0.1                         # z-step size in um for OTF
    zoomfact: 1.6                       # oversampling factor on x and y dimensions


Running NSTM reconstruction
---------------------------

To run the reconstruction, use the following command:

    .. code-block:: bash

        $ python nstm/sim3d_main.py --config path/to/config.yaml

NSTM results are saved in the specified path/to/save directory:

- ``recon_filtered.tif``: the reconstructed volume at each rotation. It is recommended to use FIJI/ImageJ to open the tiff file.
- ``recon_dense_t.npy``: the reconstructed volume at all timepoints (all rotations and phases). The shape of the array is (z, rotation, phase, y, x).
- ``motion_dense_t.npy``: the motion map estimated by motion net at all timepoints (all rotations and phases). The shape of the array is (z, rotation, phase, y, x, 3). Note that since the motion at ``t=0`` might not be zero, it is recommended to subtract the motion at ``t=0`` from the motion map.

Computation
-----------

NSTM is rather computationally intensive. The reconstruction time depends on the dimension of the raw images and number of z planes,
and the reconstructed volume dimension is also limited by the GPU memory. It is recommended to start with a volume smaller than 512x512x12 (x, y, z) for a GPU with 24GB memory.
Patch-wise reconstruction can be used to reconstruct larger volumes.
