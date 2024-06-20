.. _installation-ref-label:

Installation
============

Prerequisites
-------------
- High performance linux workstation or cluster
- NVIDIA GPU
- `Anaconda <https://www.anaconda.com/products/individual>`__ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__

Step-by-step Installation
-------------------------
1. Create a conda virtual environment and activate it

   .. code-block:: bash

      $ conda create -n nstm python=3.9
      $ conda activate nstm

2. Clone this project to your local machine. Or download the zip file and unzip it.

   .. code-block:: bash

      $ git clone https://github.com/rmcao/nstm.git

3. Install CUDA and cuDNN in conda virtual env (you may opt to skip this step if you have CUDA installed in your system and you know what you are doing)

   .. code-block:: bash

      $ conda install -c conda-forge cudatoolkit~=11.8.0 cudnn~=8.8.0
      $ conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc

4. Install jaxlib. Note that the following command is for CUDA 11.x and cuDNN 8.2+. If you have different versions of CUDA, please refer to `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`__ and make sure to match the version numbers of jaxlib and jax (as specified in requirements.txt).

   .. code-block:: bash

      $ pip install jaxlib==0.3.18+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

5. Install optional dependencies for interactive visualization via Jupyter lab

   .. code-block:: bash

      $ conda install -c conda-forge jupyterlab nodejs ipympl

6. Install the helper library and this codebase

   .. code-block:: bash

      $ pip install git+https://github.com/rmcao/CalCIL.git
      $ pip install -e ./nstm

7. Test the installation

   .. code-block:: bash

      $ python -c "import jax.numpy as jnp; print(jnp.ones(5)+jnp.zeros(5))"