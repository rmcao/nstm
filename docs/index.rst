.. Neural Space-Time Model documentation master file, created by
   sphinx-quickstart on Tue May 21 16:34:26 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to nstm documentation!
==============================

.. raw:: html

   <p align="center">
       <a style="text-decoration:none !important;" href="https://github.com/rmcao/nstm" alt="project page"><img src="https://img.shields.io/badge/Project_Page-coming_soon-blue" /></a>
       <a style="text-decoration:none !important;" href="https://www.biorxiv.org/content/10.1101/2024.01.16.575950" alt="paper link"> <img src="https://img.shields.io/badge/bioRxiv-2024.01.16.575950-b31b1b.svg?style=flat" /></a>
       <a style="text-decoration:none !important;" href="https://opensource.org/licenses/BSD-3-Clause" alt="License"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" /></a>
       <a style="text-decoration:none !important;" href="https://github.com/rmcao/nstm" alt="License"><img src="https://img.shields.io/badge/GitHub-repo-black" /></a>
   </p>

**Neural space-time model (NSTM)** is a computational image reconstruction framework that can jointly estimate the scene and its motion dynamics by modeling its spatiotemporal relationship, *without data priors or pre-training*.
It is especially useful for multi-shot imaging systems which sequentially capture multiple measurements and are susceptible to motion artifacts if the scene is dynamic.
Neural space-time model exploits the temporal redundancy of dynamic scenes. This concept, widely used in video compression, assumes that a dynamic scene evolves smoothly over adjacent timepoints.
By replacing the reconstruction matrix, neural space-time model can remove motion-induced artifacts and resolve sample dynamics, from the same set of raw measurements used for the conventional reconstruction.



.. toctree::
   installation
   try_sim
   try_dpc
   load_your_sim_data
   nstm_on_new_system
   source/modules
   :maxdepth: 2
   :caption: Contents:



Indices and tables:
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
