Getting Started
===============

Introduction
------------

CosmoLayer is a package implementing differentiable COSMO-type activity coefficient calculation layers for neural network models.

CosmoLayer leverages automatic differentiation and GPU acceleration to enable efficient computation and gradient-based optimization of COSMO model parameters.

Installation
------------

To install CosmoLayer in a conda environment, run the following command::

    conda install -c conda-forge -c mdtools cosmolayer

Or use mamba instead::

    mamba install -c mdtools cosmolayer

Usage
-----

To use CosmoLayer, import the package in your Python script or Jupyter notebook::

    import cosmolayer

Basic usage example (placeholder for future implementation)::

    # Compute sigma profile for a molecule
    # sigma_profile = cosmolayer.compute_sigma_profile(molecule)
    
    # Calculate activity coefficients
    # activity_coeffs = cosmolayer.cosmo_sac(sigma_profiles, temperature)

