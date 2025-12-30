Differentiable COSMO-Type Activity Coefficient Layer
====================================================

[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/craabreu/cosmolayer/workflows/Linux/badge.svg)](https://github.com/craabreu/cosmolayer/actions?query=workflow%3ALinux)
[![GitHub Actions Build Status](https://github.com/craabreu/cosmolayer/workflows/MacOS/badge.svg)](https://github.com/craabreu/cosmolayer/actions?query=workflow%3AMacOS)
[![GitHub Actions Build Status](https://github.com/craabreu/cosmolayer/workflows/Windows/badge.svg)](https://github.com/craabreu/cosmolayer/actions?query=workflow%3AWindows)
[![GitHub Actions Build Status](https://github.com/craabreu/cosmolayer/workflows/Linter/badge.svg)](https://github.com/craabreu/cosmolayer/actions?query=workflow%3ALinter)
[![Documentation Status](https://github.com/craabreu/cosmolayer/workflows/Docs/badge.svg)](https://github.com/craabreu/cosmolayer/actions?query=workflow%3ADocs)
[![Coverage Report](https://craabreu.github.io/cosmolayer/development/coverage/coverage.svg)](https://craabreu.github.io/cosmolayer/development/coverage)

[![Conda version](https://img.shields.io/conda/v/mdtools/cosmolayer.svg)](https://anaconda.org/mdtools/cosmolayer)
[![Conda platforms](https://img.shields.io/conda/pn/mdtools/cosmolayer.svg)](https://anaconda.org/mdtools/cosmolayer)
[![Conda downloads](https://img.shields.io/conda/dn/mdtools/cosmolayer.svg)](https://anaconda.org/mdtools/cosmolayer)

[![PyPI version](https://img.shields.io/pypi/v/cosmolayer.svg)](https://pypi.org/project/cosmolayer)
[![PyPI version](https://img.shields.io/pypi/pyversions/cosmolayer.svg)](https://pypi.org/project/cosmolayer)
[![PyPI version](https://img.shields.io/pypi/dm/cosmolayer.svg)](https://pypi.org/project/cosmolayer)

[![License](https://img.shields.io/badge/License-MIT-yellowgreen.svg?style=flat)](https://github.com/craabreu/cosmolayer/blob/main/LICENSE.md)

### Overview

CosmoLayer is a package implementing differentiable COSMO-type activity coefficient calculation layers for neural network models.

CosmoLayer leverages automatic differentiation and GPU acceleration to enable efficient computation and gradient-based optimization of COSMO model parameters.

### Installation and Usage

CosmoLayer is available as a conda package on the [mdtools] channel. To install it, run:

```bash
    conda install -c conda-forge -c mdtools cosmolayer
```

Or:

```bash
    mamba install -c mdtools cosmolayer
```

To use CosmoLayer in your own Python script or Jupyter notebook, simply import it as follows:

```python
    import cosmolayer
```

### Documentation

Documentation for the latest CosmoLayer version is available at [Github Pages].

### Copyright

Copyright (c) 2026 [Charlles Abreu](https://github.com/craabreu)


[Github Pages]: https://craabreu.github.io/cosmolayer/latest
[mdtools]: https://anaconda.org/mdtools/cosmolayer
