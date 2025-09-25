# h-emcee
Implementation of affine invariant Hamiltonian samplers [Y. Chen (2025)](https://arxiv.org/abs/2505.02987) in JAX + affine invariant schemes for tuning hyper-parameters of Hamiltonian samplers. The syntax of this package is meant as a minimal replacement for [emcee](https://github.com/dfm/emcee).

# Installation
I recommend making a virtual environment, and installing it (via pip) into said environment.
```
python -m pip install h-emcee
```

# Documentation
Documentation is available at https://h-emcee.readthedocs.io

# Todo
[ ] Implement affine invariant ChEES algorithm. This is better than NUTs (interms of control flow).