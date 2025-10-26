Installation
============

It is recommended to install :code:`h-emcee` into a virtual environment 
(for example, using :code:`conda`).

Installation using :code:`pip`
------------------------------

For a quick start, use :code:`pip` to install:

.. code-block:: bash

    python -m pip install h-emcee

The backend uses :code:`JAX`, which is installed automatically with :code:`pip`. 
**For GPU users**, this will auto-install a CPU-only version of :code:`JAX`. 
For an installation with GPU support, please see https://docs.jax.dev/en/latest/installation.html 

For GPU users, install the latest version of :code:`JAX` with GPU support:Installation using :code:`pip`
------------------------------

Installation from source
------------------------

.. code-block:: bash

    git clone https://github.com/clarkmiyamoto/hemcee.git
    cd hemcee
    pip install -e .

The :code:`-e` (“editable”) install is handy if you're iterating on the code locally.

Developer Installation
----------------------

It's recommended to make a fork of the repo, 
clone the fork, and install the package in editable mode with the dev extras.

.. code-block:: bash

    pip install -e ".[dev]"

Verify Installation
-------------------

You can verify the installation is working by running:

.. code-block:: python

    python
    >>> import hemcee
    >>> print(hemcee.__version__)

You can also verify installation is working as intended by running the test suite.
We use :code:`pytest`:

.. code-block:: bash

    python -m pip install pytest

After installing, run all tests:

.. code-block:: bash

    cd tests
    python -m pytest