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

Installation from source
------------------------

Clone the package from the source:

.. code-block:: bash

    git clone https://github.com/clarkmiyamoto/h-emcee.git

Then use :code:`pip` to install:

.. code-block:: bash

    cd h-emcee
    python -m pip install jax
    python -m pip install -e .

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
    python run_tests.py