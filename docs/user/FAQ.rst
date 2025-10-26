FAQ
==========================


I got an Out Of Memory (OOM) error, what can I do?
--------------------------------------------

The :code:`Backend` class allows you to write directly to 
storage using :code:`HDF5`. 

.. code-block:: python

    import hemcee
    backend = hemcee.backend.HDFBackend(filename = <FILENAME>)
    hemcee.HamiltonianEnsembleSampler(..., backend=backend)