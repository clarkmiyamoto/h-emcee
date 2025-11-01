import os
import h5py
import numpy as np

import jax
import jax.numpy as jnp

from hemcee.backend.backend import Backend

__version__ = 0

class HDFBackend(Backend):
    """A backend that stores the chain in an HDF5 file using h5py

    .. note:: You must install `h5py <http://www.h5py.org/>`_ to use this
        backend.

    Args:
        filename (str): The name of the HDF5 file where the chain will be saved.
        name (str): The name of the group where the chain will be saved.
        read_only (bool): If ``True``, the backend will throw a
            ``RuntimeError`` if the file is opened with write access.
        dtype (jax.numpy.dtype): JAX NumPy data type. Defaults to JAX's default of float32, or data type of read in data.
        device (jax.Device): Which device to read & write data on. Defaults to cpu
        swmr (bool): Single writer, multiple readers for HDF5 file. If `True` allows others to read file while writer is working.
    """

    def __init__(
        self,
        filename: str,
        name: str = "mcmc",
        read_only: bool = False,
        dtype: jnp.dtype = None,
        device: jax.Device = None,
        compression = None,
        compression_opts = None,
    ):
        if h5py is None:
            raise ImportError("you must install 'h5py' to use the HDFBackend")
        self.filename = filename
        self.name = name
        self.read_only = read_only
        if dtype is None:
            self.dtype_set = False
            self.dtype = jnp.float32
        else:
            self.dtype_set = True
            self.dtype = dtype
        if device is None:
            self.device = jax.devices('cpu')[0]
        else:
            self.device = device
        self.compression = compression
        self.compression_opts = compression_opts

        self.h5py_libversion = 'earliest'
    
    @property
    def initialized(self):
        if not os.path.exists(self.filename):
            return False
        try:
            with self.open() as f:
                return self.name in f
        except (OSError, IOError):
            return False
        
    def open(self, mode: str = "r") -> h5py.File:
        if self.read_only and mode != "r":
            raise RuntimeError(
                "The backend has been loaded in read-only "
                "mode. Set `read_only = False` to make "
                "changes."
            )
        
        # Single write, multiple read
        f = h5py.File(self.filename, mode, libver=self.h5py_libversion)

        # Construct open
        if not self.dtype_set and self.name in f:
            g = f[self.name]
            if "chain" in g:
                self.dtype = g["chain"].dtype
                self.dtype_set = True
        return f
    
    def reset(self, nwalkers, ndim):
        """Clear the state of the chain and empty the backend

        Args:
            nwakers (int): The size of the ensemble
            ndim (int): The number of dimensions

        """
        with self.open("a") as f:
            if self.name in f:
                del f[self.name]

            g = f.create_group(self.name)
            g.attrs["version"] = __version__
            g.attrs["nwalkers"] = nwalkers
            g.attrs["ndim"] = ndim
            g.attrs["iteration"] = 0
            g.attrs["iteration_warmup"] = 0
            g.attrs["iteration_main"] = 0
            g.attrs["warmup_end_index"] = -1  # -1 means warmup not yet complete
            g.create_dataset(
                "chain",
                (0, nwalkers, ndim),
                maxshape=(None, nwalkers, ndim),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
            g.create_dataset(
                "log_prob",
                (0, nwalkers),
                maxshape=(None, nwalkers),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
            g.create_dataset(
                "accepted",
                data=jnp.zeros(nwalkers, device=self.device),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
    
    def get_value(self, 
                  key: str, 
                  thin: int = 1, 
                  discard: int = 0):
        '''
        Gets values associated w/ associated HDF file.

        Args:
            key (str): 
            thin (int):
            discard (int): 
        '''
        if not self.initialized:
            raise AttributeError(
                "You must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )
        with self.open() as f:
            g = f[self.name]

            # Check if sampler has been ran
            iteration = g.attrs["iteration"]
            if iteration <= 0:
                raise AttributeError(
                    "You must run the sampler with "
                    "'store == True' before accessing the "
                    "results"
                )

            # Get results for specified `key`
            v = g[key][discard + thin - 1 : self.iteration : thin]
            return v


    @property
    def shape(self):
        with self.open() as f:
            g = f[self.name]
            return g.attrs["nwalkers"], g.attrs["ndim"]

    @property
    def iteration(self):
        with self.open() as f:
            return f[self.name].attrs["iteration"]
    
    @property
    def iteration_warmup(self):
        with self.open() as f:
            return f[self.name].attrs["iteration_warmup"]
    
    @property
    def iteration_main(self):
        with self.open() as f:
            return f[self.name].attrs["iteration_main"]

    @property
    def accepted(self):
        with self.open() as f:
            return f[self.name]["accepted"][...]

    def save_slice(self, 
                   coords: jnp.ndarray, 
                   log_prob: jnp.ndarray, 
                   accepted: jnp.ndarray, 
                   index: int):
        """Save a slice of a JAX array to HDF5 backend

        Args:
            coords (jnp.ndarray): Coordinates array of shape (nsteps, nwalkers, ndim)
            log_prob (jnp.ndarray): Log probability array of shape (nsteps, nwalkers)
            accepted (jnp.ndarray): Accepted array of shape (nwalkers,)
            index (int): Starting index for this slice in the full chain
        """
        with self.open("a") as f:
            g = f[self.name]
            
            # Get current dimensions
            nsteps, nwalkers, ndim = coords.shape
            
            # Resize datasets if necessary
            current_size = g["chain"].shape[0]
            new_size = max(current_size, index + nsteps)
            
            if new_size > current_size:
                g["chain"].resize((new_size, nwalkers, ndim))
                g["log_prob"].resize((new_size, nwalkers))
            
            # Write the data to storage
            g["chain"][index:index+nsteps] = coords
            g["log_prob"][index:index+nsteps] = log_prob
            g["accepted"][:] = g["accepted"][:] + accepted
            
            # Update iteration counter
            g.attrs["iteration"] = index + nsteps
            
            # Update warmup/main counters based on warmup_end_index
            warmup_end_idx = g.attrs["warmup_end_index"]
            if warmup_end_idx < 0:
                # Warmup not completed yet, count everything as warmup
                g.attrs["iteration_warmup"] = g.attrs["iteration"]
                g.attrs["iteration_main"] = 0
            else:
                # Warmup has been marked as complete
                g.attrs["iteration_warmup"] = min(g.attrs["iteration"], warmup_end_idx)
                g.attrs["iteration_main"] = max(0, g.attrs["iteration"] - warmup_end_idx)
    
    def mark_warmup_end(self):
        """Mark the current iteration as the end of warmup phase."""
        with self.open("a") as f:
            g = f[self.name]
            g.attrs["warmup_end_index"] = g.attrs["iteration"]

if __name__ == '__main__':
    nwalkers = 128
    ndim = 4
    backend = HDFBackend(filename='trial.h5', name='testing_hdf_backend', swmr=True)
    print(f"Backend initialized: {backend.initialized}")
    
    # Reset the backend
    backend.reset(nwalkers, ndim)
    print(f"Backend reset. Initialized: {backend.initialized}")
    
    # Create some test data
    nsteps = 10
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    coords = jax.random.normal(key1, (nsteps, nwalkers, ndim))
    log_prob = jax.random.normal(key2, (nsteps, nwalkers)) - 10
    accepted = jax.random.bernoulli(key3, 0.5, (nwalkers,))
    
    print(f"Test data shapes: coords={coords.shape}, log_prob={log_prob.shape}, accepted={accepted.shape}")
    
    # Test save_slice
    try:
        backend.save_slice(coords, log_prob, accepted, 0)
        print("✓ save_slice completed successfully")
        
        # Verify the data was saved
        with backend.open() as f:
            print(f"SWMR Mode: {f.swmr_mode}")
            g = f[backend.name]
            print(f"Chain shape: {g['chain'].shape}")
            print(f"Log prob shape: {g['log_prob'].shape}")
            print(f"Accepted shape: {g['accepted'].shape}")
            print(f"Iteration: {g.attrs['iteration']}")
            
            # Check a few values
            print(f"First few coords: {g['chain'][0, :2, :2]}")
            print(f"First few log probs: {g['log_prob'][0, :5]}")
            print(f"Accepted counts: {g['accepted'][:5]}")
            
    except Exception as e:
        print(f"✗ Error in save_slice: {e}")
        import traceback
        traceback.print_exc()