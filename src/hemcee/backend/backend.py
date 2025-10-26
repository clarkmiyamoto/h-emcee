'''
This is a ever-so slightly modified version of `emcee`'s backend. 
See https://github.com/dfm/emcee/tree/main/src/emcee/backends.
'''
import jax
import jax.numpy as jnp
from hemcee import autocorr
from typing import Tuple, List


__all__ = ["Backend"]

class Backend(object):
    """
    A simple default backend that stores the chain in memory.
    
    Args:
        dtype (jnp.dtype): Datatype of chains. I.e. float32 vs float64.
        device (jax.Device): Where to store the chains.
    """

    def __init__(self, 
                 dtype: jnp.dtype = None, 
                 device: jax.Device = None):
        '''
        
        '''
        if dtype is None:
            dtype = jnp.float64
        self.dtype = dtype
        self.device = jax.devices('cpu')[0] if device is None else device

    def reset(self, nwalkers: int, ndim: int):
        """Clear the state of the chain and empty the backend

        Args:
            nwakers (int): The size of the ensemble
            ndim (int): The number of dimensions

        """
        self.nwalkers: int         = int(nwalkers)
        self.ndim: int             = int(ndim)
        self.iteration: int        = 0
        self.accepted: jnp.ndarray = jnp.zeros(self.nwalkers, dtype=self.dtype, device=self.device)
        self.chain: List[jnp.ndarray]    = []
        self.log_prob: List[jnp.ndarray] = []
        self.initialized: bool     = True

    def get_value(self, 
                  name: str, 
                  flat: bool = False, 
                  thin: int = 1, 
                  discard: int = 0):
        if self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )
        v = getattr(self, name)
        if (name == 'chain') or (name == 'log_prob'):
            v = jnp.concatenate(v, axis=0)
        v = v[discard::thin]
        if flat:
            s = list(v.shape[1:])
            s[0] = jnp.prod(v.shape[:2])
            return v.reshape(s)
        return v

    def get_chain(self, **kwargs):
        """Get the stored chain of MCMC samples

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers, ndim]: The MCMC samples.

        """
        return self.get_value("chain", **kwargs)

    def get_log_prob(self, **kwargs):
        """Get the chain of log probabilities evaluated at the MCMC samples

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers]: The chain of log probabilities.

        """
        values = self.get_value("log_prob", **kwargs)
        return jnp.concatenate(values, axis=0)

    def get_last_sample(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Access the most recent sample in the chain"""
        if (not self.initialized) or (self.iteration <= 0):
            raise AttributeError(
                "you must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )
        it = self.iteration
        return (self.get_chain(discard=it - 1)[0],
                self.get_log_prob(discard = it - 1)[0])

    def get_autocorr_time(self, discard=0, thin=1, **kwargs):
        """Compute an estimate of the autocorrelation time for each parameter

        Args:
            thin (Optional[int]): Use only every ``thin`` steps from the
                chain. The returned estimate is multiplied by ``thin`` so the
                estimated time is in units of steps, not thinned steps.
                (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Other arguments are passed directly to
        :func:`hemcee.autocorr.integrated_time`.

        Returns:
            array[ndim]: The integrated autocorrelation time estimate for the
                chain for each parameter.

        """
        x = self.get_chain(discard=discard, thin=thin)
        return thin * autocorr.integrated_time(x, **kwargs)

    @property
    def shape(self):
        """The dimensions of the ensemble ``(nwalkers, ndim)``"""
        return self.nwalkers, self.ndim

    def _check(self, state, accepted):
        self._check_blobs(state.blobs)
        nwalkers, ndim = self.shape
        has_blobs = self.has_blobs()
        if state.coords.shape != (nwalkers, ndim):
            raise ValueError(
                "invalid coordinate dimensions; expected {0}".format(
                    (nwalkers, ndim)
                )
            )
        if state.log_prob.shape != (nwalkers,):
            raise ValueError(
                "invalid log probability size; expected {0}".format(nwalkers)
            )
        if state.blobs is not None and not has_blobs:
            raise ValueError("unexpected blobs")
        if state.blobs is None and has_blobs:
            raise ValueError("expected blobs, but none were given")
        if state.blobs is not None and len(state.blobs) != nwalkers:
            raise ValueError(
                "invalid blobs size; expected {0}".format(nwalkers)
            )
        if accepted.shape != (nwalkers,):
            raise ValueError(
                "invalid acceptance size; expected {0}".format(nwalkers)
            )
        
    def save_slice(self, 
                   coords: jnp.ndarray, 
                   log_prob: jnp.ndarray, 
                   accepted: jnp.ndarray, 
                   index: int):
        """Save a slice of a JAX array to backend

        Args:
            coords (jnp.ndarray): Coordinates array of shape (nsteps, nwalkers, ndim)
            log_prob (jnp.ndarray): Log probability array of shape (nsteps, nwalkers)
            accepted (jnp.ndarray): Accepted array of shape (nwalkers,)
            index (int): Starting index for the slice
        """        
        # Move to storage device
        coords = jax.device_put(coords, self.device)
        log_prob = jax.device_put(log_prob, self.device)
        accepted = jax.device_put(accepted, self.device)
            
        # Get current dimensions
        nsteps, nwalkers, ndim = coords.shape
        
        # Resize datasets if necessary
        current_size = self.iteration
        
        # Write the data to storage
        self.chain.append(coords)
        self.log_prob.append(log_prob)
        self.accepted += accepted
        
        # Update iteration counter
        self.iteration = index + nsteps

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass