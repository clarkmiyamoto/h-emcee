'''
Code from https://github.com/dfm/emcee/blob/main/src/emcee/autocorr.py
Ported to JAX.
'''

import logging

import jax.numpy as jnp

__all__ = ["function_1d", "integrated_time", "AutocorrError"]

logger = logging.getLogger(__name__)


def next_pow_two(n):
    """Returns the next power of two greater than or equal to `n`"""
    i = 1
    while i < n:
        i = i << 1
    return i


def function_1d(x):
    """Estimate the normalized autocorrelation function of a 1-D series

    Args:
        x: The series as a 1-D numpy array.

    Returns:
        array: The autocorrelation function of the time series.

    """
    x = jnp.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = jnp.fft.fft(x - jnp.mean(x), n=2 * n)
    acf = jnp.fft.ifft(f * jnp.conjugate(f))[: len(x)].real
    acf /= acf[0]
    return acf


def auto_window(taus, c):
    m = jnp.arange(len(taus)) < c * taus
    if jnp.any(m):
        return jnp.argmin(m)
    return len(taus) - 1


def integrated_time(x, c=5, tol=50, quiet=False, has_walkers=True):
    """Estimate the integrated autocorrelation time of a time series.

    This estimate uses the iterative procedure described on page 16 of
    `Sokal's notes <https://www.semanticscholar.org/paper/Monte-Carlo-Methods-in-Statistical-Mechanics%3A-and-Sokal/0bfe9e3db30605fe2d4d26e1a288a5e2997e7225>`_ to
    determine a reasonable window size.

    Args:
        x (jnp.ndarray): The time series. If 2-dimensional, the array
            dimesions are interpreted as ``(n_step, n_walker)`` unless
            ``has_walkers==False``, in which case they are interpreted as
            ``(n_step, n_param)``. If 3-dimensional, the dimensions are
            interperted as ``(n_step, n_walker, n_param)``.
        c (Optional[float]): The step size for the window search. (default:
            ``5``)
        tol (Optional[float]): The minimum number of autocorrelation times
            needed to trust the estimate. (default: ``50``)
        quiet (Optional[bool]): This argument controls the behavior when the
            chain is too short. If ``True``, give a warning instead of raising
            an :class:`AutocorrError`. (default: ``False``)
        has_walkers (Optional[bool]): Whether the last axis should be
            interpreted as walkers or parameters if ``x`` has 2 dimensions.
            (default: ``True``)

    Returns:
        float or array: An estimate of the integrated autocorrelation time of
            the time series ``x``.

    Raises
        AutocorrError: If the autocorrelation time can't be reliably estimated
            from the chain and ``quiet`` is ``False``. This normally means
            that the chain is too short.

    """
    x = jnp.atleast_1d(x)
    if len(x.shape) == 1:
        x = x[:, jnp.newaxis, jnp.newaxis]
    if len(x.shape) == 2:
        if not has_walkers:
            x = x[:, jnp.newaxis, :]
        else:
            x = x[:, :, jnp.newaxis]
    if len(x.shape) != 3:
        raise ValueError("invalid dimensions")

    n_t, n_w, n_d = x.shape
    tau_est = jnp.empty(n_d)
    windows = jnp.empty(n_d, dtype=int)

    # Loop over parameters
    for d in range(n_d):
        f = jnp.zeros(n_t)
        for k in range(n_w):
            f += function_1d(x[:, k, d])
        f /= n_w
        taus = 2.0 * jnp.cumsum(f) - 1.0
        windows = windows.at[d].set(auto_window(taus, c))
        tau_est = tau_est.at[d].set(taus[windows[d]])


    # Check convergence
    flag = tol * tau_est > n_t

    # Warn or raise in the case of non-convergence
    if jnp.any(flag):
        msg = (
            "The chain is shorter than {0} times the integrated "
            "autocorrelation time for {1} parameter(s). Use this estimate "
            "with caution and run a longer chain!\n"
        ).format(tol, jnp.sum(flag))
        msg += "N/{0} = {1:.0f};\ntau: {2}".format(tol, n_t / tol, tau_est)
        if not quiet:
            raise AutocorrError(tau_est, msg)
        logger.warning(msg)

    return tau_est


class AutocorrError(Exception):
    """Raised if the chain is too short to estimate an autocorrelation time.

    The current estimate of the autocorrelation time can be accessed via the
    ``tau`` attribute of this exception.

    """

    def __init__(self, tau, *args, **kwargs):
        self.tau = tau
        super(AutocorrError, self).__init__(*args, **kwargs)