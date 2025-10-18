import jax
import jax.numpy as jnp
import pytest
from typing import List

def make_distributions(
    seeds: List[int],
    dim: int,
):
    """
    Make a list of distributions.

    Args:
        seeds (list[int]): List of seeds.
        dim (int): Dimension of the distribution.

    Returns:
        distributions (list[Callable]): List of distributions.
    """
    distributions = []
    for seed in seeds:
        key = jax.random.PRNGKey(seed)
        distributions.append(make_gaussian_skewed(key, dim=dim, cond_number=1))
        distributions.append(make_gaussian_skewed(key, dim=dim, cond_number=100))
        distributions.append(make_gaussian_cov_wishart(key, dim=dim, dof=10))
        # distributions.append(make_rosenbrock(key))
    return distributions

def make_gaussian(
    precision: jnp.ndarray,
    true_mean: jnp.ndarray,
):
    '''
    Make a Gaussian distribution.

    Args:
        precision (jnp.ndarray): Precision matrix.
        true_mean (jnp.ndarray): True mean of the distribution.

    Returns:
        log_prob (Callable): Log probability function.
    '''
    def log_prob(x: jnp.ndarray) -> jnp.ndarray:
        centered = x - true_mean
        cov_centered = jnp.linalg.solve(precision, centered)
        return - 0.5 * jnp.einsum('j,j->', centered, cov_centered)
    return log_prob

def make_gaussian_skewed(
    key: jax.random.PRNGKey, 
    dim: int,
    cond_number: int = 1000,
):
    '''
    Make a Gaussian distribution with a random covariance matrix.

    Args:
        key (jax.random.PRNGKey): Random number generator key.
        dim (int): Dimension of the distribution.
        cond_number (int): Condition number of the covariance matrix.

    Returns:
        log_prob (Callable): Log probability function.
        true_mean (jnp.ndarray): True mean of the distribution.
        precision (jnp.ndarray): Precision matrix.
    '''
    keys = jax.random.split(key, 2)
    precision = _make_covariance_skewed(keys[0], dim, cond_number)
    true_mean = jax.random.normal(keys[1], shape=(dim,))
    
    log_prob = make_gaussian(precision, true_mean)

    return log_prob

def _make_covariance_skewed(
    key: jax.random.PRNGKey,
    dim: int,
    cond_number: int = 1000,
):
    '''
    Make a covariance matrix.

    Args:
        key (jax.random.PRNGKey): Random number generator key.
        dim (int): Dimension of the distribution.
        cond_number (int): Condition number of the covariance matrix.

    Returns:
        precision (jnp.ndarray): Precision matrix.
    '''
    eigenvals = 0.1 * jnp.linspace(1, cond_number, dim)
    H = jax.random.normal(key, shape=(dim, dim))
    Q, _ = jnp.linalg.qr(H)
    precision = Q @ jnp.diag(eigenvals) @ Q.T
    precision = 0.5 * (precision + precision.T)
    return precision

def make_gaussian_cov_wishart(
    key: jax.random.PRNGKey, 
    dim: int,
    dof: int,
):
    '''
    Make a Gaussian distribution with a Wishart covariance matrix.

    Args:
        key (jax.random.PRNGKey): Random number generator key.
        dim (int): Dimension of the distribution.
        dof (int): Degrees of freedom of the Wishart distribution.
    
    Returns:
        log_prob (Callable): Log probability function.
    '''
    keys = jax.random.split(key, 2)
    S = _make_covariance_wishart(keys[0], dim, dof)
    true_mean = jax.random.normal(keys[1], shape=(dim,))
    
    log_prob = make_gaussian(S, true_mean)

    return log_prob

def _make_covariance_wishart(
    key: jax.random.PRNGKey,
    dim: int,  
    dof: int,
):
    '''
    Make a covariance matrix.

    Args:
        key (jax.random.PRNGKey): Random number generator key.
        dim (int): Dimension of the distribution.
        dof (int): Degrees of freedom of the Wishart distribution.

    Returns:
        precision (jnp.ndarray): Precision matrix.
    '''
    G = jax.random.normal(key, shape=(dof, dim))
    S = jnp.einsum('bi,bj->ij', G, G)
    return S

def make_rosenbrock(
    key: jax.random.PRNGKey,
):
    '''
    Make a Rosenbrock distribution.

    Args:
        key (jax.random.PRNGKey): Random number generator key.

    Returns:
        log_prob (Callable): Log probability function.
    '''
    a, b = 1.0, 100.0
    return lambda x: - jnp.sum(b * (x[1:] - x[:-1]**2)**2 + (a - x[:-1])**2)

def make_neal_funnel(
    dim: int,
):
    '''
    Make a Neal's Funnel distribution.
    
    .. math::

        p(x, \nu) \propto \left[\prod_{i=1}^d \mathcal N(x_i | 0, e^{\nu / 2})\right] \times \mathcal N(\nu | 0, 3)

    The resulting log probability has :math:`\nu` as the [0]th dimension.

    Args:
        dim (int): Dimension of the :math:`x` variable.
    '''
    def log_prob(x: jnp.ndarray) -> jnp.ndarray:
        return  0.5 * jnp.exp(-x[0]) * x[1:].T @ x[1:] + (0.5/jnp.sqrt(3)) * x[0]**2
    return log_prob


def make_allen_cahn(
    lattice_spacing: float,
):
    '''
    Make an Allen-Cahn distribution. It's an infinite dimensional probability distribution.

    .. math::
        
        p(x) \propto \exp \left( - \int_0^1 \frac{1}{2} (\partial_x u(x))^2 + V(u(x)) ~dx \right)
    
    where :math:`V(u) = (1 - u^2)^2`.

    In practice, we discretize the integral as a sum, and approximate the derivative as a finite difference.

    .. math::
        
        p(x) \propto \exp \left( \frac{1}{2h} \sum_{i=0}^{N-1} (u_{i+1} - u_i)^2 + h \sum_{i=0}^{N} V(u_i) \right)

    Args:
        lattice_spacing (float): Lattice spacing. In the equaiton above, :math:`h` is the lattice spacing.
    '''
    def log_prob(x: jnp.ndarray) -> jnp.ndarray:
        kinetic = 0.5 / lattice_spacing* jnp.sum((x[1:] - x[:-1])**2)
        potential = lattice_spacing * jnp.sum((1 - x**2)**2)
        return kinetic + potential
    return log_prob


    

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gaussian_skewed(seed):
    key = jax.random.PRNGKey(seed)
    precision = _make_covariance_skewed(key, dim=10, cond_number=100)

    # Is it symmetric?
    assert jnp.array_equal(precision, precision.T)
    # Is it positive definite?
    assert jnp.all(jnp.linalg.eigvals(precision) > 0)

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gaussian_wishart(seed):
    key = jax.random.PRNGKey(seed)
    S = _make_covariance_wishart(key, dim=10, dof=10)

    # Is it symmetric?
    assert jnp.array_equal(S, S.T)
    # Is it positive definite?
    assert jnp.all(jnp.linalg.eigvals(S) > 0)