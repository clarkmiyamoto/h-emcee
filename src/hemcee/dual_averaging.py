import jax.numpy as jnp
from typing import Tuple

def dual_averaging_step(
    log_eps0: float,
    H_bar: float,
    log_epsilon: float,
    log_epsilon_bar: float,
    current_accept_rate: float,
    m: int,                     # 1-indexed warmup iteration count
    target_accept: float = 0.65,
    gamma: float = 0.05,
    t0: float = 10.0,
    kappa: float = 0.75,
) -> Tuple[float, float, float, float]:
    """
    One dual-averaging update step (Nesterov 2009; Hoffman & Gelman 2014, Stan).

    Args:
        log_eps0: Log of initial step size.
        H_bar: Average of the gradient.
        log_epsilon: Log of the current step size.
        log_epsilon_bar: Log of the average of the step size.
        current_accept_rate: Current acceptance rate.
        m: 1-indexed warmup iteration count.
        target_accept: Target acceptance rate.
        gamma: Dual averaging parameter.
        t0: Dual averaging parameter.
        kappa: Dual averaging parameter.
    
    Returns: 
        Tuple of (H_bar, log_epsilon, log_epsilon_bar, eps_now)
    """
    # Dual averaging update
    eta = 1.0 / (m + t0)

    # Update log step size
    H_bar = (1 - eta) * H_bar + eta * (target_accept - current_accept_rate)

    # Compute log step size with shrinkage
    log_epsilon = log_eps0 - jnp.sqrt(m) / gamma * H_bar

    # Update log_epsilon_bar for final step size
    eta_bar = m**(-kappa)
    log_epsilon_bar = (1 - eta_bar) * log_epsilon_bar + eta_bar * log_epsilon
    
    # Current step size (for this iteration)
    eps_now = jnp.exp(log_epsilon)

    return H_bar, log_epsilon, log_epsilon_bar, eps_now


def dual_averaging_update(
    log_eps0: float,
    H_bar: float,
    log_epsilon: float,
    log_epsilon_bar: float,
    current_accept_rate: float,
    m: int,
    target_accept: float = 0.65,
    gamma: float = 0.05,
    t0: float = 10.0,
    kappa: float = 0.75,
) -> float:
    """
    Dual averaging update function that returns only the updated log_epsilon_bar.
    This is designed to be used with jax.lax.cond in the main sampling loop.
    
    Args:
        log_eps0: Log of initial step size.
        H_bar: Average of the gradient.
        log_epsilon: Log of the current step size.
        log_epsilon_bar: Log of the average of the step size.
        current_accept_rate: Current acceptance rate.
        m: 1-indexed warmup iteration count.
        target_accept: Target acceptance rate.
        gamma: Dual averaging parameter.
        t0: Dual averaging parameter.
        kappa: Dual averaging parameter.
    
    Returns:
        Updated log_epsilon_bar
    """
    _, _, log_epsilon_bar, _ = dual_averaging_step(
        log_eps0, H_bar, log_epsilon, log_epsilon_bar,
        current_accept_rate, m, target_accept, gamma, t0, kappa
    )
    return log_epsilon_bar


class DualAveraging:
    """
    Dual averaging state manager for step size adaptation.
    """
    
    def __init__(
        self,
        log_eps0: float,
        target_accept: float = 0.65,
        gamma: float = 0.05,
        t0: float = 10.0,
        kappa: float = 0.75,
    ):
        """
        Initialize dual averaging state.
        
        Args:
            log_eps0: Log of initial step size.
            target_accept: Target acceptance rate.
            gamma: Dual averaging parameter.
            t0: Dual averaging parameter.
            kappa: Dual averaging parameter.
        """
        self.log_eps0 = log_eps0
        self.target_accept = target_accept
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa
        
        # Initialize state
        self.H_bar = 0.0
        self.log_epsilon = log_eps0
        self.log_epsilon_bar = 0.0
        self.m = 0
    
    def update(self, current_accept_rate: float) -> Tuple[float, float, float]:
        """
        Update dual averaging state and return new step size.
        
        Args:
            current_accept_rate: Current acceptance rate.
            
        Returns:
            Tuple of (H_bar, log_epsilon, log_epsilon_bar)
        """
        self.m += 1
        
        self.H_bar, self.log_epsilon, self.log_epsilon_bar, _ = dual_averaging_step(
            self.log_eps0, self.H_bar, self.log_epsilon, self.log_epsilon_bar,
            current_accept_rate, self.m, self.target_accept, self.gamma, self.t0, self.kappa
        )
        
        return self.H_bar, self.log_epsilon, self.log_epsilon_bar
    
    def get_final_step_size(self) -> float:
        """Get the final step size after warmup."""
        return jnp.exp(self.log_epsilon_bar)