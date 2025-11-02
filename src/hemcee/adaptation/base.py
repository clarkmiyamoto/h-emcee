'''
I've developed the adapters to mirror BlackJAX's 
(https://github.com/blackjax-devs/blackjax) adapters...
Obviously this is very scuffed, but I don't have a good alternative.
I'm very open to collaboration here!
'''
from abc import ABC, abstractmethod
from typing import NamedTuple
import jax.numpy as jnp


class Adapter(ABC):
    """Abstract base class for adaptation algorithms."""
    
    @abstractmethod
    def init(self, dim: int):
        """Initialize adapter state with initial hyperparameter value and dimensionality."""
        pass
    
    @abstractmethod
    def update(self, 
               state, 
               log_accept_rate: jnp.ndarray, 
               **kwargs):
        """Update state given acceptance rate and current positions."""
        pass
    
    @abstractmethod
    def value(self, state) -> tuple[float, float]:
        """Extract current adapted values during warmup. Returns (step_size, integration_time)."""
        pass
    
    @abstractmethod
    def finalize(self, state) -> tuple[float, float]:
        """Return final adapted values after warmup. Returns (step_size, integration_time)."""
        pass
