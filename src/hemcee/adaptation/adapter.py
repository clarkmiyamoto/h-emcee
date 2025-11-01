from typing import NamedTuple, Callable
import jax.numpy as jnp
from .base import Adapter
from .dual_averaging import DAParameters, DAState, DualAveragingAdapter
from .chees import ChEESParameters, ChEESState, ChEESAdapter


class NoOpState(NamedTuple):
    """Trivial state that just holds constant values."""
    constant_step_size: float
    constant_integration_time: float


class NoOpAdapter(Adapter):
    """No-op adapter for when adaptation is disabled."""
    
    def __init__(self, initial_step_size: float, initial_L: float):
        self.constant_step_size = initial_step_size
        self.constant_integration_time = initial_L
    
    def init(self, dim: int) -> NoOpState:
        """Initialize with the provided constant values."""
        return NoOpState(constant_step_size=self.constant_step_size, constant_integration_time=self.constant_integration_time)
    
    def update(self, state: NoOpState, **kwargs) -> NoOpState:
        """No-op: return state unchanged."""
        return state
    
    def value(self, state: NoOpState) -> tuple[float, float]:
        """Return constant values."""
        return (state.constant_step_size, state.constant_integration_time)
    
    def finalize(self, state: NoOpState) -> tuple[float, float]:
        """Return constant values."""
        return (state.constant_step_size, state.constant_integration_time)


class CompositeState(NamedTuple):
    """Combined state for both adapters."""
    da_state: DAState
    chees_state: ChEESState


class CompositeAdapter(Adapter):
    """Composite adapter that combines DA + ChEES."""
    
    def __init__(self, da_parameters: DAParameters, chees_parameters: ChEESParameters, move_type: str,
                 initial_step_size: float, initial_L: float):
        self.da_adapter = DualAveragingAdapter(da_parameters, initial_step_size, initial_L)
        self.chees_adapter = ChEESAdapter(chees_parameters, move_type, initial_step_size, initial_L)
    
    def init(self, dim: int) -> CompositeState:
        """Initialize both adapters."""
        da_state = self.da_adapter.init(dim)
        chees_state = self.chees_adapter.init(dim)
        return CompositeState(da_state=da_state, chees_state=chees_state)
    
    def update(self, state: CompositeState, **kwargs) -> CompositeState:
        """Update both adapters."""
        da_state_new = self.da_adapter.update(state.da_state, log_accept_rate=kwargs['log_accept_rate'])
        chees_state_new = self.chees_adapter.update(state.chees_state, **kwargs)
        return CompositeState(da_state=da_state_new, chees_state=chees_state_new)
    
    def value(self, state: CompositeState) -> tuple[float, float]:
        """Return (step_size, integration_time) tuple."""
        step_size, _ = self.da_adapter.value(state.da_state)
        T = self.chees_adapter._get_T(state.chees_state)
        integration_length = jnp.maximum(jnp.round(T/step_size), 1)
        integration_length = jnp.astype(integration_length, int)
        return (step_size, integration_length)
    
    def finalize(self, state: CompositeState) -> tuple[float, float]:
        """Return final (step_size, integration_time) tuple."""
        step_size, _ = self.da_adapter.finalize(state.da_state)
        T = self.chees_adapter._get_T(state.chees_state)
        integration_length = jnp.maximum(jnp.round(T/step_size), 1)
        integration_length = jnp.astype(integration_length, int)
        return (step_size, integration_length)


def select_adapter(da_parameters: bool | DAParameters,
                   chees_parameters: bool | ChEESParameters,
                   move: Callable,
                   initial_step_size: float,
                   initial_L: float) -> Adapter:
    """Select appropriate adapter based on parameter configuration.
    
    Args:
        da_parameters: True for default DA, False to disable, or custom DAParameters
        chees_parameters: True for default ChEES, False to disable, or custom ChEESParameters
        move: Callable move function to use for ChEES adapter.
        initial_step_size: Initial step size value to use for pass-through
        initial_L: Initial integration time value to use for pass-through
        
    Returns:
        Adapter instance configured according to parameters
        
    Raises:
        ValueError: If both adapters are disabled
    """
    move_type = getattr(move, '__name__', None)
    if move_type not in ['hmc_move', 'hmc_walk_move', 'hmc_side_move']:
        raise ValueError(f'Move type for ChEES tuner is not suppported. Choose between ["hmc_move", "hmc_walk_move", "hmc_side_move"]. Got {move_type}')

    # Convert True to default parameters
    if da_parameters is True:
        da_parameters = DAParameters()  # Default step size
    if chees_parameters is True:
        chees_parameters = ChEESParameters()
    
    # Determine which adapters are enabled
    da_enabled = da_parameters is not False
    chees_enabled = chees_parameters is not False
    
    # Check for invalid configuration
    if not da_enabled and not chees_enabled:
        return NoOpAdapter(initial_step_size, initial_L)
    
    # Return appropriate adapter
    if da_enabled and not chees_enabled:
        return DualAveragingAdapter(da_parameters, initial_step_size, initial_L)
    elif not da_enabled and chees_enabled:
        return ChEESAdapter(chees_parameters, move_type, initial_step_size, initial_L)
    else:  # Both enabled
        return CompositeAdapter(da_parameters, chees_parameters, move_type, initial_step_size, initial_L)