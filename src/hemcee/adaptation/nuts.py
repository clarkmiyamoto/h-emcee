"""
Ensemble NUTS Algorithm Implementation in JAX
Adapted from the standard NUTS algorithm to work with ensemble samplers
"""

from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from jax import random, lax
from jax.flatten_util import ravel_pytree
from collections import namedtuple

from hemcee.adaptation.nuts_walk import u_turn_condition as walk_u_turn_condition
from hemcee.adaptation.nuts_side import u_turn_condition as side_u_turn_condition
from hemcee.moves.hamiltonian.hmc_walk import hmc_walk_move, leapfrog_walk_move
from hemcee.moves.hamiltonian.hmc_side import hmc_side_move, leapfrog_side_move

# Data structures
IntegratorState = namedtuple("IntegratorState", ["z", "r", "potential_energy", "z_grad"])
TreeInfo = namedtuple("TreeInfo", [
    "z_left", "r_left", "z_left_grad",
    "z_right", "r_right", "z_right_grad", 
    "z_proposal", "z_proposal_pe", "z_proposal_grad", "z_proposal_energy",
    "depth", "weight", "r_sum", "turning", "diverging",
    "sum_accept_probs", "num_proposals"
])


def euclidean_kinetic_energy(r: jnp.ndarray, is_walk: bool = True) -> jnp.ndarray:
    """
    Euclidean kinetic energy function for ensemble momentum.
    
    Args:
        r: Momentum. Shape (n_chains_per_group, n_chains_per_group) for walk,
           or (n_chains_per_group,) for side
        is_walk: Whether this is walk mode (matrix momentum) or side mode (vector momentum)
        
    Returns:
        Kinetic energy scalar
    """
    if is_walk:
        # Walk mode: momentum is a matrix (n_chains_per_group, n_chains_per_group)
        return 0.5 * jnp.sum(r**2)
    else:
        # Side mode: momentum is a vector (n_chains_per_group,)
        return 0.5 * jnp.sum(r**2)


def leaf_idx_to_ckpt_idxs(n: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert leaf index to checkpoint indices.
    
    Args:
        n: Leaf index
        
    Returns:
        Tuple of (idx_min, idx_max)
    """
    idx_max = jnp.bitwise_count(n >> 1).astype(jnp.int32)
    num_subtrees = jnp.bitwise_count((~n & (n + 1)) - 1).astype(jnp.int32)
    idx_min = idx_max - num_subtrees + 1
    return idx_min, idx_max


def get_leaf(tree: TreeInfo, going_right: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Get the appropriate leaf from a tree based on direction.
    
    Args:
        tree: Tree to extract leaf from
        going_right: Direction
        
    Returns:
        Tuple of (z, r, z_grad)
    """
    return lax.cond(
        going_right,
        tree,
        lambda t: (t.z_right, t.r_right, t.z_right_grad),
        tree,
        lambda t: (t.z_left, t.r_left, t.z_left_grad)
    )


def uniform_transition_kernel(current_tree: TreeInfo, new_tree: TreeInfo) -> jnp.ndarray:
    """
    Compute transition probability for subtrees.
    
    Args:
        current_tree: Current tree
        new_tree: New tree
        
    Returns:
        Transition probability
    """
    new_weight = new_tree.weight
    current_weight = current_tree.weight
    
    # Numerically stable computation
    max_weight = jnp.maximum(new_weight, current_weight)
    exp_new = jnp.exp(new_weight - max_weight)
    exp_current = jnp.exp(current_weight - max_weight)
    
    return exp_new / (exp_new + exp_current)


def biased_transition_kernel(current_tree: TreeInfo, new_tree: TreeInfo) -> jnp.ndarray:
    """
    Compute transition probability for main trees.
    
    Args:
        current_tree: Current tree
        new_tree: New tree
        
    Returns:
        Transition probability
    """
    new_weight = new_tree.weight
    current_weight = current_tree.weight
    
    # Numerically stable computation
    max_weight = jnp.maximum(new_weight, current_weight)
    exp_new = jnp.exp(new_weight - max_weight)
    exp_current = jnp.exp(current_weight - max_weight)
    
    transition_prob = exp_new / (exp_new + exp_current)
    return jnp.clip(transition_prob, None, 1.0)


def is_turning(z_left: jnp.ndarray,
               r_left: jnp.ndarray,
               z_right: jnp.ndarray, 
               r_right: jnp.ndarray,
               z_initial: jnp.ndarray,
               r_initial: jnp.ndarray,
               turning_predicate: Callable,
               is_walk: bool,
               centering: jnp.ndarray = None,
               inv_covariance: jnp.ndarray = None) -> jnp.ndarray:
    """
    Check if the trajectory is turning back using the provided turning predicate.
    
    Args:
        z_left: Left position
        r_left: Left momentum
        z_right: Right position
        r_right: Right momentum
        z_initial: Initial position
        r_initial: Initial momentum
        turning_predicate: Turning predicate function (walk or side)
        is_walk: Whether this is walk mode
        centering: Centering matrix for walk mode (optional)
        inv_covariance: Inverse covariance for walk mode (optional)
        
    Returns:
        Boolean indicating if trajectory is turning
    """
    if is_walk:
        # Walk turning predicate needs centering and inv_covariance
        return (turning_predicate(r_left, z_left, z_initial, centering, inv_covariance) | 
                turning_predicate(r_right, z_right, z_initial, centering, inv_covariance))
    else:
        # Side turning predicate: r_current · r_initial <= 0
        return turning_predicate(r_left, r_initial) | turning_predicate(r_right, r_initial)


def build_basetree(leapfrog_fn: Callable,
                   kinetic_fn: Callable,
                   z: jnp.ndarray,
                   r: jnp.ndarray,
                   z_grad: jnp.ndarray,
                   step_size: float,
                   going_right: bool,
                   energy_current: float,
                   max_delta_energy: float,
                   log_prob: Callable,
                   grad_log_prob: Callable,
                   complementary_data: jnp.ndarray,
                   is_walk: bool) -> TreeInfo:
    """
    Build a single step (base tree) using ensemble leapfrog.
    
    Args:
        leapfrog_fn: Leapfrog function (walk or side)
        kinetic_fn: Kinetic energy function
        z: Position (n_chains_per_group, dim)
        r: Momentum (shape depends on walk/side)
        z_grad: Position gradient
        step_size: Step size
        going_right: Direction
        energy_current: Current energy
        max_delta_energy: Maximum energy difference
        log_prob: Log probability function
        grad_log_prob: Gradient of log probability
        complementary_data: Centered group2 (walk) or diff_particles (side)
        is_walk: Whether this is walk mode
        
    Returns:
        Base tree
    """
    step_size_signed = jnp.where(going_right, step_size, -step_size)
    
    # Perform one leapfrog step
    z_new, r_new = leapfrog_fn(z, r, grad_log_prob, step_size_signed, 1, complementary_data)
    
    # Compute new energy
    potential_energy_new = -log_prob(z_new)
    z_new_grad = -grad_log_prob(z_new)
    kinetic_energy_new = kinetic_fn(r_new, is_walk)
    energy_new = potential_energy_new + kinetic_energy_new
    
    delta_energy = energy_new - energy_current
    delta_energy = jnp.where(jnp.isnan(delta_energy), jnp.inf, delta_energy)
    tree_weight = -delta_energy
    
    diverging = delta_energy > max_delta_energy
    accept_prob = jnp.clip(jnp.exp(-delta_energy), None, 1.0)
    
    return TreeInfo(
        z_new, r_new, z_new_grad,  # left
        z_new, r_new, z_new_grad,  # right
        z_new, potential_energy_new, z_new_grad, energy_new,  # proposal
        0, tree_weight, r_new, False, diverging, accept_prob, 1  # metadata
    )


def is_iterative_turning(z_ckpts: jnp.ndarray,
                        r_ckpts: jnp.ndarray,
                        z_current: jnp.ndarray,
                        r_current: jnp.ndarray,
                        z_initial: jnp.ndarray,
                        r_initial: jnp.ndarray,
                        idx_min: int,
                        idx_max: int,
                        turning_predicate: Callable,
                        is_walk: bool,
                        centering: jnp.ndarray = None,
                        inv_covariance: jnp.ndarray = None) -> jnp.ndarray:
    """
    Iteratively check for turning conditions.
    
    Args:
        z_ckpts: Checkpoint positions
        r_ckpts: Checkpoint momenta
        z_current: Current position
        r_current: Current momentum
        z_initial: Initial position
        r_initial: Initial momentum
        idx_min: Minimum checkpoint index
        idx_max: Maximum checkpoint index
        turning_predicate: Turning predicate function
        is_walk: Whether this is walk mode
        centering: Centering matrix for walk mode (optional)
        inv_covariance: Inverse covariance for walk mode (optional)
        
    Returns:
        Boolean indicating if turning is detected
    """
    def body_fn(state):
        i, _ = state
        z_ckpt = z_ckpts[i]
        r_ckpt = r_ckpts[i]
        
        # For side mode, extract scalar from checkpoint array (shape (1,) -> scalar)
        # is_walk is statically known, so we can use Python if
        if not is_walk and r_ckpt.size > 0:
            r_ckpt = r_ckpt[0]
        
        # Check if there's turning at this checkpoint
        if is_walk:
            is_turning_at_i = turning_predicate(r_ckpt, z_ckpt, z_initial, centering, inv_covariance)
        else:
            is_turning_at_i = turning_predicate(r_ckpt, r_initial)
        
        return i - 1, is_turning_at_i
    
    # Run while loop: continue while (i >= idx_min) AND (NOT turning)
    _, turning = lax.while_loop(
        lambda state: (state[0] >= idx_min) & ~state[1],
        body_fn,
        (idx_max, False)
    )
    
    return turning


def combine_tree(current_tree: TreeInfo,
                 new_tree: TreeInfo,
                 going_right: bool,
                 rng_key: jnp.ndarray,
                 biased_transition: bool,
                 turning_predicate: Callable,
                 r_initial: jnp.ndarray,
                 is_walk: bool,
                 centering: jnp.ndarray = None,
                 inv_covariance: jnp.ndarray = None) -> TreeInfo:
    """
    Combine current tree with new tree.
    
    Args:
        current_tree: Current tree
        new_tree: New tree
        going_right: Direction of tree building
        rng_key: Random key
        biased_transition: Whether to use biased transition kernel
        turning_predicate: Turning predicate function
        r_initial: Initial momentum for turning condition
        is_walk: Whether this is walk mode
        centering: Centering matrix for walk mode (optional)
        inv_covariance: Inverse covariance for walk mode (optional)
        
    Returns:
        Combined tree
    """
    # Determine left and right leaves based on direction
    z_left, r_left, z_left_grad, z_right, r_right, z_right_grad = lax.cond(
        going_right,
        (current_tree, new_tree),
        lambda trees: (
            trees[0].z_left, trees[0].r_left, trees[0].z_left_grad,
            trees[1].z_right, trees[1].r_right, trees[1].z_right_grad
        ),
        (new_tree, current_tree),
        lambda trees: (
            trees[0].z_left, trees[0].r_left, trees[0].z_left_grad,
            trees[1].z_right, trees[1].r_right, trees[1].z_right_grad
        )
    )
    
    # Combine momentum sums
    r_sum = jax.tree.map(jnp.add, current_tree.r_sum, new_tree.r_sum)
    
    # Calculate transition probability
    # Note: centering and inv_covariance need to be in closure for is_turning
    def compute_with_turning(trees):
        ct, nt, zl, rl, zr, rr = trees
        return (
            biased_transition_kernel(ct, nt),
            nt.turning | is_turning(zl, rl, zr, rr, ct.z_left, r_initial, 
                                    turning_predicate, is_walk, centering, inv_covariance)
        )
    
    transition_prob, turning = lax.cond(
        biased_transition,
        (current_tree, new_tree, z_left, r_left, z_right, r_right),
        compute_with_turning,
        (current_tree, new_tree),
        lambda x: (uniform_transition_kernel(x[0], x[1]), x[0].turning)
    )
    
    # Stochastically select which tree's proposal to keep
    accept_new = random.bernoulli(rng_key, transition_prob)
    
    z_proposal, z_proposal_pe, z_proposal_grad, z_proposal_energy = lax.cond(
        accept_new,
        new_tree,
        lambda tree: (tree.z_proposal, tree.z_proposal_pe, tree.z_proposal_grad, tree.z_proposal_energy),
        current_tree,
        lambda tree: (tree.z_proposal, tree.z_proposal_pe, tree.z_proposal_grad, tree.z_proposal_energy)
    )
    
    # Combine other tree properties
    tree_depth = current_tree.depth + 1
    tree_weight = jnp.logaddexp(current_tree.weight, new_tree.weight)
    sum_accept_probs = current_tree.sum_accept_probs + new_tree.sum_accept_probs
    num_proposals = current_tree.num_proposals + new_tree.num_proposals
    
    return TreeInfo(
        z_left, r_left, z_left_grad,
        z_right, r_right, z_right_grad,
        z_proposal, z_proposal_pe, z_proposal_grad, z_proposal_energy,
        tree_depth, tree_weight, r_sum, turning,
        new_tree.diverging, sum_accept_probs, num_proposals
    )


def iterative_build_subtree(prototype_tree: TreeInfo,
                           leapfrog_fn: Callable,
                           kinetic_fn: Callable,
                           step_size: float,
                           going_right: bool,
                           rng_key: jnp.ndarray,
                           energy_current: float,
                           max_delta_energy: float,
                           z_ckpts: jnp.ndarray,
                           r_ckpts: jnp.ndarray,
                           turning_predicate: Callable,
                           r_initial: jnp.ndarray,
                           log_prob: Callable,
                           grad_log_prob: Callable,
                           complementary_data: jnp.ndarray,
                           is_walk: bool,
                           centering: jnp.ndarray = None,
                           inv_covariance: jnp.ndarray = None) -> TreeInfo:
    """
    Build subtree iteratively.
    
    Args:
        prototype_tree: Prototype tree
        leapfrog_fn: Leapfrog function
        kinetic_fn: Kinetic energy function
        step_size: Step size
        going_right: Direction
        rng_key: Random key
        energy_current: Current energy
        max_delta_energy: Maximum energy difference
        z_ckpts: Checkpoint positions
        r_ckpts: Checkpoint momenta
        turning_predicate: Turning predicate function
        r_initial: Initial momentum
        log_prob: Log probability function
        grad_log_prob: Gradient of log probability
        complementary_data: Centered group2 (walk) or diff_particles (side)
        is_walk: Whether this is walk mode
        
    Returns:
        Built subtree
    """
    max_num_proposals = 2 ** prototype_tree.depth
    
    def cond_fn(state):
        tree, turning, _, _, _ = state
        return (tree.num_proposals < max_num_proposals) & ~turning & ~tree.diverging
    
    def body_fn(state):
        current_tree, _, z_ckpts, r_ckpts, rng_key = state
        rng_key, transition_rng_key = random.split(rng_key)
        
        # Get the leaf to extend from
        z, r, z_grad = get_leaf(current_tree, going_right)
        
        # Build a single step (base tree)
        new_leaf = build_basetree(
            leapfrog_fn, kinetic_fn, z, r, z_grad,
            step_size, going_right,
            energy_current, max_delta_energy,
            log_prob, grad_log_prob, complementary_data, is_walk
        )
        
        # Combine with current subtree
        new_tree = lax.cond(
            current_tree.num_proposals == 0,
            new_leaf,
            lambda x: x,
            (current_tree, new_leaf, going_right, transition_rng_key),
            lambda x: combine_tree(x[0], x[1], x[2], x[3], False, turning_predicate, r_initial, is_walk, centering, inv_covariance)
        )
        
        # Update checkpoints (every even leaf index)
        leaf_idx = current_tree.num_proposals
        ckpt_idx_min, ckpt_idx_max = leaf_idx_to_ckpt_idxs(leaf_idx)
        
        # Prepare checkpoint values (handle scalar momentum for side mode)
        z_ckpt_val = new_leaf.z_right
        # For side mode, wrap scalar in array; for walk mode, use as is
        r_ckpt_val = jnp.atleast_1d(new_leaf.r_right) if not is_walk else new_leaf.r_right
        
        z_ckpts, r_ckpts = lax.cond(
            leaf_idx % 2 == 0,
            (z_ckpts, r_ckpts, z_ckpt_val, r_ckpt_val, ckpt_idx_max),
            lambda x: (x[0].at[x[4]].set(x[2]), x[1].at[x[4]].set(x[3])),
            (z_ckpts, r_ckpts, z_ckpt_val, r_ckpt_val, ckpt_idx_max),
            lambda x: (x[0], x[1])
        )
        
        # Check for turning using iterative method
        turning = is_iterative_turning(
            z_ckpts, r_ckpts,
            new_leaf.z_right, new_leaf.r_right,
            prototype_tree.z_left, r_initial,
            ckpt_idx_min, ckpt_idx_max,
            turning_predicate, is_walk, centering, inv_covariance
        )
        
        return new_tree, turning, z_ckpts, r_ckpts, rng_key
    
    basetree = prototype_tree._replace(num_proposals=0)
    tree, turning, _, _, _ = lax.while_loop(
        cond_fn, body_fn, (basetree, False, z_ckpts, r_ckpts, rng_key)
    )
    
    return TreeInfo(
        tree.z_left, tree.r_left, tree.z_left_grad,
        tree.z_right, tree.r_right, tree.z_right_grad,
        tree.z_proposal, tree.z_proposal_pe, tree.z_proposal_grad, tree.z_proposal_energy,
        prototype_tree.depth, tree.weight, tree.r_sum,
        turning, tree.diverging, tree.sum_accept_probs, tree.num_proposals
    )


def double_tree(current_tree: TreeInfo,
                leapfrog_fn: Callable,
                kinetic_fn: Callable,
                step_size: float,
                going_right: bool,
                rng_key: jnp.ndarray,
                energy_current: float,
                max_delta_energy: float,
                z_ckpts: jnp.ndarray,
                r_ckpts: jnp.ndarray,
                turning_predicate: Callable,
                r_initial: jnp.ndarray,
                log_prob: Callable,
                grad_log_prob: Callable,
                complementary_data: jnp.ndarray,
                is_walk: bool,
                centering: jnp.ndarray = None,
                inv_covariance: jnp.ndarray = None) -> TreeInfo:
    """
    Double the tree in the chosen direction.
    
    Args:
        current_tree: Current tree
        leapfrog_fn: Leapfrog function
        kinetic_fn: Kinetic energy function
        step_size: Step size
        going_right: Direction
        rng_key: Random key
        energy_current: Current energy
        max_delta_energy: Maximum energy difference
        z_ckpts: Checkpoint positions
        r_ckpts: Checkpoint momenta
        turning_predicate: Turning predicate function
        r_initial: Initial momentum
        log_prob: Log probability function
        grad_log_prob: Gradient of log probability
        complementary_data: Centered group2 (walk) or diff_particles (side)
        is_walk: Whether this is walk mode
        
    Returns:
        Doubled tree
    """
    key, transition_key = random.split(rng_key)
    
    # Build subtree in the chosen direction
    new_subtree = iterative_build_subtree(
        current_tree, leapfrog_fn, kinetic_fn,
        step_size, going_right, key, energy_current,
        max_delta_energy, z_ckpts, r_ckpts, turning_predicate, r_initial,
        log_prob, grad_log_prob, complementary_data, is_walk, centering, inv_covariance
    )
    
    # Combine the current tree with the new subtree
    return combine_tree(
        current_tree, new_subtree,
        going_right, transition_key, True, turning_predicate, r_initial, is_walk, centering, inv_covariance
    )


def build_tree(leapfrog_fn: Callable,
               kinetic_fn: Callable,
               z_initial: jnp.ndarray,
               r_initial: jnp.ndarray,
               log_prob: Callable,
               grad_log_prob: Callable,
               step_size: float,
               rng_key: jnp.ndarray,
               max_delta_energy: float,
               max_tree_depth: int,
               turning_predicate: Callable,
               complementary_data: jnp.ndarray,
               is_walk: bool) -> TreeInfo:
    """
    Build a binary tree. This is the main NUTS tree building function.
    
    Args:
        leapfrog_fn: Leapfrog function (walk or side)
        kinetic_fn: Kinetic energy function
        z_initial: Initial position (dim,) - single chain
        r_initial: Initial momentum
        log_prob: Log probability function
        grad_log_prob: Gradient of log probability
        step_size: Step size
        rng_key: Random key
        max_delta_energy: Maximum energy difference
        max_tree_depth: Maximum tree depth
        turning_predicate: Turning predicate function
        complementary_data: Centered group2 (walk) or diff_particles (side)
        is_walk: Whether this is walk mode
        
    Returns:
        Built tree
    """
    # Compute initial energy
    potential_energy = -log_prob(z_initial)
    z_grad = -grad_log_prob(z_initial)
    kinetic_energy = kinetic_fn(r_initial, is_walk)
    energy_current = potential_energy + kinetic_energy
    
    # For walk mode, compute centering and inv_covariance for turning checks
    if is_walk:
        # complementary_data is centered group2 with shape (n_chains_per_group, dim)
        # centering should be shape (dim, n_chains_per_group) 
        centering = complementary_data.T  # Transpose to get (dim, n_chains_per_group)
        
        # Compute empirical covariance and inverse
        # Cov = (1/n) * complementary_data^T * complementary_data
        empirical_cov = jnp.dot(complementary_data.T, complementary_data) / complementary_data.shape[0]
        
        # Add regularization for numerical stability
        dim = z_initial.shape[0]
        regularization = 1e-6 * jnp.trace(empirical_cov) / dim * jnp.eye(dim)
        empirical_cov_reg = empirical_cov + regularization
        
        inv_covariance = jnp.linalg.inv(empirical_cov_reg)
    else:
        centering = None
        inv_covariance = None
    
    # Determine checkpoint array size based on position and momentum structure
    # Note: z_initial is a single chain's position with shape (dim,)
    # r_initial shape depends on mode: (n_chains_per_group,) for walk, () for side
    dim = z_initial.shape[0]
    latent_size_z = dim
    
    if is_walk:
        # Walk mode: position (dim,), momentum (n_chains_per_group,)
        latent_size_r = r_initial.shape[0]
    else:
        # Side mode: position (dim,), momentum scalar (shape ())
        latent_size_r = 1
    
    # Initialize checkpoints
    z_ckpts = jnp.zeros((max_tree_depth, latent_size_z))
    r_ckpts = jnp.zeros((max_tree_depth, latent_size_r))
    
    # Initialize tree
    tree = TreeInfo(
        z_initial, r_initial, z_grad,  # left
        z_initial, r_initial, z_grad,  # right
        z_initial, potential_energy, z_grad, energy_current,  # proposal
        0, jnp.zeros(()), r_initial, jnp.array(False), jnp.array(False),  # metadata
        jnp.zeros(()), jnp.array(0, dtype=jnp.result_type(int))
    )
    
    def cond_fn(state):
        tree, _ = state
        return (tree.depth < max_tree_depth) & ~tree.turning & ~tree.diverging
    
    def body_fn(state):
        tree, key = state
        key, direction_key, doubling_key = random.split(key, 3)
        going_right = random.bernoulli(direction_key)
        
        tree = double_tree(
            tree, leapfrog_fn, kinetic_fn,
            step_size, going_right, doubling_key, energy_current,
            max_delta_energy, z_ckpts, r_ckpts, turning_predicate, r_initial,
            log_prob, grad_log_prob, complementary_data, is_walk, centering, inv_covariance
        )
        return tree, key
    
    state = (tree, rng_key)
    tree, _ = lax.while_loop(cond_fn, body_fn, state)
    return tree


def nuts_step(key: jax.random.PRNGKey,
              group1: jnp.ndarray,
              group2: jnp.ndarray,
              log_prob: Callable,
              grad_log_prob: Callable,
              step_size: float,
              leapfrog: Callable,
              turning_predicate: Callable,
              max_tree_depth: int = 5,
              max_delta_energy: float = 1000.0,
              ) -> Tuple:
    """
    Perform one NUTS step for ensemble sampling.
    
    This operates on the entire group1 at once, with each chain running
    independent NUTS but sharing the complementary group2 structure.
    
    Args:
        key: Random key
        group1: Positions of group 1. Shape (n_chains_per_group, dim)
        group2: Positions of group 2. Shape (n_chains_per_group, dim)
        log_prob: Unnormalized log probability to sample from (vectorized)
        grad_log_prob: Gradient of log probability (vectorized)
        step_size: Step size
        leapfrog: Leapfrog function (walk or side)
        turning_predicate: Turning predicate function
        max_tree_depth: Maximum tree depth
        max_delta_energy: Maximum energy difference
        
    Returns:
        Tuple of (new_group1, log_accept_prob) where both have shape (n_chains_per_group,)
    """
    n_chains_per_group = int(group1.shape[0])
    dim = int(group1.shape[1])
    
    # Determine if this is walk or side mode
    is_walk = (leapfrog == hmc_walk_move or leapfrog.__name__ == 'leapfrog_walk_move')
    
    # Prepare complementary data for all chains
    if is_walk:
        # Walk mode: use centered group2
        centered2 = (group2 - jnp.mean(group2, axis=0)[None, :]) / jnp.sqrt(n_chains_per_group)
        complementary_data = centered2
        leapfrog_fn = leapfrog_walk_move
    else:
        # Side mode: use particle differences (one per chain)
        key, key_choices = jax.random.split(key)
        keys = jax.random.split(key_choices, n_chains_per_group)
        
        indices = jnp.arange(n_chains_per_group)
        choices = jax.vmap(
            lambda k: jax.random.choice(k, indices, shape=(2,), replace=False)
        )(keys)
        
        random_indices1 = choices[:, 0]
        random_indices2 = choices[:, 1]
        diff_particles = (group2[random_indices1] - group2[random_indices2]) / jnp.sqrt(2 * dim)
        complementary_data = diff_particles
        leapfrog_fn = leapfrog_side_move
    
    # Run NUTS for each chain in group1 independently
    def one_chain_nuts(z0: jnp.ndarray, subkey: jax.random.PRNGKey, comp_data_i):
        """Run NUTS for a single chain."""
        # Sample momentum
        if is_walk:
            # Walk: momentum is vector of length n_chains_per_group
            r0 = jax.random.normal(subkey, shape=(n_chains_per_group,))
            kinetic_fn = lambda r, is_w: 0.5 * jnp.sum(r**2)
            
            # For walk mode, all chains share the same centered2
            comp_data_chain = complementary_data
        else:
            # Side: momentum is scalar
            r0 = jax.random.normal(subkey, shape=())
            kinetic_fn = lambda r, is_w: 0.5 * r**2
            
            # For side mode, each chain has its own diff_particles
            comp_data_chain = comp_data_i
        
        # Wrap log_prob and grad for single chain
        def log_prob_single(z):
            return log_prob(z[None, :])[0]
        
        def grad_log_prob_single(z):
            return grad_log_prob(z[None, :])[0]
        
        # Build tree
        tree = build_tree(
            leapfrog_fn, kinetic_fn,
            z0, r0,
            log_prob_single, grad_log_prob_single,
            step_size, subkey,
            max_delta_energy, max_tree_depth,
            turning_predicate, comp_data_chain, is_walk
        )
        
        # Extract acceptance probability
        accept_prob = tree.sum_accept_probs / jnp.maximum(tree.num_proposals, 1)
        log_accept_prob = jnp.log(jnp.clip(accept_prob, 1e-10, 1.0))
        
        return tree.z_proposal, log_accept_prob
    
    # Split keys for each chain
    keys = jax.random.split(key, n_chains_per_group)
    
    # Vectorize across chains
    if is_walk:
        # For walk mode, complementary_data is shared (centered2)
        new_group1, log_accept_probs = jax.vmap(
            lambda z, k: one_chain_nuts(z, k, None)
        )(group1, keys)
    else:
        # For side mode, each chain has its own diff_particles row
        new_group1, log_accept_probs = jax.vmap(one_chain_nuts)(group1, keys, complementary_data)
    
    return new_group1, log_accept_probs


def select_nuts_step(move: Callable) -> Callable:
    """
    Select and configure NUTS step function based on move type.
    
    Args:
        move: HMC move function (hmc_walk_move or hmc_side_move)
        
    Returns:
        Configured NUTS step function
    """
    if move == hmc_walk_move or (hasattr(move, '__name__') and 'walk' in move.__name__):
        # Walk mode
        def configured_nuts_step(key, group1, group2, log_prob, grad_log_prob,
                                step_size, max_tree_depth=5, max_delta_energy=1000.0):
            return nuts_step(
                key, group1, group2, log_prob, grad_log_prob, step_size,
                leapfrog=leapfrog_walk_move,
                turning_predicate=walk_u_turn_condition,
                max_tree_depth=max_tree_depth,
                max_delta_energy=max_delta_energy
            )
        return configured_nuts_step
    
    elif move == hmc_side_move or (hasattr(move, '__name__') and 'side' in move.__name__):
        # Side mode
        def configured_nuts_step(key, group1, group2, log_prob, grad_log_prob,
                                step_size, max_tree_depth=5, max_delta_energy=1000.0):
            return nuts_step(
                key, group1, group2, log_prob, grad_log_prob, step_size,
                leapfrog=leapfrog_side_move,
                turning_predicate=side_u_turn_condition,
                max_tree_depth=max_tree_depth,
                max_delta_energy=max_delta_energy
            )
        return configured_nuts_step
    
    else:
        raise ValueError(f"Unknown move type: {move}")
