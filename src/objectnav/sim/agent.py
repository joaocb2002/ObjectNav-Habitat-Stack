from __future__ import annotations

import random
from typing import Optional, Sequence

import numpy as np
import habitat_sim
import habitat

from objectnav.utils.spatial.rotations import yaw_to_quaternion

def init_agent(
    sim: habitat_sim.Simulator,
    position: Optional[Sequence[float]] = None,
    yaw_degrees: Optional[float] = None,
    agent_id: int = 0,
) -> habitat_sim.Agent:
    """
    Initializes and returns an agent from a Habitat simulator object.

    This function wraps `sim.initialize_agent(agent_id)` and then sets the
    agent state (position + rotation).

    The agent orientation is constrained to be orthogonal to the floor plane:
    only yaw (heading) is applied; pitch and roll are zero.

    Args:
        sim: The Habitat simulator object (habitat_sim.Simulator).
        position: Optional agent position in world space as a length-3 array-like
            `[x, y, z]`. If None, a random navigable point is sampled via the
            simulator pathfinder.
        yaw_degrees: Optional agent yaw rotation (heading) in degrees. If None,
            a random yaw is sampled uniformly from `[-180, 180)`.
        agent_id: Habitat agent index to initialize (defaults to `0`).

    Returns:
        The initialized agent object (habitat_sim.Agent).

    Raises:
        ValueError: If `position` is not length 3, if `yaw_degrees` is not finite,
            if the agent cannot stand at `position`, or if `position` is not on the
            largest non-outdoor navmesh island.
        RuntimeError: If the pathfinder/navmesh is not loaded, or if no valid
            non-outdoor island can be found for placement.
    """
    if not hasattr(sim, "pathfinder"):
        raise AttributeError("Simulator does not have a pathfinder attribute.")

    largest_island_index = habitat.datasets.rearrange.navmesh_utils.get_largest_island_index(
        sim.pathfinder,
        sim,
        allow_outdoor=False,
    )
    if largest_island_index is None or int(largest_island_index) < 0:
        raise RuntimeError(
            "Could not determine a valid non-outdoor navmesh island for placement "
            f"(got {largest_island_index})."
        )
    largest_island_index = int(largest_island_index)

    if position is None:
        # Sample directly from the selected island to avoid outdoor regions.
        position_arr = np.asarray(
            sim.pathfinder.get_random_navigable_point(
                max_tries=100,
                island_index=largest_island_index,
            ),
            dtype=np.float32,
        )
    else:
        position_arr = np.asarray(position, dtype=np.float32)

    if position_arr.shape != (3,):
        raise ValueError(f"position must be length 3, got shape {position_arr.shape}")

    if not sim.pathfinder.is_navigable(position_arr):
        raise ValueError(f"position {position_arr.tolist()} is not navigable")

    position_island_index = int(sim.pathfinder.get_island(position_arr))
    if position_island_index != largest_island_index:
        raise ValueError(
            "position is on a different navmesh island than the largest non-outdoor island "
            f"(position island={position_island_index}, expected={largest_island_index})"
        )

    if yaw_degrees is None:
        # Uniform in [-180, 180) without ever returning 180.
        yaw_degrees = (random.random() * 360.0) - 180.0
    else:
        if not np.isfinite(float(yaw_degrees)):
            raise ValueError(f"yaw_degrees must be finite, got {yaw_degrees}")
        # Canonicalize to [-180, 180).
        yaw_degrees = ((float(yaw_degrees) + 180.0) % 360.0) - 180.0

    rotation_q = yaw_to_quaternion(float(yaw_degrees), degrees=True)

    agent = sim.initialize_agent(agent_id)

    agent_state = habitat_sim.AgentState()
    agent_state.position = position_arr
    agent_state.rotation = rotation_q

    # Keep sensors consistent with the newly-set pose (standard Habitat pattern).
    try:
        agent.set_state(agent_state, reset_sensors=True)
    except TypeError:
        agent.set_state(agent_state)

    return agent