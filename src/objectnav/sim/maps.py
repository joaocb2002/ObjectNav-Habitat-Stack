import numpy as np
import habitat_sim
from objectnav.sim.config import GridMapConfig
from habitat.utils.visualizations import maps

def build_grid_map_from_navmesh(sim: habitat_sim.Simulator) -> np.ndarray:
    """
    Builds a grid map (top-down occupancy map) from the simulator's navigation mesh.

    The returned map is a 2D numpy array where:
        0 = occupied (obstacle)
        1 = unoccupied (free space)
        2 = border

    Args:
        sim (habitat_sim.Simulator): The simulator instance.

    Returns:
        np.ndarray: A 2D numpy array representing the top-down occupancy map with values 0, 1, or 2.
    """

    # Ensure the simulator has a pathfinder
    if not hasattr(sim, "pathfinder"):
        raise AttributeError("Simulator does not have a pathfinder attribute.")
    pathfinder = sim.pathfinder

    # Get grid map configuration
    grid_map_config = GridMapConfig()

    # Generate the top-down map using Habitat's utility
    topdown_map = maps.get_topdown_map(pathfinder, grid_map_config.height, grid_map_config.map_resolution, grid_map_config.draw_border)

    return topdown_map