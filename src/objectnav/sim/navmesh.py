import habitat_sim
from objectnav.sim.config import NavmeshConfig

def compute_navmesh(sim: habitat_sim.Simulator) -> bool:
    """Compute the navigation mesh for the current scene in the simulator.
    Args:
        sim (habitat_sim.Simulator): The simulator instance.

    Returns:
        bool: True if the navigation mesh was successfully computed, False otherwise.
    """

    navmesh_config = NavmeshConfig()
    navmesh_settings = create_navmesh_settings(navmesh_config)

    # Recompute the navmesh for the current scene
    return sim.recompute_navmesh(sim.pathfinder, navmesh_settings)


def create_navmesh_settings(navmesh_config: NavmeshConfig) -> habitat_sim.NavMeshSettings:
    """
    Description:
        Creates and returns NavMesh settings for the Habitat simulator.

    Arguments:
        navmesh_config (NavmeshConfig): Configuration parameters for the navmesh.
    
    Returns:
        habitat_sim.NavMeshSettings: The NavMesh settings configured for the agent.
    """
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.agent_height = navmesh_config.agent_height
    navmesh_settings.agent_radius = navmesh_config.agent_radius
    navmesh_settings.agent_max_climb = navmesh_config.agent_max_climb
    navmesh_settings.agent_max_slope = navmesh_config.agent_max_slope
    navmesh_settings.include_static_objects = navmesh_config.include_static_objects
    navmesh_settings.cell_size = navmesh_config.cell_size
    navmesh_settings.cell_height = navmesh_config.cell_height
    navmesh_settings.filter_low_hanging_obstacles = navmesh_config.filter_low_hanging_obstacles
    navmesh_settings.filter_ledge_spans = navmesh_config.filter_ledge_spans
    navmesh_settings.filter_walkable_low_height_spans = navmesh_config.filter_walkable_low_height_spans
    
    return navmesh_settings