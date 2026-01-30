import random
from pathlib import Path
from objectnav.sim.simulator import make_sim
from objectnav.sim.navmesh import compute_navmesh
from objectnav.sim.maps import build_grid_map_from_navmesh
from objectnav.utils.visualization.maps import plot_map, plot_map_with_agent
from objectnav.sim.agent import init_agent

def main():
    # Pick a fixed seed (or read from CLI/env)
    # seed = 1234 # deterministic seed for testing
    seed = random.randint(0, 2**32 - 1) # random seed for non-deterministic runs

    # Launch simulator
    simulator = make_sim(scene_dataset_config=Path("datasets/ai2thor-hab/ai2thor-hab/ai2thor-hab.scene_dataset_config.json"), scene_id="FloorPlan1_physics")
        
    # Setting a seed for reproducibility    
    random.seed(seed) # This will make random sampling reproducible (eg. yaw degree)
    simulator.sim.seed(seed)
    simulator.sim.pathfinder.seed(seed)

    # Get navigation mesh
    if compute_navmesh(simulator.sim):
        grid_map = build_grid_map_from_navmesh(simulator.sim)
        plot_map(grid_map, title="Grid Map", save_path="outputs/grid_map.png")
    else:
        raise RuntimeError("Failed to compute navigation mesh.")

    # Initialize the (only) agent: configuration was made in simulator creation
    agent = init_agent(simulator.sim)

    # Get agent state
    agent_state = agent.get_state()
    print("agent_state: position", type(agent_state.position) , agent_state.position, "rotation", type(agent_state.rotation), agent_state.rotation)
    # print("sensor_states:", agent_state.sensor_states, type(agent_state.sensor_states))
    
    # Plot initial agent position on the map
    plot_map_with_agent(
        grid_map,
        agent_state.position,
        agent_state.rotation,
        sim=simulator.sim,
        title="Grid Map + Agent",
        save_path="outputs/grid_map_with_agent.png",
        agent_radius_px=20,
    )

    # Print navmesh AABB (both corners)
    bounds_min, bounds_max = simulator.sim.pathfinder.get_bounds()
    print(f"Navmesh bounds min (x,y,z): {bounds_min}")
    print(f"Navmesh bounds max (x,y,z): {bounds_max}")

    # Reset simulator
    print("Resetting simulator...")
    simulator.reset() 

    # Close simulator
    print("Closing simulator...")
    simulator.close()


if __name__ == "__main__":
    main()