import habitat_sim
from pathlib import Path
from objectnav.sim.config import SimConfig

class SimulatorWrapper:
    """A wrapper for habitat_sim.Simulator with configuration utilities."""

    def __init__(self, sim_config: SimConfig):
        """Initialize the simulator with the given configuration."""
        self.cfg = self._make_cfg(sim_config)
        self.sim = habitat_sim.Simulator(self.cfg)

    def reset(self):
        """Reset the simulator to the initial state."""
        self.sim.reset()

    def close(self):
        """Close the simulator and release resources."""
        self.sim.close()

    def step(self, action):
        """Take a simulation step with the given action."""
        return self.sim.step(action)

    @staticmethod
    def _make_cfg(sim_config: SimConfig):
        # Simulator configuration
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        if not sim_config.scene_dataset_config:
            raise ValueError("sim_config.scene_dataset_config must be provided.")
        if not sim_config.scene_id:
            raise ValueError("sim_config.scene_id must be provided.")
        sim_cfg.scene_dataset_config_file = str(sim_config.scene_dataset_config)
        sim_cfg.scene_id = sim_config.scene_id
        sim_cfg.enable_physics = sim_config.enable_physics
        sim_cfg.allow_sliding = sim_config.allow_sliding

        # Agents sensor and action space (no geometry needed here)
        sensors = {
            "color_sensor": {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": [sim_config.rgb_height, sim_config.rgb_width],
                "position": [0.0, sim_config.sensor_height, 0.0],
            },
            "depth_sensor": {
                "sensor_type": habitat_sim.SensorType.DEPTH,
                "resolution": [sim_config.rgb_height, sim_config.rgb_width],
                "position": [0.0, sim_config.sensor_height, 0.0],
            },
        }
        sensor_specs = []
        for sensor_uuid, params in sensors.items():
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = sensor_uuid
            spec.sensor_type = params["sensor_type"]
            spec.resolution = params["resolution"]
            spec.position = params["position"]
            sensor_specs.append(spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=sim_config.forward_step)),
            "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=sim_config.turn_deg)),
            "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=sim_config.turn_deg)),
        }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def make_sim(scene_dataset_config: Path, scene_id: str) -> SimulatorWrapper:
    """
    Create a Simulator instance using SimConfig, overriding dataset and scene.
    Raises:
        TypeError: if argument types are incorrect.
        ValueError: if arguments are missing.
    """
    if scene_dataset_config is None:
        raise ValueError("scene_dataset_config must be provided (got None).")
    if not isinstance(scene_dataset_config, Path):
        raise TypeError(f"scene_dataset_config must be a pathlib.Path, got {type(scene_dataset_config).__name__}.")
    if scene_id is None:
        raise ValueError("scene_id must be provided (got None).")
    if not isinstance(scene_id, str):
        raise TypeError(f"scene_id must be a str, got {type(scene_id).__name__}.")

    sim_config = SimConfig(scene_dataset_config=scene_dataset_config, scene_id=scene_id)
    return SimulatorWrapper(sim_config)