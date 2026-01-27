from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class SimConfig:
    """
    Simulation/environment and agent configuration constants.
    Prefer overriding via a run config (YAML) rather than editing code.
    """

    # --- Simulation parameters ---
    scene_dataset_config: Optional[Path] = field(default=None, metadata={"help": "Path to scene dataset config file."})
    scene_id: Optional[str] = field(default=None, metadata={"help": "Scene ID to load."})
    enable_physics: bool = field(default=False, metadata={"help": "Enable physics simulation."})

    # --- Sensor parameters ---
    obs_scale: float = field(default=9/16, metadata={"help": "Aspect ratio for observations (height/width)."})
    rgb_width: int = field(default=1024, metadata={"help": "Width of RGB sensor output."})
    rgb_height: int = field(init=False, metadata={"help": "Height of RGB sensor output, computed from obs_scale."})
    sensor_height: float = field(default=1.5, metadata={"help": "Sensor height from ground in meters."})
    hfov_deg: float = field(default=90.0, metadata={"help": "Horizontal field of view in degrees."})

    # --- Action space magnitudes ---
    forward_step: float = field(default=0.25, metadata={"help": "Forward step size in meters."})
    turn_deg: float = field(default=30.0, metadata={"help": "Turn angle in degrees."})

    def __post_init__(self):
        # Set rgb_height based on obs_scale and rgb_width
        object.__setattr__(self, 'rgb_height', int(self.rgb_width * self.obs_scale))


@dataclass(frozen=True)
class NavmeshConfig:
    """
    Navigation mesh configuration constants.
    Prefer overriding via a run config (YAML) rather than editing code.
    """
    
    # --- Navmesh parameters ---
    include_static_objects: bool = field(default=True, metadata={"help": "Include static objects in navmesh generation."})
    cell_size: float = field(default=0.05, metadata={"help": "Cell size for navmesh generation in meters."})
    cell_height: float = field(default=0.1, metadata={"help": "Cell height for navmesh generation in meters."})
    filter_low_hanging_obstacles: bool = field(default=True, metadata={"help": "Filter low hanging obstacles during navmesh generation."})
    filter_ledge_spans: bool = field(default=True, metadata={"help": "Filter ledge spans during navmesh generation."})
    filter_walkable_low_height_spans: bool = field(default=True, metadata={"help": "Filter walkable low height spans during navmesh generation."})
   
    # --- Agent geometry ---
    agent_height: float = field(default=1.5, metadata={"help": "Height of the agent in meters."})
    agent_radius: float = field(default=0.20, metadata={"help": "Radius of the agent in meters."})
    agent_max_climb: float = field(default=0.2, metadata={"help": "Maximum climbable height in meters."})
    agent_max_slope: float = field(default=45.0, metadata={"help": "Maximum navigable slope in degrees."})


@dataclass(frozen=True)
class GridMapConfig:
    """
    Grid map configuration constants.
    Prefer overriding via a run config (YAML) rather than editing code.
    """

    # --- Grid map parameters ---
    map_resolution: float = field(default=1024, metadata={"help": "Length of the longest side of the map. Used to calculate meters_per_pixel."})
    height: float = field(default=0.0, metadata={"help": "The height in the environment to make the topdown map from."})
    meters_per_pixel: float = field(init=False, metadata={"help": "Meters per pixel for the grid map, computed from map_resolution."})
    draw_border: bool = field(default=True, metadata={"help": "Whether to draw a border around the map."})