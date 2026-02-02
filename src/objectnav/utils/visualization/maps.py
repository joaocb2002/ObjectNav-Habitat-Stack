import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple, Union

from objectnav.utils.spatial.rotations import quaternion_to_yaw
from habitat.utils.visualizations import maps as habitat_maps

Position3D = Union[Sequence[float], np.ndarray]
GridCoord = Tuple[int, int]

def plot_map(
	grid_map: np.ndarray,
	title: str = None,
	palette: np.ndarray = np.array(
		[[240, 240, 240], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
	),
	show_axis: bool = False,
	save_path: str = None,
	figsize: tuple = (8, 8),
	ax: plt.Axes = None,
) -> None:
	"""
	Plot a 2D grid map (numpy array) as an RGB image using a color palette.

	The input map should be a 2D array of integer labels (e.g., 0=occupied, 1=unoccupied, 2=border).
	The function recolors the map to a 3D RGB image using the provided palette and displays it.

	Args:
		grid_map (np.ndarray): 2D array representing the map (integer labels).
		title (str, optional): Title for the plot.
		palette (np.ndarray, optional): Array of shape (N, 3) with RGB values for each label.
			Default: white for 0, gray for 1, black for 2.
		show_axis (bool, optional): Whether to show axis ticks/labels. Default is False.
		save_path (str, optional): If provided, saves the plot to this path.
		figsize (tuple, optional): Figure size. Default is (8, 8).
		ax (plt.Axes, optional): Matplotlib Axes to plot on. If None, creates a new figure.
	"""
	rgb_map = recolor_map(grid_map, palette)
	created_fig = ax is None
	if created_fig:
		_, ax = plt.subplots(figsize=figsize)
	_ = ax.imshow(rgb_map)
	if title:
		ax.set_title(title)
	if not show_axis:
		ax.axis("off")
	if save_path:
		plt.savefig(save_path, bbox_inches="tight")
	if created_fig:
		plt.show()


def plot_map_with_agent(
	grid_map: np.ndarray,
	agent_position: Union[Position3D, GridCoord],
	agent_rotation: Optional[object] = None,
	*,
	sim: Optional[object] = None,
	pathfinder: Optional[object] = None,
	title: Optional[str] = None,
	palette: np.ndarray = np.array(
		[[240, 240, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
	),
	show_axis: bool = False,
	save_path: Optional[str] = None,
	figsize: Tuple[int, int] = (8, 8),
	ax: Optional[plt.Axes] = None,
	agent_radius_px: int = 5,
) -> None:
	"""Plot a labeled grid map with the agent composited on top.

	Uses Habitat's visualization helpers:
	- `habitat_maps.to_grid` to convert world (x, y, z) into map coordinates.
	- `habitat_maps.draw_agent` to draw the agent marker + orientation.

	Args:
		grid_map: 2D int label map (e.g., from `habitat_maps.get_topdown_map`).
		agent_position: Either a world position `[x, y, z]` or a grid coordinate `(row, col)`.
		agent_rotation: Either a yaw angle in radians (float) or a quaternion-like
			object with attributes `w,x,y,z` (e.g. `quaternion.quaternion`). If None,
			yaw defaults to 0.
		sim: Optional Habitat simulator. If provided, `sim.pathfinder` will be used.
		pathfinder: Optional Habitat pathfinder. Required if `agent_position` is world
			coordinates and `sim` is not provided.
	"""
	if grid_map.ndim != 2:
		raise ValueError("grid_map must be a 2D array of integer labels.")

	if pathfinder is None and sim is not None and hasattr(sim, "pathfinder"):
		pathfinder = sim.pathfinder

	# Convert position
	agent_center: GridCoord
	if isinstance(agent_position, tuple) and len(agent_position) == 2:
		agent_center = (int(agent_position[0]), int(agent_position[1]))
	elif isinstance(agent_position, (list, np.ndarray, tuple)) and len(agent_position) == 3:
		if pathfinder is None and sim is None:
			raise ValueError(
				"agent_position looks like world coordinates; pass `sim=` or `pathfinder=` "
				"so it can be converted via habitat_maps.to_grid()."
			)
		x, _, z = (float(agent_position[0]), float(agent_position[1]), float(agent_position[2]))
		# Habitat uses (z, x) for topdown/grid conversions.
		agent_center = tuple(
			int(v)
			for v in habitat_maps.to_grid(
				z,
				x,
				(grid_map.shape[0], grid_map.shape[1]),
				sim=sim,
				pathfinder=pathfinder,
			)
		)
	else:
		raise ValueError(
			"agent_position must be a grid coord (row, col) or a world position (x, y, z)."
		)

	# Convert rotation to yaw (radians)
	if agent_rotation is None:
		yaw_rad = 0.0
	elif isinstance(agent_rotation, (int, float, np.floating)):
		yaw_rad = float(agent_rotation)
	else:
		yaw_rad = float(quaternion_to_yaw(agent_rotation, degrees=False))

	rgb_map = recolor_map(grid_map, palette)
	# draw_agent modifies the image in-place; return value is the same array.
	rgb_map = habitat_maps.draw_agent(
		rgb_map,
		agent_center_coord=agent_center,
		agent_rotation=yaw_rad,
		agent_radius_px=int(agent_radius_px),
	)

	created_fig = ax is None
	if created_fig:
		_, ax = plt.subplots(figsize=figsize)
	_ = ax.imshow(rgb_map, origin="upper")
	if title:
		ax.set_title(title)
	if not show_axis:
		ax.axis("off")
	if save_path:
		plt.savefig(save_path, bbox_inches="tight")
	if created_fig:
		plt.show()


def recolor_map(
	label_map: np.ndarray,
	palette: np.ndarray = np.array(
		[[240, 240, 240], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
	),
) -> np.ndarray:
	"""
	Convert a labeled map (with integer values) to an RGB image using a color palette.

	Args:
		label_map (np.ndarray): 2D array with integer labels (e.g., 0=occupied, 1=unoccupied, 2=border).
		palette (np.ndarray): Array of shape (N, 3) with RGB values for each label.
			Default: white for 0, gray for 1, black for 2.

	Returns:
		np.ndarray: 3D array (H, W, 3) with RGB image.
	"""
	if label_map.ndim != 2:
		raise ValueError("label_map must be a 2D array of integer labels.")
	if not np.issubdtype(label_map.dtype, np.integer):
		raise ValueError("label_map must have integer dtype.")
	if np.max(label_map) >= palette.shape[0]:
		raise ValueError("label_map contains label(s) outside the palette range.")
	return palette[label_map]