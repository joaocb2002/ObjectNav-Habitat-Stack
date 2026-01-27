import numpy as np
import matplotlib.pyplot as plt

def plot_map(
	grid_map: np.ndarray,
	title: str = None,
	palette: np.ndarray = np.array(
		[[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
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
	if ax is None:
		fig, ax = plt.subplots(figsize=figsize)
	im = ax.imshow(rgb_map, origin="lower")
	if title:
		ax.set_title(title)
	if not show_axis:
		ax.axis("off")
	if save_path:
		plt.savefig(save_path, bbox_inches="tight")
	if ax is None or ax.figure is plt.gcf():
		plt.show()


def recolor_map(
	label_map: np.ndarray,
	palette: np.ndarray = np.array(
		[[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
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