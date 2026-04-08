
"""Quick Waymo v1 TFRecord inspector for one segment.

This script loads the first frame from one segment and extracts:
- camera data (decoded RGB images)
- ego_pose (4x4 world-from-vehicle transform)
- map_features (static map proto objects)
- lidar_box (from frame.laser_labels)

Usage:
  python /home/yw4142/ad/r2dreamer/test/test_waymo_perception_data.py
"""

from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib.pyplot as plt
import numpy as np


PATH = "/scratch/yw4142/datasets/ad/waymo_open_dataset_v_1_4_3/training"
parquet_PATH_camera_image = "/scratch/yw4142/datasets/ad/waymo_open_dataset_v_2_0_1/training/camera_image"
parquet_PATH_lidar_box = "/scratch/yw4142/datasets/ad/waymo_open_dataset_v_2_0_1/training/lidar_box"
parquet_PATH_map_features = "/scratch/yw4142/datasets/ad/waymo_open_dataset_v_2_0_1/training/map_features"
parquet_PATH_vehicle_pose = "/scratch/yw4142/datasets/ad/waymo_open_dataset_v_2_0_1/training/vehicle_pose"
OUTPUT_DIR = Path("/home/yw4142/ad/r2dreamer/test/waymo_vis_outputs/parquet")
segment_lists = ['10023947602400723454_1120_000_1140_000', '10017090168044687777_6380_000_6400_000']
MAX_FRAMES = None  # None means use all frames in each segment.
ENTER_PDB = True

CAMERA_NAME_TO_LABEL = {
	1: "FRONT",
	2: "FRONT_LEFT",
	3: "FRONT_RIGHT",
	4: "SIDE_LEFT",
	5: "SIDE_RIGHT",
}

FRAME_TIMESTAMP_COL = "key.frame_timestamp_micros"
CAMERA_NAME_COL = "key.camera_name"
CAMERA_IMAGE_COL = "[CameraImageComponent].image"
VEHICLE_POSE_TRANSFORM_COL = "[VehiclePoseComponent].world_from_vehicle.transform"
LIDAR_BOX_CENTER_X_COL = "[LiDARBoxComponent].box.center.x"
LIDAR_BOX_CENTER_Y_COL = "[LiDARBoxComponent].box.center.y"
LIDAR_BOX_CENTER_Z_COL = "[LiDARBoxComponent].box.center.z"
LIDAR_BOX_SIZE_X_COL = "[LiDARBoxComponent].box.size.x"
LIDAR_BOX_SIZE_Y_COL = "[LiDARBoxComponent].box.size.y"
LIDAR_BOX_SIZE_Z_COL = "[LiDARBoxComponent].box.size.z"
LIDAR_BOX_HEADING_COL = "[LiDARBoxComponent].box.heading"
LIDAR_BOX_TYPE_COL = "[LiDARBoxComponent].type"
LIDAR_BOX_ID_COL = "key.laser_object_id"

MAP_FEATURE_TYPE_COL = "[MapFeatureComponent].feature_type"
MAP_LANE_POLYLINE_X_COL = "[MapFeatureComponent].lane.polyline.x"
MAP_LANE_POLYLINE_Y_COL = "[MapFeatureComponent].lane.polyline.y"
MAP_ROAD_EDGE_POLYLINE_X_COL = "[MapFeatureComponent].road_edge.polyline.x"
MAP_ROAD_EDGE_POLYLINE_Y_COL = "[MapFeatureComponent].road_edge.polyline.y"
MAP_ROAD_LINE_POLYLINE_X_COL = "[MapFeatureComponent].road_line.polyline.x"
MAP_ROAD_LINE_POLYLINE_Y_COL = "[MapFeatureComponent].road_line.polyline.y"
MAP_STOP_SIGN_POSITION_X_COL = "[MapFeatureComponent].stop_sign.position.x"
MAP_STOP_SIGN_POSITION_Y_COL = "[MapFeatureComponent].stop_sign.position.y"
MAP_CROSSWALK_POLYGON_X_COL = "[MapFeatureComponent].crosswalk.polygon.x"
MAP_CROSSWALK_POLYGON_Y_COL = "[MapFeatureComponent].crosswalk.polygon.y"
MAP_DRIVEWAY_POLYGON_X_COL = "[MapFeatureComponent].driveway.polygon.x"
MAP_DRIVEWAY_POLYGON_Y_COL = "[MapFeatureComponent].driveway.polygon.y"
MAP_SPEED_BUMP_POLYGON_X_COL = "[MapFeatureComponent].speed_bump.polygon.x"
MAP_SPEED_BUMP_POLYGON_Y_COL = "[MapFeatureComponent].speed_bump.polygon.y"

def import_waymo_runtime():
	"""Import tensorflow runtime with eager mode enabled."""
	import tensorflow.compat.v1 as tf

	if hasattr(tf, "executing_eagerly") and not tf.executing_eagerly():
		tf.enable_eager_execution()
	return tf, None

def import_parquet_runtime():
	"""Import pyarrow parquet runtime with a clear error when unavailable."""
	try:
		import pyarrow.parquet as pq
	except ModuleNotFoundError as exc:
		raise ModuleNotFoundError(
			"pyarrow is required for parquet loading. Install with `pip install pyarrow`."
		) from exc
	return pq

def find_segment_file(root: Path) -> Path:
	"""Find one parquet segment file under the camera component directory."""
	if root.is_file() and root.suffix == ".parquet":
		return root

	candidates = sorted(p for p in root.rglob("*.parquet") if p.is_file())
	if not candidates:
		raise FileNotFoundError(f"No parquet segment file found under {root}")
	return candidates[0]

def list_segment_files(root: Path) -> list[Path]:
	"""List segment ids shared across all parquet components as synthetic paths."""
	components = {
		"camera_image": Path(parquet_PATH_camera_image),
		"lidar_box": Path(parquet_PATH_lidar_box),
		"map_features": Path(parquet_PATH_map_features),
		"vehicle_pose": Path(parquet_PATH_vehicle_pose),
	}
	segment_sets = []
	for component_name, component_dir in components.items():
		if not component_dir.is_dir():
			raise FileNotFoundError(f"Missing parquet component directory: {component_name} -> {component_dir}")
		segment_sets.append({path.stem for path in component_dir.glob("*.parquet")})

	shared = sorted(set.intersection(*segment_sets)) if segment_sets else []
	if segment_lists:
		allowed = set(segment_lists)
		shared = [segment_id for segment_id in shared if segment_id in allowed]
	if not shared:
		raise FileNotFoundError("No shared parquet segments found across camera/lidar/map/pose components")

	camera_dir = components["camera_image"]
	return [camera_dir / f"{segment_id}.parquet" for segment_id in shared]

def segment_stem(segment_path: Path) -> str:
	"""Normalize segment file name to a compact segment id for filenames."""
	name = segment_path.name
	for suffix in (".parquet", ".tfrecord.gz", ".tfrecord.gzip", ".tfrecords", ".tfrecord"):
		if name.endswith(suffix):
			name = name[: -len(suffix)]
			break
	return name

def load_first_frame(segment_path: Path, tf, open_dataset):
	"""Load the first frame from one segment."""
	dataset = tf.data.TFRecordDataset(
		[str(segment_path)],
		compression_type="",
		buffer_size=8 << 20,
		num_parallel_reads=1,
	)
	for raw in dataset:
		frame = open_dataset.Frame()
		frame.ParseFromString(raw.numpy())
		return frame
	raise RuntimeError(f"Segment has no frames: {segment_path}")

def iter_segment_frames(segment_path: Path, tf, open_dataset, max_frames=None):
	"""Yield frame-like objects assembled from parquet component tables."""
	pq = import_parquet_runtime()
	segment_id = segment_stem(segment_path)

	camera_table = pq.read_table(
		Path(parquet_PATH_camera_image) / f"{segment_id}.parquet",
		columns=[FRAME_TIMESTAMP_COL, CAMERA_NAME_COL, CAMERA_IMAGE_COL],
		use_threads=True,
		memory_map=True,
	)
	pose_table = pq.read_table(
		Path(parquet_PATH_vehicle_pose) / f"{segment_id}.parquet",
		columns=[FRAME_TIMESTAMP_COL, VEHICLE_POSE_TRANSFORM_COL],
		use_threads=True,
		memory_map=True,
	)
	lidar_schema = set(pq.read_schema(Path(parquet_PATH_lidar_box) / f"{segment_id}.parquet").names)
	lidar_columns = [
		FRAME_TIMESTAMP_COL,
		LIDAR_BOX_CENTER_X_COL,
		LIDAR_BOX_CENTER_Y_COL,
		LIDAR_BOX_SIZE_X_COL,
		LIDAR_BOX_SIZE_Y_COL,
		LIDAR_BOX_HEADING_COL,
	]
	for optional_col in (LIDAR_BOX_CENTER_Z_COL, LIDAR_BOX_SIZE_Z_COL, LIDAR_BOX_TYPE_COL, LIDAR_BOX_ID_COL):
		if optional_col in lidar_schema:
			lidar_columns.append(optional_col)
	lidar_table = pq.read_table(
		Path(parquet_PATH_lidar_box) / f"{segment_id}.parquet",
		columns=lidar_columns,
		use_threads=True,
		memory_map=True,
	)

	map_schema = set(pq.read_schema(Path(parquet_PATH_map_features) / f"{segment_id}.parquet").names)
	map_columns = [
		MAP_FEATURE_TYPE_COL,
		MAP_LANE_POLYLINE_X_COL,
		MAP_LANE_POLYLINE_Y_COL,
		MAP_ROAD_EDGE_POLYLINE_X_COL,
		MAP_ROAD_EDGE_POLYLINE_Y_COL,
		MAP_ROAD_LINE_POLYLINE_X_COL,
		MAP_ROAD_LINE_POLYLINE_Y_COL,
		MAP_STOP_SIGN_POSITION_X_COL,
		MAP_STOP_SIGN_POSITION_Y_COL,
		MAP_CROSSWALK_POLYGON_X_COL,
		MAP_CROSSWALK_POLYGON_Y_COL,
		MAP_DRIVEWAY_POLYGON_X_COL,
		MAP_DRIVEWAY_POLYGON_Y_COL,
		MAP_SPEED_BUMP_POLYGON_X_COL,
		MAP_SPEED_BUMP_POLYGON_Y_COL,
	]
	map_columns = [col for col in map_columns if col in map_schema]
	map_table = pq.read_table(
		Path(parquet_PATH_map_features) / f"{segment_id}.parquet",
		columns=map_columns,
		use_threads=True,
		memory_map=True,
	)

	def _col_pylist(table, name, default=None):
		if name not in table.column_names:
			return default
		return table.column(name).to_pylist()

	def _xy_points(xs, ys):
		if xs is None or ys is None:
			return []
		return [SimpleNamespace(x=float(x), y=float(y)) for x, y in zip(xs, ys)]

	types = _col_pylist(map_table, MAP_FEATURE_TYPE_COL, default=[])
	lane_x = _col_pylist(map_table, MAP_LANE_POLYLINE_X_COL)
	lane_y = _col_pylist(map_table, MAP_LANE_POLYLINE_Y_COL)
	road_edge_x = _col_pylist(map_table, MAP_ROAD_EDGE_POLYLINE_X_COL)
	road_edge_y = _col_pylist(map_table, MAP_ROAD_EDGE_POLYLINE_Y_COL)
	road_line_x = _col_pylist(map_table, MAP_ROAD_LINE_POLYLINE_X_COL)
	road_line_y = _col_pylist(map_table, MAP_ROAD_LINE_POLYLINE_Y_COL)
	stop_sign_x = _col_pylist(map_table, MAP_STOP_SIGN_POSITION_X_COL)
	stop_sign_y = _col_pylist(map_table, MAP_STOP_SIGN_POSITION_Y_COL)
	crosswalk_x = _col_pylist(map_table, MAP_CROSSWALK_POLYGON_X_COL)
	crosswalk_y = _col_pylist(map_table, MAP_CROSSWALK_POLYGON_Y_COL)
	driveway_x = _col_pylist(map_table, MAP_DRIVEWAY_POLYGON_X_COL)
	driveway_y = _col_pylist(map_table, MAP_DRIVEWAY_POLYGON_Y_COL)
	speed_bump_x = _col_pylist(map_table, MAP_SPEED_BUMP_POLYGON_X_COL)
	speed_bump_y = _col_pylist(map_table, MAP_SPEED_BUMP_POLYGON_Y_COL)

	map_features = []
	n_map_rows = map_table.num_rows
	for idx in range(n_map_rows):
		feature_type = str(types[idx]).lower() if idx < len(types) and types[idx] is not None else "unknown"

		feat = SimpleNamespace()
		feat.lane = SimpleNamespace(polyline=_xy_points(lane_x[idx], lane_y[idx])) if lane_x and lane_y else SimpleNamespace(polyline=[])
		feat.road_edge = (
			SimpleNamespace(polyline=_xy_points(road_edge_x[idx], road_edge_y[idx]))
			if road_edge_x and road_edge_y
			else SimpleNamespace(polyline=[])
		)
		feat.road_line = (
			SimpleNamespace(polyline=_xy_points(road_line_x[idx], road_line_y[idx]))
			if road_line_x and road_line_y
			else SimpleNamespace(polyline=[])
		)
		feat.stop_sign = (
			SimpleNamespace(position=SimpleNamespace(x=float(stop_sign_x[idx]), y=float(stop_sign_y[idx])))
			if stop_sign_x and stop_sign_y and stop_sign_x[idx] is not None and stop_sign_y[idx] is not None
			else SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0))
		)
		feat.crosswalk = (
			SimpleNamespace(polygon=_xy_points(crosswalk_x[idx], crosswalk_y[idx]))
			if crosswalk_x and crosswalk_y
			else SimpleNamespace(polygon=[])
		)
		feat.driveway = (
			SimpleNamespace(polygon=_xy_points(driveway_x[idx], driveway_y[idx]))
			if driveway_x and driveway_y
			else SimpleNamespace(polygon=[])
		)
		feat.speed_bump = (
			SimpleNamespace(polygon=_xy_points(speed_bump_x[idx], speed_bump_y[idx]))
			if speed_bump_x and speed_bump_y
			else SimpleNamespace(polygon=[])
		)

		if feature_type not in {
			"lane",
			"road_edge",
			"road_line",
			"stop_sign",
			"crosswalk",
			"driveway",
			"speed_bump",
		}:
			if feat.lane.polyline:
				feature_type = "lane"
			elif feat.road_edge.polyline:
				feature_type = "road_edge"
			elif feat.road_line.polyline:
				feature_type = "road_line"
			elif stop_sign_x and stop_sign_y and stop_sign_x[idx] is not None and stop_sign_y[idx] is not None:
				feature_type = "stop_sign"
			elif feat.crosswalk.polygon:
				feature_type = "crosswalk"
			elif feat.driveway.polygon:
				feature_type = "driveway"
			elif feat.speed_bump.polygon:
				feature_type = "speed_bump"

		feat._feature_type = feature_type
		feat.WhichOneof = lambda _, _feat=feat: _feat._feature_type
		map_features.append(feat)

	camera_by_timestamp = {}
	cam_ts = np.asarray(camera_table.column(FRAME_TIMESTAMP_COL).to_numpy(), dtype=np.int64)
	cam_name = np.asarray(camera_table.column(CAMERA_NAME_COL).to_numpy(), dtype=np.int64)
	cam_img = camera_table.column(CAMERA_IMAGE_COL).to_pylist()
	for ts, name, image in zip(cam_ts.tolist(), cam_name.tolist(), cam_img):
		camera_by_timestamp.setdefault(int(ts), []).append(SimpleNamespace(name=int(name), image=image))
	for ts in list(camera_by_timestamp.keys()):
		camera_by_timestamp[ts].sort(key=lambda img: int(img.name))

	pose_by_timestamp = {}
	pose_ts = np.asarray(pose_table.column(FRAME_TIMESTAMP_COL).to_numpy(), dtype=np.int64)
	pose_tf = pose_table.column(VEHICLE_POSE_TRANSFORM_COL).to_pylist()
	for ts, transform in zip(pose_ts.tolist(), pose_tf):
		if int(ts) in pose_by_timestamp:
			continue
		pose_by_timestamp[int(ts)] = np.asarray(transform, dtype=np.float64).reshape(4, 4).reshape(-1).tolist()

	def _optional_col_array(table, col_name, default_value=0.0):
		if col_name in table.column_names:
			return table.column(col_name).to_pylist()
		return [default_value] * table.num_rows

	lidar_by_timestamp = {}
	lid_ts = np.asarray(lidar_table.column(FRAME_TIMESTAMP_COL).to_numpy(), dtype=np.int64)
	lid_x = lidar_table.column(LIDAR_BOX_CENTER_X_COL).to_pylist()
	lid_y = lidar_table.column(LIDAR_BOX_CENTER_Y_COL).to_pylist()
	lid_z = _optional_col_array(lidar_table, LIDAR_BOX_CENTER_Z_COL, default_value=0.0)
	lid_l = lidar_table.column(LIDAR_BOX_SIZE_X_COL).to_pylist()
	lid_w = lidar_table.column(LIDAR_BOX_SIZE_Y_COL).to_pylist()
	lid_h = _optional_col_array(lidar_table, LIDAR_BOX_SIZE_Z_COL, default_value=0.0)
	lid_heading = lidar_table.column(LIDAR_BOX_HEADING_COL).to_pylist()
	lid_type = _optional_col_array(lidar_table, LIDAR_BOX_TYPE_COL, default_value=0)
	lid_id = _optional_col_array(lidar_table, LIDAR_BOX_ID_COL, default_value="")
	for idx, ts in enumerate(lid_ts.tolist()):
		box = SimpleNamespace(
			center_x=float(lid_x[idx]),
			center_y=float(lid_y[idx]),
			center_z=float(lid_z[idx]),
			length=float(lid_l[idx]),
			width=float(lid_w[idx]),
			height=float(lid_h[idx]),
			heading=float(lid_heading[idx]),
		)
		label = SimpleNamespace(box=box, type=int(lid_type[idx]), id=str(lid_id[idx]))
		lidar_by_timestamp.setdefault(int(ts), []).append(label)

	timestamps = sorted(set(camera_by_timestamp.keys()) & set(pose_by_timestamp.keys()))
	if max_frames is not None:
		timestamps = timestamps[: int(max_frames)]

	for timestamp in timestamps:
		yield SimpleNamespace(
			timestamp_micros=int(timestamp),
			context=SimpleNamespace(name=segment_id),
			images=camera_by_timestamp[timestamp],
			pose=SimpleNamespace(transform=pose_by_timestamp[timestamp]),
			map_features=map_features,
			laser_labels=lidar_by_timestamp.get(timestamp, []),
		)


def _plot_feature_geometry(ax, map_features):
	"""Draw static map feature geometry onto an axis."""
	for feat in map_features:
		t = feat.WhichOneof("feature_data")

		if t == "lane":
			pts = np.array([[p.x, p.y] for p in feat.lane.polyline], dtype=np.float64)
			if pts.size:
				ax.plot(pts[:, 0], pts[:, 1], "g-", linewidth=1)

		elif t == "road_edge":
			pts = np.array([[p.x, p.y] for p in feat.road_edge.polyline], dtype=np.float64)
			if pts.size:
				ax.plot(pts[:, 0], pts[:, 1], "k-", linewidth=2)

		elif t == "road_line":
			pts = np.array([[p.x, p.y] for p in feat.road_line.polyline], dtype=np.float64)
			if pts.size:
				ax.plot(pts[:, 0], pts[:, 1], color="gray", linewidth=1, linestyle="--")

		elif t == "stop_sign":
			pos = feat.stop_sign.position
			ax.plot(pos.x, pos.y, marker="x", color="magenta", markersize=7, markeredgewidth=2)

		elif t == "crosswalk":
			poly = np.array([[p.x, p.y] for p in feat.crosswalk.polygon], dtype=np.float64)
			if len(poly) > 0:
				poly = np.vstack([poly, poly[0]])
				ax.plot(poly[:, 0], poly[:, 1], color="blue", linewidth=1.5)

		elif t == "driveway":
			poly = np.array([[p.x, p.y] for p in feat.driveway.polygon], dtype=np.float64)
			if len(poly) > 0:
				poly = np.vstack([poly, poly[0]])
				ax.plot(poly[:, 0], poly[:, 1], color="orange", linewidth=1.5)

		elif t == "speed_bump":
			poly = np.array([[p.x, p.y] for p in feat.speed_bump.polygon], dtype=np.float64)
			if len(poly) > 0:
				poly = np.vstack([poly, poly[0]])
				ax.plot(poly[:, 0], poly[:, 1], color="red", linewidth=1.5)

def plot_map_features(frame, save_path: Path, ego_trajectory=None):
	"""Plot lane/map geometry from frame.map_features and save as PNG.

	Structure notes:
	- each entry in frame.map_features has a oneof named "feature_data"
	- we dispatch by feature_data type and plot XY coordinates in world frame
	- ego_trajectory is optional: np.ndarray shape (T, 2) with global XY per frame
	"""
	fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
	_plot_feature_geometry(ax, frame.map_features)

	# Draw full ego trajectory across frames when provided.
	if ego_trajectory is not None and len(ego_trajectory) > 0:
		ego_trajectory = np.asarray(ego_trajectory, dtype=np.float64)
		ax.plot(
			ego_trajectory[:, 0],
			ego_trajectory[:, 1],
			color="cyan",
			linewidth=2.0,
			alpha=0.9,
		)
		ax.plot(ego_trajectory[0, 0], ego_trajectory[0, 1], marker="s", color="cyan", markersize=5)
		ax.plot(ego_trajectory[-1, 0], ego_trajectory[-1, 1], marker="o", color="cyan", markersize=6)

	ax.set_title(f"Waymo map features: {save_path.stem}")
	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")
	ax.set_aspect("equal", adjustable="box")
	ax.grid(alpha=0.25)
	fig.tight_layout()
	save_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(save_path)
	plt.close(fig)

def _lidar_boxes_global_corners(frame):
	"""Convert per-frame lidar boxes from ego frame to global XY box corners."""
	ego = ego_global_pose_from_frame(frame)
	transform = ego["transform"]
	r2 = transform[:2, :2]
	t2 = transform[:2, 3]
	ego_yaw = ego["yaw"]
	boxes = parse_lidar_boxes(frame)[0]
	polygons = []

	for box in boxes:
		cx, cy, _cz, length, width, _height, heading = [float(v) for v in box]
		center_global = r2 @ np.array([cx, cy], dtype=np.float64) + t2
		theta = ego_yaw + heading
		c = np.cos(theta)
		s = np.sin(theta)
		rot = np.array([[c, -s], [s, c]], dtype=np.float64)
		half_l = 0.5 * length
		half_w = 0.5 * width
		local = np.array(
			[
				[half_l, half_w],
				[half_l, -half_w],
				[-half_l, -half_w],
				[-half_l, half_w],
			],
			dtype=np.float64,
		)
		corners = (local @ rot.T) + center_global
		corners = np.vstack([corners, corners[0]])
		polygons.append(corners)
	return polygons

def plot_map_features_w_lidar(frames, save_path: Path, ego_trajectory=None, fps=10):
	"""Render a per-segment video with map, moving ego, and moving lidar boxes.

	- Uses static map features from the first frame.
	- Draws ego trajectory up to current frame.
	- Draws current frame lidar boxes in global frame.
	"""
	if not frames:
		raise ValueError("Expected at least one frame to render video.")

	save_path.parent.mkdir(parents=True, exist_ok=True)
	video_path = save_path.with_name(f"{save_path.stem}_ego_lidar.mp4")
	map_features = frames[0].map_features

	if ego_trajectory is None:
		ego_poses = [ego_global_pose_from_frame(frame) for frame in frames]
		ego_trajectory = np.asarray([[pose["x"], pose["y"]] for pose in ego_poses], dtype=np.float64)
	else:
		ego_trajectory = np.asarray(ego_trajectory, dtype=np.float64)

	if len(ego_trajectory) == 0:
		raise ValueError("ego_trajectory must contain at least one point.")

	writer = None
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")

	# Stable axis limits reduce jitter in the video.
	x_min = float(np.min(ego_trajectory[:, 0])) - 30.0
	x_max = float(np.max(ego_trajectory[:, 0])) + 30.0
	y_min = float(np.min(ego_trajectory[:, 1])) - 30.0
	y_max = float(np.max(ego_trajectory[:, 1])) + 30.0

	for idx, frame in enumerate(frames):
		fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
		_plot_feature_geometry(ax, map_features)

		traj_now = ego_trajectory[: idx + 1]
		ax.plot(traj_now[:, 0], traj_now[:, 1], color="cyan", linewidth=2.0, alpha=0.9)
		ax.plot(traj_now[0, 0], traj_now[0, 1], marker="s", color="cyan", markersize=5)
		ax.plot(traj_now[-1, 0], traj_now[-1, 1], marker="o", color="cyan", markersize=6)

		ego_now = ego_global_pose_from_frame(frame)
		ex = ego_now["x"]
		ey = ego_now["y"]
		yaw = ego_now["yaw"]
		arrow_len = 4.0
		ax.arrow(
			ex,
			ey,
			arrow_len * np.cos(yaw),
			arrow_len * np.sin(yaw),
			width=0.15,
			head_width=0.9,
			head_length=1.2,
			color="cyan",
			length_includes_head=True,
		)

		for poly in _lidar_boxes_global_corners(frame):
			ax.plot(poly[:, 0], poly[:, 1], color="yellow", linewidth=1.0, alpha=0.9)

		ax.set_title(f"Waymo map + ego + lidar: {save_path.stem} | frame {idx}")
		ax.set_xlabel("x (m)")
		ax.set_ylabel("y (m)")
		ax.set_aspect("equal", adjustable="box")
		ax.set_xlim(x_min, x_max)
		ax.set_ylim(y_min, y_max)
		ax.grid(alpha=0.25)
		fig.tight_layout()

		fig.canvas.draw()
		rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
		rgb = rgb.reshape(fig.canvas.get_width_height()[::-1] + (3,))
		bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

		if writer is None:
			h, w = bgr.shape[:2]
			writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (w, h))
		writer.write(bgr)
		plt.close(fig)

	if writer is not None:
		writer.release()
	return video_path

def decode_camera_images(frame, tf, open_dataset):
	"""Return decoded images as {camera_name: np.ndarray[H, W, 3] uint8}.

	Data structure notes:
	- frame.images is a repeated CameraImage proto (one item per camera)
	- image.image stores compressed bytes (usually JPEG)
	- output arrays are RGB uint8 with shape (H, W, 3)
	"""
	camera_images = {}
	for image in frame.images:
		# tf.io.decode_image handles JPEG/PNG and returns HWC uint8 tensors.
		decoded = tf.io.decode_image(image.image, channels=3).numpy()
		camera_name = CAMERA_NAME_TO_LABEL.get(int(image.name), f"UNKNOWN_{int(image.name)}")
		camera_images[camera_name] = decoded
	return camera_images

def parse_ego_pose(frame):
	"""Return ego pose as np.ndarray with shape (4, 4).

	Data structure notes:
	- frame.pose.transform is 16 floats in row-major order
	- matrix maps from vehicle frame -> world frame
	"""
	return np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4)

def ego_global_pose_from_frame(frame):
	"""Get ego global pose from one frame.

	Waymo frame.pose.transform is world-from-vehicle (4x4), so translation is
	the global position and yaw comes from the rotation block.
	"""
	transform = parse_ego_pose(frame)
	x = float(transform[0, 3])
	y = float(transform[1, 3])
	yaw = float(np.arctan2(transform[1, 0], transform[0, 0]))
	return {"x": x, "y": y, "yaw": yaw, "transform": transform}

def parse_lidar_boxes(frame):
	"""Return LiDAR boxes and metadata from frame.laser_labels.

	Data structure notes:
	- frame.laser_labels is a repeated LaserLabel proto (one row per object)
	- each label has box params in the ego-vehicle coordinate frame
	- boxes array shape is (N, 7):
	  [center_x, center_y, center_z, length, width, height, heading]
	"""
	boxes = []
	types = []
	ids = []
	for label in frame.laser_labels:
		box = label.box
		boxes.append(
			[
				box.center_x,
				box.center_y,
				box.center_z,
				box.length,
				box.width,
				box.height,
				box.heading,
			]
		)
		types.append(int(label.type))
		ids.append(label.id)

	if boxes:
		box_array = np.asarray(boxes, dtype=np.float32)
	else:
		box_array = np.zeros((0, 7), dtype=np.float32)
	return box_array, np.asarray(types, dtype=np.int32), ids

def summarize_frame(frame, tf, open_dataset):
	camera_images = decode_camera_images(frame, tf, open_dataset)
	ego_pose = parse_ego_pose(frame)

	# map_features is a repeated proto list; shape is best described by count + type composition.
	map_features = list(frame.map_features)
	map_feature_types = {}
	for feature in map_features:
		feature_type = feature.WhichOneof("feature_data") or "UNKNOWN"
		map_feature_types[feature_type] = map_feature_types.get(feature_type, 0) + 1

	lidar_boxes, lidar_types, lidar_ids = parse_lidar_boxes(frame)

	print("===== Waymo Frame Summary =====")
	print(f"timestamp_micros: {frame.timestamp_micros}")
	print(f"context_name: {frame.context.name}")
	print("\n[Camera]")
	print(f"num_cameras_in_frame: {len(camera_images)}")
	for name, img in sorted(camera_images.items()):
		print(f"  {name:>20}: shape={img.shape}, dtype={img.dtype}")

	print("\n[Ego Pose]")
	print(f"ego_pose shape: {ego_pose.shape}, dtype={ego_pose.dtype}")
	print("ego_pose (4x4):")
	print(ego_pose)

	print("\n[Map Features]")
	print(f"map_features count: {len(map_features)}")
	print(f"map_feature type counts: {map_feature_types}")

	print("\n[LiDAR Boxes]")
	print(f"lidar_boxes shape: {lidar_boxes.shape}, dtype={lidar_boxes.dtype}")
	print(f"lidar_types shape: {lidar_types.shape}, dtype={lidar_types.dtype}")
	print(f"lidar_ids count: {len(lidar_ids)}")

	return {
		"camera_images": camera_images,
		"ego_pose": ego_pose,
		"map_features": map_features,
		"map_feature_type_counts": map_feature_types,
		"lidar_boxes": lidar_boxes,
		"lidar_types": lidar_types,
		"lidar_ids": lidar_ids,
		"frame": frame,
	}

def main():
	tf, open_dataset = import_waymo_runtime()
	segment_paths = list_segment_files(Path(PATH))
	print(f"Found {len(segment_paths)} segment files from parquet components")

	for seg_idx, segment_path in enumerate(segment_paths):
		frames = list(iter_segment_frames(segment_path, tf, open_dataset, max_frames=MAX_FRAMES))
		if not frames:
			print(f"[{seg_idx + 1}/{len(segment_paths)}] skipped empty segment: {segment_path}")
			continue

		frame = frames[0]
		segment_id = segment_stem(segment_path)
		map_path = OUTPUT_DIR / f"{segment_id}_map.png"
		# get ego global pose per frame and plot map features in global frame
		ego_poses = [ego_global_pose_from_frame(f) for f in frames]
		ego_trajectory = np.asarray([[pose["x"], pose["y"]] for pose in ego_poses], dtype=np.float64)
		plot_map_features(frame, map_path, ego_trajectory=ego_trajectory)
		video_path = plot_map_features_w_lidar(frames, map_path, ego_trajectory=ego_trajectory, fps=10)
		print(f"[{seg_idx + 1}/{len(segment_paths)}] saved map plot: {map_path}")
		print(f"[{seg_idx + 1}/{len(segment_paths)}] saved video: {video_path}")

		# Keep detailed inspection for the first segment to avoid stopping repeatedly.
		if seg_idx == 0:
			parsed = summarize_frame(frame, tf, open_dataset)
			if ENTER_PDB:
				import pdb
				pdb.set_trace()

if __name__ == "__main__":
	main()