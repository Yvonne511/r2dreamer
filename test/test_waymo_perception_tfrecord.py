
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

import cv2
import matplotlib.pyplot as plt
import numpy as np


PATH = "/scratch/yw4142/datasets/ad/waymo_open_dataset_v_1_4_3/training"
OUTPUT_DIR = Path("/home/yw4142/ad/r2dreamer/test/waymo_vis_outputs")
MAX_FRAMES = None  # None means use all frames in each segment.
ENTER_PDB = True

CAMERA_NAME_TO_LABEL = {
	1: "FRONT",
	2: "FRONT_LEFT",
	3: "FRONT_RIGHT",
	4: "SIDE_LEFT",
	5: "SIDE_RIGHT",
}

def import_waymo_runtime():
	"""Import tensorflow + Waymo protobuf runtime with eager mode enabled."""
	import tensorflow.compat.v1 as tf
	from waymo_open_dataset import dataset_pb2 as open_dataset

	if hasattr(tf, "executing_eagerly") and not tf.executing_eagerly():
		tf.enable_eager_execution()
	return tf, open_dataset

def find_segment_file(root: Path) -> Path:
	"""Find one TFRecord segment file under PATH."""
	if root.is_file() and ".tfrecord" in root.name:
		return root

	candidates = sorted(p for p in root.rglob("*") if p.is_file() and ".tfrecord" in p.name)
	if not candidates:
		raise FileNotFoundError(f"No TFRecord segment file found under {root}")
	return candidates[0]

def list_segment_files(root: Path) -> list[Path]:
	"""List all TFRecord segment files under PATH."""
	if root.is_file() and ".tfrecord" in root.name:
		return [root]

	segment_paths = sorted(p for p in root.rglob("*") if p.is_file() and ".tfrecord" in p.name)
	if not segment_paths:
		raise FileNotFoundError(f"No TFRecord segment file found under {root}")
	return segment_paths

def segment_stem(segment_path: Path) -> str:
	"""Normalize TFRecord name to a compact segment id for filenames."""
	name = segment_path.name
	for suffix in (".tfrecord.gz", ".tfrecord.gzip", ".tfrecords", ".tfrecord"):
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
	"""Yield all frames from one segment TFRecord."""
	dataset = tf.data.TFRecordDataset(
		[str(segment_path)],
		compression_type="",
		buffer_size=8 << 20,
		num_parallel_reads=1,
	)
	for frame_idx, raw in enumerate(dataset):
		frame = open_dataset.Frame()
		frame.ParseFromString(raw.numpy())
		yield frame
		if max_frames is not None and frame_idx + 1 >= int(max_frames):
			break


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
	print(f"Found {len(segment_paths)} segment files under: {PATH}")

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