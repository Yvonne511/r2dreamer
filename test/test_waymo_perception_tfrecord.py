
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

import matplotlib.pyplot as plt
import numpy as np


PATH = "/scratch/yw4142/datasets/ad/waymo_open_dataset_v_1_4_3/training"
OUTPUT_DIR = Path("/home/yw4142/ad/r2dreamer/test/waymo_vis_outputs")
MAX_FRAMES = 1
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

def plot_map_features(frame, save_path: Path):
	"""Plot lane/map geometry from frame.map_features and save as PNG.

	Structure notes:
	- each entry in frame.map_features has a oneof named "feature_data"
	- we dispatch by feature_data type and plot XY coordinates in world frame
	"""
	fig, ax = plt.subplots(figsize=(10, 10), dpi=120)

	for feat in frame.map_features:
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

	ax.set_title(f"Waymo map features: {save_path.stem}")
	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")
	ax.set_aspect("equal", adjustable="box")
	ax.grid(alpha=0.25)
	fig.tight_layout()
	save_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(save_path)
	plt.close(fig)

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
		frame = load_first_frame(segment_path, tf, open_dataset)
		segment_id = segment_stem(segment_path)
		map_path = OUTPUT_DIR / f"{segment_id}_map.png"
		plot_map_features(frame, map_path)
		print(f"[{seg_idx + 1}/{len(segment_paths)}] saved map plot: {map_path}")

		# Keep detailed inspection for the first segment to avoid stopping repeatedly.
		if seg_idx == 0:
			parsed = summarize_frame(frame, tf, open_dataset)
			if ENTER_PDB:
				import pdb
				pdb.set_trace()

if __name__ == "__main__":
	main()