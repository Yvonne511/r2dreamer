import argparse
import json
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

tf.enable_eager_execution()


DEFAULT_TRAIN_DIR = Path("/scratch/yw4142/datasets/ad/waymo_open_dataset_v_1_4_3/training")
DEFAULT_OUTPUT_DIR = Path("/home/yw4142/ad/r2dreamer/test/waymo_vis_outputs")
CAMERA_ORDER = [
    open_dataset.CameraName.SIDE_LEFT,
    open_dataset.CameraName.FRONT_LEFT,
    open_dataset.CameraName.FRONT,
    open_dataset.CameraName.FRONT_RIGHT,
    open_dataset.CameraName.SIDE_RIGHT,
]
CAMERA_NAME_TO_LABEL = {
    open_dataset.CameraName.FRONT: "front",
    open_dataset.CameraName.FRONT_LEFT: "front_left",
    open_dataset.CameraName.FRONT_RIGHT: "front_right",
    open_dataset.CameraName.SIDE_LEFT: "side_left",
    open_dataset.CameraName.SIDE_RIGHT: "side_right",
}


def list_segment_files(root: Path) -> list[Path]:
    files = sorted(path for path in root.glob("*.tfrecord") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No TFRecord files found in {root}")
    return files


def iter_segment_frames(segment_path: Path):
    dataset = tf.data.TFRecordDataset([str(segment_path)], compression_type="")
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(data.numpy())
        yield frame


def decode_camera_images(frame: open_dataset.Frame) -> list[tuple[str, np.ndarray]]:
    decoded = {}
    for image in frame.images:
        decoded[int(image.name)] = tf.image.decode_jpeg(image.image).numpy()

    ordered = []
    for camera_name in CAMERA_ORDER:
        if camera_name in decoded:
            ordered.append((CAMERA_NAME_TO_LABEL[camera_name], decoded[camera_name]))
    return ordered


def stitch_camera_views(camera_images: list[tuple[str, np.ndarray]]) -> np.ndarray:
    annotated = []
    target_height = max(image.shape[0] for _, image in camera_images)
    for label, image in camera_images:
        if image.shape[0] != target_height:
            width = int(round(image.shape[1] * target_height / image.shape[0]))
            image = cv2.resize(image, (width, target_height), interpolation=cv2.INTER_LINEAR)
        image = image.copy()
        cv2.putText(
            image,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        annotated.append(image)
    return np.concatenate(annotated, axis=1)


def resize_canvas(canvas: np.ndarray, max_width: int) -> np.ndarray:
    if canvas.shape[1] <= max_width:
        return canvas
    scale = max_width / float(canvas.shape[1])
    resized_height = max(2, int(round(canvas.shape[0] * scale)))
    resized_width = max(2, int(round(canvas.shape[1] * scale)))
    if resized_width % 2 == 1:
        resized_width -= 1
    if resized_height % 2 == 1:
        resized_height -= 1
    return cv2.resize(canvas, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def pose_matrix(frame: open_dataset.Frame) -> np.ndarray:
    return np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4)


def yaw_from_pose(transform: np.ndarray) -> float:
    return float(np.arctan2(transform[1, 0], transform[0, 0]))


def wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def point_xy(proto_point) -> np.ndarray:
    return np.asarray([proto_point.x, proto_point.y], dtype=np.float64)


def polyline_xy(polyline) -> np.ndarray:
    return np.asarray([[point.x, point.y] for point in polyline], dtype=np.float64)


def closest_point_on_polyline(point_xy_world: np.ndarray, polyline: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, int, float]:
    if len(polyline) == 0:
        raise ValueError("Polyline must contain at least one point")
    if len(polyline) == 1:
        return float(np.linalg.norm(point_xy_world - polyline[0])), polyline[0], np.array([1.0, 0.0]), 0, 0.0

    best_distance = float("inf")
    best_point = polyline[0]
    best_tangent = np.array([1.0, 0.0], dtype=np.float64)
    best_segment = 0
    best_u = 0.0

    for idx in range(len(polyline) - 1):
        start = polyline[idx]
        end = polyline[idx + 1]
        segment = end - start
        seg_len_sq = float(np.dot(segment, segment))
        if seg_len_sq <= 1e-9:
            continue
        u = float(np.clip(np.dot(point_xy_world - start, segment) / seg_len_sq, 0.0, 1.0))
        candidate = start + u * segment
        distance = float(np.linalg.norm(point_xy_world - candidate))
        if distance < best_distance:
            best_distance = distance
            best_point = candidate
            best_tangent = segment / np.sqrt(seg_len_sq)
            best_segment = idx
            best_u = u

    return best_distance, best_point, best_tangent, best_segment, best_u


def lane_progress(polyline: np.ndarray, segment_idx: int, u: float) -> float:
    if len(polyline) <= 1:
        return 0.0
    progress = 0.0
    for idx in range(segment_idx):
        progress += float(np.linalg.norm(polyline[idx + 1] - polyline[idx]))
    progress += u * float(np.linalg.norm(polyline[segment_idx + 1] - polyline[segment_idx]))
    return progress


def box_corners_2d(center_xy: np.ndarray, length: float, width: float, heading: float) -> np.ndarray:
    half_l = length / 2.0
    half_w = width / 2.0
    local = np.asarray(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float64,
    )
    c = np.cos(heading)
    s = np.sin(heading)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return local @ rot.T + center_xy


def polygons_intersect(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    def axes(poly: np.ndarray) -> list[np.ndarray]:
        out = []
        for idx in range(len(poly)):
            edge = poly[(idx + 1) % len(poly)] - poly[idx]
            norm = np.linalg.norm(edge)
            if norm <= 1e-9:
                continue
            axis = np.asarray([-edge[1], edge[0]], dtype=np.float64) / norm
            out.append(axis)
        return out

    for axis in axes(poly_a) + axes(poly_b):
        a_proj = poly_a @ axis
        b_proj = poly_b @ axis
        if a_proj.max() < b_proj.min() or b_proj.max() < a_proj.min():
            return False
    return True


def build_map_index(map_features) -> dict:
    lanes = {}
    road_edges = []
    stop_signs = []

    for feature in map_features:
        feature_type = feature.WhichOneof("feature_data")
        if feature_type == "lane":
            lane = feature.lane
            lanes[int(feature.id)] = {
                "id": int(feature.id),
                "polyline": polyline_xy(lane.polyline),
                "lane": lane,
            }
        elif feature_type == "road_edge":
            road_edges.append(
                {
                    "id": int(feature.id),
                    "polyline": polyline_xy(feature.road_edge.polyline),
                }
            )
        elif feature_type == "stop_sign":
            stop_signs.append(
                {
                    "id": int(feature.id),
                    "position": point_xy(feature.stop_sign.position),
                    "lane_ids": [int(lane_id) for lane_id in feature.stop_sign.lane],
                }
            )

    return {
        "lanes": lanes,
        "road_edges": road_edges,
        "stop_signs": stop_signs,
    }


def ego_pose_xy_heading(frame: open_dataset.Frame) -> tuple[np.ndarray, float]:
    transform = pose_matrix(frame)
    return transform[:2, 3].copy(), yaw_from_pose(transform)


def nearest_lane_info(frame: open_dataset.Frame, map_index: dict) -> Optional[dict]:
    if not map_index["lanes"]:
        return None

    ego_xy, ego_heading = ego_pose_xy_heading(frame)
    best = None
    for lane_id, lane_info in map_index["lanes"].items():
        polyline = lane_info["polyline"]
        if len(polyline) == 0:
            continue
        distance, closest_xy, tangent_xy, segment_idx, u = closest_point_on_polyline(ego_xy, polyline)
        heading = float(np.arctan2(tangent_xy[1], tangent_xy[0]))
        alignment = wrap_angle(ego_heading - heading)
        candidate = {
            "lane_id": lane_id,
            "distance_to_center": distance,
            "closest_xy": closest_xy,
            "tangent_xy": tangent_xy,
            "alignment_angle": alignment,
            "progress": lane_progress(polyline, segment_idx, u),
        }
        if best is None or candidate["distance_to_center"] < best["distance_to_center"]:
            best = candidate
    return best


def nearest_road_edge_distance(frame: open_dataset.Frame, map_index: dict) -> Optional[float]:
    ego_xy, _ = ego_pose_xy_heading(frame)
    best_distance = None
    for edge in map_index["road_edges"]:
        polyline = edge["polyline"]
        if len(polyline) == 0:
            continue
        distance, _, _, _, _ = closest_point_on_polyline(ego_xy, polyline)
        if best_distance is None or distance < best_distance:
            best_distance = distance
    return best_distance


def detect_collision(frame: open_dataset.Frame, ego_length: float = 4.8, ego_width: float = 2.0) -> bool:
    ego_poly = box_corners_2d(np.zeros(2, dtype=np.float64), ego_length, ego_width, 0.0)
    for label in frame.laser_labels:
        if not label.box.ByteSize():
            continue
        other_poly = box_corners_2d(
            np.asarray([label.box.center_x, label.box.center_y], dtype=np.float64),
            float(label.box.length),
            float(label.box.width),
            float(label.box.heading),
        )
        if polygons_intersect(ego_poly, other_poly):
            return True
    return False


def detect_cross_stop_line(frame: open_dataset.Frame, previous_frame: Optional[open_dataset.Frame], lane_info: Optional[dict], map_index: dict) -> bool:
    if previous_frame is None or lane_info is None:
        return False

    previous_xy, _ = ego_pose_xy_heading(previous_frame)
    current_xy, _ = ego_pose_xy_heading(frame)
    tangent_xy = lane_info["tangent_xy"]
    normal_xy = np.asarray([-tangent_xy[1], tangent_xy[0]], dtype=np.float64)

    for stop_sign in map_index["stop_signs"]:
        if lane_info["lane_id"] not in stop_sign["lane_ids"]:
            continue
        lateral_distance = abs(float(np.dot(current_xy - stop_sign["position"], normal_xy)))
        if lateral_distance > 6.0:
            continue
        prev_longitudinal = float(np.dot(previous_xy - stop_sign["position"], tangent_xy))
        curr_longitudinal = float(np.dot(current_xy - stop_sign["position"], tangent_xy))
        if prev_longitudinal > 0.0 and curr_longitudinal <= 0.0:
            return True
    return False


def calculate_map_metrics(frame: open_dataset.Frame, map_index: dict, previous_frame: Optional[open_dataset.Frame] = None) -> dict:
    lane_info = nearest_lane_info(frame, map_index)
    road_edge_distance = nearest_road_edge_distance(frame, map_index)

    lane_alignment = None if lane_info is None else float(lane_info["alignment_angle"])
    lane_centering = None if lane_info is None else float(lane_info["distance_to_center"])
    leaves_road_boundary = bool(lane_info is None)
    if lane_info is not None:
        leaves_road_boundary = lane_info["distance_to_center"] > 2.5
        if road_edge_distance is not None:
            leaves_road_boundary = leaves_road_boundary or road_edge_distance < 1.0

    return {
        "collision": detect_collision(frame),
        "leaves_road_boundary": leaves_road_boundary,
        "lane_alignment_rad": lane_alignment,
        "lane_center_offset_m": lane_centering,
        "cross_stop_line": detect_cross_stop_line(frame, previous_frame, lane_info, map_index),
        "nearest_lane_id": None if lane_info is None else int(lane_info["lane_id"]),
        "nearest_road_edge_distance_m": None if road_edge_distance is None else float(road_edge_distance),
    }


def derive_action(previous_frame: open_dataset.Frame | None, frame: open_dataset.Frame) -> dict:
    if previous_frame is None:
        return {
            "dt_sec": 0.0,
            "dx_vehicle": 0.0,
            "dy_vehicle": 0.0,
            "dz_world": 0.0,
            "yaw_delta": 0.0,
            "speed_mps": 0.0,
            "yaw_rate_rps": 0.0,
        }

    current_pose = pose_matrix(frame)
    previous_pose = pose_matrix(previous_frame)
    current_translation = current_pose[:3, 3]
    previous_translation = previous_pose[:3, 3]
    delta_world = current_translation - previous_translation
    previous_rotation = previous_pose[:3, :3]
    delta_vehicle = previous_rotation.T @ delta_world

    current_yaw = yaw_from_pose(current_pose)
    previous_yaw = yaw_from_pose(previous_pose)
    yaw_delta = wrap_angle(current_yaw - previous_yaw)

    dt_sec = max((frame.timestamp_micros - previous_frame.timestamp_micros) / 1e6, 1e-6)
    planar_speed = float(np.linalg.norm(delta_vehicle[:2]) / dt_sec)

    return {
        "dt_sec": float(dt_sec),
        "dx_vehicle": float(delta_vehicle[0]),
        "dy_vehicle": float(delta_vehicle[1]),
        "dz_world": float(delta_world[2]),
        "yaw_delta": float(yaw_delta),
        "speed_mps": planar_speed,
        "yaw_rate_rps": float(yaw_delta / dt_sec),
    }


def map_feature_summary(frame: open_dataset.Frame) -> dict:
    counts = Counter()
    for feature in frame.map_features:
        feature_type = feature.WhichOneof("feature_data") or "unknown"
        counts[feature_type] += 1
    return dict(sorted(counts.items()))


def print_context_and_maps(frame: open_dataset.Frame) -> None:
    stats = frame.context.stats
    print("Context:", frame.context.name)
    print("Timestamp:", frame.timestamp_micros)
    print("Location:", stats.location)
    print("Time of day:", stats.time_of_day)
    print("Weather:", stats.weather)
    print("Camera calibrations:", len(frame.context.camera_calibrations))
    print("Laser calibrations:", len(frame.context.laser_calibrations))
    print("Map feature counts:", map_feature_summary(frame))


def save_video_and_actions(segment_path: Path, output_dir: Path, max_frames: int | None = None, max_video_width: int = 1920) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = iter_segment_frames(segment_path)
    first_frame = next(frames, None)
    if first_frame is None:
        raise ValueError(f"No frames found in {segment_path}")
    map_index = build_map_index(first_frame.map_features)

    print_context_and_maps(first_frame)

    first_images = decode_camera_images(first_frame)
    if not first_images:
        raise ValueError("No camera images found in first frame")

    first_canvas = resize_canvas(stitch_camera_views(first_images), max_video_width)
    video_path = output_dir / f"{segment_path.stem}.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (first_canvas.shape[1], first_canvas.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")

    action_path = output_dir / f"{segment_path.stem}_actions.jsonl"
    map_path = output_dir / f"{segment_path.stem}_map_summary.json"

    map_payload = {
        "context_name": first_frame.context.name,
        "location": first_frame.context.stats.location,
        "time_of_day": first_frame.context.stats.time_of_day,
        "weather": first_frame.context.stats.weather,
        "map_feature_counts": map_feature_summary(first_frame),
    }
    map_path.write_text(json.dumps(map_payload, indent=2), encoding="utf-8")

    processed = 0
    previous_frame = None
    with action_path.open("w", encoding="utf-8") as f:
        for frame in chain([first_frame], frames):
            camera_images = decode_camera_images(frame)
            if not camera_images:
                continue
            canvas = resize_canvas(stitch_camera_views(camera_images), max_video_width)
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

            action = derive_action(previous_frame, frame)
            map_metrics = calculate_map_metrics(frame, map_index=map_index, previous_frame=previous_frame)
            action_record = {
                "frame_idx": processed,
                "timestamp_micros": int(frame.timestamp_micros),
                "action": action,
                "map_metrics": map_metrics,
            }
            f.write(json.dumps(action_record) + "\n")

            if processed < 5:
                print(f"Action[{processed}]:", action_record)

            previous_frame = frame
            processed += 1
            if max_frames is not None and processed >= max_frames:
                break

    writer.release()
    print(f"Saved video to {video_path}")
    print(f"Saved action log to {action_path}")
    print(f"Saved map summary to {map_path}")
    print(f"Processed {processed} frames")


def parse_args():
    parser = argparse.ArgumentParser(description="Dump one Waymo perception trajectory to video plus simple action/map summaries.")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--segment", type=Path, default=None, help="Optional explicit TFRecord segment path.")
    parser.add_argument("--segment-index", type=int, default=0, help="Index into sorted training TFRecord files when --segment is unset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-video-width", type=int, default=1920)
    return parser.parse_args()


def main():
    args = parse_args()
    segment_path = args.segment
    if segment_path is None:
        segment_path = list_segment_files(args.train_dir)[args.segment_index]

    print(f"Using segment: {segment_path}")
    save_video_and_actions(
        segment_path=segment_path,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        max_video_width=args.max_video_width,
    )


if __name__ == "__main__":
    main()
