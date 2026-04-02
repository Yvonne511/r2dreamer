from collections import OrderedDict
from itertools import chain
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


FRONT3_CAMERA_ORDER = (2, 1, 3)
ACTION_KEYS = ("acceleration_mps2", "steer_rad")


class WaymoPerceptionDataset(Dataset):
    def __init__(self, dataset_config):
        self.segment_paths = self.list_waymo_segments(Path(dataset_config.path))
        n_rollout = dataset_config.n_rollout
        if n_rollout is not None:
            self.segment_paths = self.segment_paths[:n_rollout]
        self.max_frames = None  # dataset_config.max_frames_per_episode
        self.image_size = _normalize_image_size(dataset_config.image_size)
        self.wheelbase_m = float(dataset_config.wheelbase_m)  # distance between the front and rear axles, in meters
        self.max_steer_rad = float(dataset_config.max_steer_rad)  # maximum steering angle of the front wheels, in radians

    def __len__(self):
        return len(self.segment_paths)  # TODOL: support slices later

    def __getitem__(self, index):
        '''
        Return episode dict with keys:
        - image: (T, H, W, C) uint8 RGB front camera images
        - action: (T, A) float32 continuous actions with "acceleration_mps2" and "steer_rad" components
        - reward: (T, 5) float32 rewards
        - is_first: (T, 1)
        - is_last: (T, 1)
        - is_terminal: (T, 1)
        - reward_lane_alignment: (T, 1) float32 
        - reward_lane_center_offset: (T, 1) float32
        - reward_collision: (T, 1) -1, 0
        - reward_road_boundary: (T, 1) -1, 0
        - reward_cross_stop_line: (T, 1) -1, 0
        '''
        segment_path = self.segment_paths[index]
        episode = self.build_episode(
            self.iter_segment_frames(segment_path, max_frames=self.max_frames),
        )
        return episode

    # Waymo perception data loading utils
    def list_waymo_segments(self, root):
        root = Path(root)
        patterns = ("*.tfrecord",)
        files = []
        for pattern in patterns:
            files.extend(path for path in root.glob(pattern) if path.is_file())
        deduped = sorted(set(files))
        if not deduped:
            raise FileNotFoundError(f"No TFRecord files found in {root}")
        return deduped

    def _import_waymo_runtime(self):
        # https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb
        try:
            import tensorflow.compat.v1 as tf
            from waymo_open_dataset import dataset_pb2 as open_dataset
        except ImportError as exc:
            raise ImportError("Waymo perception loading requires tensorflow and waymo_open_dataset.") from exc
        if hasattr(tf, "executing_eagerly") and not tf.executing_eagerly():
            tf.enable_eager_execution()
        return tf, open_dataset

    def iter_segment_frames(self, segment_path, max_frames=None):
        tf, open_dataset = self._import_waymo_runtime()
        dataset = tf.data.TFRecordDataset([str(segment_path)], compression_type="")
        for frame_idx, data in enumerate(dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(data.numpy())
            yield frame
            if max_frames is not None and frame_idx + 1 >= max_frames:
                break

    def build_episode(self, frames):
        frames = iter(frames)
        first_frame = next(frames, None)
        if first_frame is None:
            raise ValueError("Expected at least one frame to build an episode.")

        map_index = self.build_map_index(getattr(first_frame, "map_features", []))
        images = []
        actions = []
        reward_rows = []
        reward_components = OrderedDict()
        previous_frame = None
        previous_speed_mps = 0.0

        for frame in chain((first_frame,), frames):
            images.append(self.stack_front3_images(frame, image_size=self.image_size))
            action_dict, previous_speed_mps = self.derive_action(
                previous_frame,
                frame,
                previous_speed_mps=previous_speed_mps,
                wheelbase_m=self.wheelbase_m,
                max_steer_rad=self.max_steer_rad,
            )
            actions.append(np.asarray([action_dict[key] for key in ACTION_KEYS], dtype=np.float32))
            reward_dict = self.compute_reward_components(frame, previous_frame, map_index, action_dict)
            if not reward_components:
                reward_components = OrderedDict((key, []) for key in reward_dict)
            for key in reward_components:
                reward_components[key].append([float(reward_dict[key])])
            # Keep reward categories separate; the info dict stores the matching key order.
            reward_rows.append([float(reward_dict[key]) for key in reward_components])
            previous_frame = frame

        time_dim = len(images)
        episode = {
            "image": np.stack(images, axis=0).astype(np.uint8),
            "action": np.stack(actions, axis=0).astype(np.float32),
            "reward": np.asarray(reward_rows, dtype=np.float32),
            "is_first": _flag(time_dim, first=True),
            "is_last": _flag(time_dim, last=True),
            "is_terminal": _flag(time_dim, last=True),
        }
        for key, values in reward_components.items():
            episode[key] = np.asarray(values, dtype=np.float32)
        return episode

    def stack_front3_images(self, frame, image_size=None):
        # 1280, 5760, 3 (stacked front 3 cameras))
        decoded = {}
        for image in getattr(frame, "images", []):
            decoded[int(image.name)] = _decode_image(getattr(image, "image", image))
        image_list = []
        for camera_name in FRONT3_CAMERA_ORDER:
            if camera_name not in decoded:
                raise ValueError(f"Missing front camera {camera_name} in frame")
            image = decoded[camera_name]
            if image_size is not None and image.shape[:2] != tuple(image_size):
                image = cv2.resize(image, tuple(image_size[::-1]), interpolation=cv2.INTER_AREA)
            image_list.append(image)
        return np.concatenate(image_list, axis=1)

    def derive_action(self, previous_frame, frame, previous_speed_mps=0.0, wheelbase_m=2.8, max_steer_rad=0.7):
        if previous_frame is None:
            return {
                "acceleration_mps2": 0.0,
                "steer_rad": 0.0,
                "forward_progress_m": 0.0,
                "lateral_progress_m": 0.0,
                "yaw_delta_rad": 0.0,
                "speed_mps": 0.0,
                "yaw_rate_rps": 0.0,
            }, 0.0

        current_pose = self.pose_matrix(frame)
        previous_pose = self.pose_matrix(previous_frame)
        current_translation = current_pose[:3, 3]
        previous_translation = previous_pose[:3, 3]
        delta_world = current_translation - previous_translation
        delta_vehicle = previous_pose[:3, :3].T @ delta_world
        yaw_delta = self.wrap_angle(self.yaw_from_pose(current_pose) - self.yaw_from_pose(previous_pose))
        dt_sec = max((float(frame.timestamp_micros) - float(previous_frame.timestamp_micros)) / 1e6, 1e-6)
        speed_mps = float(delta_vehicle[0] / dt_sec)
        acceleration_mps2 = float((speed_mps - previous_speed_mps) / dt_sec)
        yaw_rate_rps = float(yaw_delta / dt_sec)
        steer_rad = 0.0 if abs(speed_mps) < 1e-4 else float(np.arctan(wheelbase_m * yaw_rate_rps / speed_mps))
        steer_rad = float(np.clip(steer_rad, -max_steer_rad, max_steer_rad))
        return {
            "acceleration_mps2": acceleration_mps2,
            "steer_rad": steer_rad,
            "forward_progress_m": float(delta_vehicle[0]),
            "lateral_progress_m": float(delta_vehicle[1]),
            "yaw_delta_rad": float(yaw_delta),
            "speed_mps": speed_mps,
            "yaw_rate_rps": yaw_rate_rps,
        }, speed_mps

    def compute_reward_components(self, frame, previous_frame, map_index, action_dict):
        map_metrics = self.calculate_map_metrics(frame, map_index=map_index, previous_frame=previous_frame)
        lane_alignment = 0.0 if map_metrics["lane_alignment_rad"] is None else -abs(float(map_metrics["lane_alignment_rad"]))
        lane_center = 0.0 if map_metrics["lane_center_offset_m"] is None else -abs(float(map_metrics["lane_center_offset_m"]))
        return {
            # "reward_forward_progress": float(action_dict["forward_progress_m"]),
            "reward_lane_alignment": lane_alignment,
            "reward_lane_center_offset": lane_center,
            "reward_collision": -1.0 if map_metrics["collision"] else 0.0,
            "reward_road_boundary": -1.0 if map_metrics["leaves_road_boundary"] else 0.0,
            "reward_cross_stop_line": -1.0 if map_metrics["cross_stop_line"] else 0.0,
        }

    def pose_matrix(self, frame):
        return np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4)

    def yaw_from_pose(self, transform):
        return float(np.arctan2(transform[1, 0], transform[0, 0]))

    def wrap_angle(self, angle):
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def point_xy(self, proto_point):
        return np.asarray([proto_point.x, proto_point.y], dtype=np.float64)

    def polyline_xy(self, polyline):
        return np.asarray([[point.x, point.y] for point in polyline], dtype=np.float64)

    def closest_point_on_polyline(self, point_xy_world, polyline):
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

    def lane_progress(self, polyline, segment_idx, u):
        if len(polyline) <= 1:
            return 0.0
        progress = 0.0
        for idx in range(segment_idx):
            progress += float(np.linalg.norm(polyline[idx + 1] - polyline[idx]))
        progress += u * float(np.linalg.norm(polyline[segment_idx + 1] - polyline[segment_idx]))
        return progress

    def box_corners_2d(self, center_xy, length, width, heading):
        half_l = length / 2.0
        half_w = width / 2.0
        local = np.asarray([[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]], dtype=np.float64)
        c = np.cos(heading)
        s = np.sin(heading)
        rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
        return local @ rot.T + center_xy

    def polygons_intersect(self, poly_a, poly_b):
        def axes(poly):
            out = []
            for idx in range(len(poly)):
                edge = poly[(idx + 1) % len(poly)] - poly[idx]
                norm = np.linalg.norm(edge)
                if norm <= 1e-9:
                    continue
                out.append(np.asarray([-edge[1], edge[0]], dtype=np.float64) / norm)
            return out

        for axis in axes(poly_a) + axes(poly_b):
            a_proj = poly_a @ axis
            b_proj = poly_b @ axis
            if a_proj.max() < b_proj.min() or b_proj.max() < a_proj.min():
                return False
        return True

    def build_map_index(self, map_features):
        lanes = {}
        road_edges = []
        stop_signs = []
        for feature in map_features:
            feature_type = feature.WhichOneof("feature_data")
            if feature_type == "lane":
                lanes[int(feature.id)] = {
                    "id": int(feature.id),
                    "polyline": self.polyline_xy(feature.lane.polyline),
                    "lane": feature.lane,
                }
            elif feature_type == "road_edge":
                road_edges.append({"id": int(feature.id), "polyline": self.polyline_xy(feature.road_edge.polyline)})
            elif feature_type == "stop_sign":
                stop_signs.append(
                    {
                        "id": int(feature.id),
                        "position": self.point_xy(feature.stop_sign.position),
                        "lane_ids": [int(lane_id) for lane_id in feature.stop_sign.lane],
                    }
                )
        return {"lanes": lanes, "road_edges": road_edges, "stop_signs": stop_signs}

    def ego_pose_xy_heading(self, frame):
        transform = self.pose_matrix(frame)
        return transform[:2, 3].copy(), self.yaw_from_pose(transform)

    def nearest_lane_info(self, frame, map_index):
        if not map_index["lanes"]:
            return None
        ego_xy, ego_heading = self.ego_pose_xy_heading(frame)
        best = None
        for lane_id, lane_info in map_index["lanes"].items():
            polyline = lane_info["polyline"]
            if len(polyline) == 0:
                continue
            distance, closest_xy, tangent_xy, segment_idx, u = self.closest_point_on_polyline(ego_xy, polyline)
            heading = float(np.arctan2(tangent_xy[1], tangent_xy[0]))
            candidate = {
                "lane_id": lane_id,
                "distance_to_center": distance,
                "closest_xy": closest_xy,
                "tangent_xy": tangent_xy,
                "alignment_angle": self.wrap_angle(ego_heading - heading),
                "progress": self.lane_progress(polyline, segment_idx, u),
            }
            if best is None or candidate["distance_to_center"] < best["distance_to_center"]:
                best = candidate
        return best

    def nearest_road_edge_distance(self, frame, map_index):
        ego_xy, _ = self.ego_pose_xy_heading(frame)
        best_distance = None
        for edge in map_index["road_edges"]:
            polyline = edge["polyline"]
            if len(polyline) == 0:
                continue
            distance, _, _, _, _ = self.closest_point_on_polyline(ego_xy, polyline)
            if best_distance is None or distance < best_distance:
                best_distance = distance
        return best_distance

    def detect_collision(self, frame, ego_length=4.8, ego_width=2.0):
        # Laser labels are already expressed in the ego vehicle frame.
        ego_poly = self.box_corners_2d(np.zeros(2, dtype=np.float64), ego_length, ego_width, 0.0)
        for label in getattr(frame, "laser_labels", []):
            if hasattr(label.box, "ByteSize") and label.box.ByteSize() == 0:
                continue
            other_poly = self.box_corners_2d(
                np.asarray([label.box.center_x, label.box.center_y], dtype=np.float64),
                float(label.box.length),
                float(label.box.width),
                float(label.box.heading),
            )
            if self.polygons_intersect(ego_poly, other_poly):
                return True
        return False

    def detect_cross_stop_line(self, frame, previous_frame, lane_info, map_index):
        if previous_frame is None or lane_info is None:
            return False
        previous_xy, _ = self.ego_pose_xy_heading(previous_frame)
        current_xy, _ = self.ego_pose_xy_heading(frame)
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

    def calculate_map_metrics(self, frame, map_index, previous_frame=None):
        lane_info = self.nearest_lane_info(frame, map_index)
        road_edge_distance = self.nearest_road_edge_distance(frame, map_index)
        lane_alignment = None if lane_info is None else float(lane_info["alignment_angle"])
        lane_centering = None if lane_info is None else float(lane_info["distance_to_center"])
        leaves_road_boundary = bool(lane_info is None)
        if lane_info is not None:
            leaves_road_boundary = lane_info["distance_to_center"] > 2.5
            if road_edge_distance is not None:
                leaves_road_boundary = leaves_road_boundary or road_edge_distance < 1.0
        return {
            "collision": self.detect_collision(frame),
            "leaves_road_boundary": leaves_road_boundary,
            "lane_alignment_rad": lane_alignment,
            "lane_center_offset_m": lane_centering,
            "cross_stop_line": self.detect_cross_stop_line(frame, previous_frame, lane_info, map_index),
            # "nearest_lane_id": None if lane_info is None else int(lane_info["lane_id"]),
            "nearest_road_edge_distance_m": None if road_edge_distance is None else float(road_edge_distance),
        }

    def build_dataset_info(self, sample_infos):
        return {
            "num_episodes": len(sample_infos),
            "reward_keys": sample_infos[0]["reward_keys"] if sample_infos else [],
            "episodes": sample_infos,
        }


def make_waymo_perception_dataloader(dataset_config, dataset=None):
    dataset = dataset or WaymoPerceptionDataset(dataset_config)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        multiprocessing_context="spawn",
    )

def _flag(time_dim, first=False, last=False):
    value = np.zeros((time_dim, 1), dtype=bool)
    if first:
        value[0, 0] = True
    if last:
        value[-1, 0] = True
    return value

def _decode_image(image_value):
    if isinstance(image_value, np.ndarray):
        return image_value
    if hasattr(image_value, "numpy"):
        maybe = image_value.numpy()
        if isinstance(maybe, np.ndarray):
            return maybe
        image_value = maybe
    if isinstance(image_value, (bytes, bytearray)):
        buf = np.frombuffer(image_value, dtype=np.uint8)
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Failed to decode camera image bytes.")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    raise TypeError(f"Unsupported image payload type: {type(image_value)}")

def _normalize_image_size(value):
    if value is None:
        return None
    return tuple(value)

def _episode_to_torch(episode):
    return {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in episode.items()}
