from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import hydra
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import DataLoader, Dataset
from types import SimpleNamespace


FRONT3_CAMERA_ORDER = (2, 1, 3)
FRONT3_CAMERA_TO_INDEX = {camera_name: idx for idx, camera_name in enumerate(FRONT3_CAMERA_ORDER)}
ACTION_KEYS = ("acceleration_mps2", "steer_rad")
MAP_REWARD_KEYS = (
    "reward_lane_alignment",
    "reward_lane_center_offset",
    "reward_collision",
    "reward_road_boundary",
)
WAYMO_COMPONENT_DIRS = ("camera_image", "lidar_box", "vehicle_pose", "map_features")

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

MAP_FEATURE_ID_COL = "key.map_feature_id"
MAP_FEATURE_TYPE_COL = "[MapFeatureComponent].feature_type"
MAP_LANE_POLYLINE_X_COL = "[MapFeatureComponent].lane.polyline.x"
MAP_LANE_POLYLINE_Y_COL = "[MapFeatureComponent].lane.polyline.y"
MAP_ROAD_EDGE_POLYLINE_X_COL = "[MapFeatureComponent].road_edge.polyline.x"
MAP_ROAD_EDGE_POLYLINE_Y_COL = "[MapFeatureComponent].road_edge.polyline.y"
MAP_ROAD_LINE_POLYLINE_X_COL = "[MapFeatureComponent].road_line.polyline.x"
MAP_ROAD_LINE_POLYLINE_Y_COL = "[MapFeatureComponent].road_line.polyline.y"
MAP_STOP_SIGN_POSITION_X_COL = "[MapFeatureComponent].stop_sign.position.x"
MAP_STOP_SIGN_POSITION_Y_COL = "[MapFeatureComponent].stop_sign.position.y"
MAP_STOP_SIGN_LANES_COL = "[MapFeatureComponent].stop_sign.lane"
MAP_CROSSWALK_POLYGON_X_COL = "[MapFeatureComponent].crosswalk.polygon.x"
MAP_CROSSWALK_POLYGON_Y_COL = "[MapFeatureComponent].crosswalk.polygon.y"
MAP_DRIVEWAY_POLYGON_X_COL = "[MapFeatureComponent].driveway.polygon.x"
MAP_DRIVEWAY_POLYGON_Y_COL = "[MapFeatureComponent].driveway.polygon.y"
MAP_SPEED_BUMP_POLYGON_X_COL = "[MapFeatureComponent].speed_bump.polygon.x"
MAP_SPEED_BUMP_POLYGON_Y_COL = "[MapFeatureComponent].speed_bump.polygon.y"
MAP_FEATURE_COLUMNS = (
    MAP_FEATURE_ID_COL,
    MAP_FEATURE_TYPE_COL,
    MAP_LANE_POLYLINE_X_COL,
    MAP_LANE_POLYLINE_Y_COL,
    MAP_ROAD_EDGE_POLYLINE_X_COL,
    MAP_ROAD_EDGE_POLYLINE_Y_COL,
    MAP_ROAD_LINE_POLYLINE_X_COL,
    MAP_ROAD_LINE_POLYLINE_Y_COL,
    MAP_STOP_SIGN_POSITION_X_COL,
    MAP_STOP_SIGN_POSITION_Y_COL,
    MAP_STOP_SIGN_LANES_COL,
    MAP_CROSSWALK_POLYGON_X_COL,
    MAP_CROSSWALK_POLYGON_Y_COL,
    MAP_DRIVEWAY_POLYGON_X_COL,
    MAP_DRIVEWAY_POLYGON_Y_COL,
    MAP_SPEED_BUMP_POLYGON_X_COL,
    MAP_SPEED_BUMP_POLYGON_Y_COL,
)

class WaymoPerceptionDataset(Dataset):
    def __init__(
        self,
        path,
        n_rollout=None,
        image_size=None,
        window_size=20,
        frame_skip=1,
        wheelbase_m=2.8,
        max_steer_rad=0.7,
        **kwargs,
    ):
        self.path = Path(path)
        self.window_size = window_size+1
        self.frame_skip = int(frame_skip)
        self.image_size = _normalize_image_size(image_size)
        self.wheelbase_m = float(wheelbase_m)
        self.max_steer_rad = float(max_steer_rad)
        self._ego_collision_poly = self.box_corners_2d(np.zeros(2, dtype=np.float64), 4.8, 2.0, 0.0)
        self._map_index_cache = {}
        self._map_feature_columns_cache = {}

        self.component_dirs = self.resolve_component_dirs(self.path)
        self.segment_paths = self.list_waymo_segments()
        if n_rollout:
            self.segment_paths = self.segment_paths[: int(n_rollout)]
        
        # preload vehicle pose
        self.vehicle_poses = []
        # list of -> ( list of timestep [...], list vehicle pose [...])
        self.load_poses()
        self.slices = []
        for segment_index, segment_id in enumerate(self.segment_paths):
            timestamps = self.vehicle_poses[segment_index][0]
            for start in range(0, len(timestamps), self.frame_skip):
                end = start + self.window_size * self.frame_skip
                if end > len(timestamps):
                    break
                self.slices.append((segment_index, start, end))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        segment_index, start, end = self.slices[index]
        segment_id = self.segment_paths[segment_index]
        segment_length = len(self.vehicle_poses[segment_index][0])
        timestamps = self.vehicle_poses[segment_index][0][start:end:self.frame_skip]
        step_ids = np.arange(start, end, self.frame_skip, dtype=np.int64)
        # load vehicle poses
        vehicle_poses = self.vehicle_poses[segment_index][1][start:end:self.frame_skip]
        actions = self.derive_action(vehicle_poses, timestamps=timestamps)
        # load camera images
        cameras = self._load_front_camera_images(segment_id, timestamps=timestamps)
        images = self.decode_camera_images(cameras)
        # load lidar boxes
        boxes = self._load_lidar_boxes(segment_id, timestamps=timestamps)
        # load map index
        map_feature = self._load_segment_map_index(segment_id)
        reward_columns = self.calculate_reward(
            map_feature=map_feature,
            vehicle_poss=vehicle_poses,
            lidar_boxes=boxes,
        )
        is_first = np.zeros((len(timestamps), 1), dtype=bool)
        is_last = np.zeros((len(timestamps), 1), dtype=bool)
        is_terminal = np.zeros((len(timestamps), 1), dtype=bool)
        if start == 0:
            is_first[0, 0] = True
        if end == segment_length:
            is_last[-1, 0] = True
            is_terminal[-1, 0] = True

        episode = {
            "image": images,
            "action": actions,
            "reward": reward_columns.sum(axis=-1, keepdims=True),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "episode": np.full((len(timestamps),), segment_index, dtype=np.int64),
            "step": step_ids,
        }
        for reward_idx, key in enumerate(MAP_REWARD_KEYS):
            episode[key] = reward_columns[:, reward_idx : reward_idx + 1].copy()
        return episode

    def resolve_component_dirs(self, root):
        root = Path(root)
        component_dirs = {name: root / name for name in WAYMO_COMPONENT_DIRS}
        missing = [str(path) for path in component_dirs.values() if not path.is_dir()]
        return component_dirs

    def list_waymo_segments(self):
        segment_sets = []
        for component_name in WAYMO_COMPONENT_DIRS:
            component_dir = self.component_dirs[component_name]
            segment_sets.append({path.stem for path in component_dir.glob("*.parquet")})
        shared_segments = sorted(set.intersection(*segment_sets)) if segment_sets else []
        return shared_segments

    def _component_path(self, component_name, segment_id):
        return self.component_dirs[component_name] / f"{segment_id}.parquet"

    def load_poses(self):
        # load all poses for all segments and store in self.vehicle_poses
        for segment_id in self.segment_paths:
            pose_data = self._load_pose_transforms(segment_id)
            self.vehicle_poses.append(pose_data)

    def _load_pose_transforms(self, segment_id):
        pose_table = pq.read_table(
            self._component_path("vehicle_pose", segment_id),
            columns=[FRAME_TIMESTAMP_COL, VEHICLE_POSE_TRANSFORM_COL],
            use_threads=True,
            memory_map=True,
        )
        timestamps = np.asarray(pose_table.column(FRAME_TIMESTAMP_COL).to_numpy(), dtype=np.int64)
        if timestamps.size == 0:
            return [], []

        transforms = pose_table.column(VEHICLE_POSE_TRANSFORM_COL).to_pylist()
        order = np.argsort(timestamps, kind="stable")
        pose_records = []
        for idx in order.tolist():
            pose_records.append(
                {
                    FRAME_TIMESTAMP_COL: int(timestamps[idx]),
                    VEHICLE_POSE_TRANSFORM_COL: np.asarray(
                        transforms[idx], dtype=np.float64
                    ).reshape(4, 4),
                }
            )

        unique_timestamps = []
        unique_poses = []
        seen = set()
        for record in pose_records:
            timestamp = record[FRAME_TIMESTAMP_COL]
            if timestamp in seen:
                continue
            seen.add(timestamp)
            unique_timestamps.append(timestamp)
            unique_poses.append(record[VEHICLE_POSE_TRANSFORM_COL])
        return unique_timestamps, unique_poses

    def _read_table(self, component_name, segment_id, columns):
        return pq.read_table(
            self._component_path(component_name, segment_id),
            columns=list(columns),
            use_threads=True,
            memory_map=True,
        )

    def _map_feature_columns(self, segment_id):
        cached = self._map_feature_columns_cache.get(segment_id)
        if cached is not None:
            return cached

        map_path = self._component_path("map_features", segment_id)
        available_columns = set(pq.read_schema(map_path).names)
        map_columns = tuple(column for column in MAP_FEATURE_COLUMNS if column in available_columns)
        if MAP_FEATURE_ID_COL not in map_columns or MAP_FEATURE_TYPE_COL not in map_columns:
            raise KeyError(
                f"Map features parquet {map_path} is missing required columns "
                f"{MAP_FEATURE_ID_COL!r} and/or {MAP_FEATURE_TYPE_COL!r}"
            )
        self._map_feature_columns_cache[segment_id] = map_columns
        return map_columns

    def _load_front_camera_images(self, segment_id, timestamps):
        requested_timestamps = timestamps

        camera_table = pq.read_table(
            self._component_path("camera_image", segment_id),
            columns=[FRAME_TIMESTAMP_COL, CAMERA_NAME_COL, CAMERA_IMAGE_COL],
            filters=[
                (FRAME_TIMESTAMP_COL, "in", requested_timestamps),
                (CAMERA_NAME_COL, "in", list(FRONT3_CAMERA_ORDER)),
            ],
            use_threads=True,
            memory_map=True,
        )
        table_timestamps = np.asarray(camera_table.column(FRAME_TIMESTAMP_COL).to_numpy(), dtype=np.int64)
        camera_names = np.asarray(camera_table.column(CAMERA_NAME_COL).to_numpy(), dtype=np.int64)
        image_values = camera_table.column(CAMERA_IMAGE_COL).to_pylist()

        camera_by_timestamp = {}
        for ts, name, image in zip(table_timestamps.tolist(), camera_names.tolist(), image_values):
            camera_by_timestamp.setdefault(int(ts), []).append(
                SimpleNamespace(name=int(name), image=image)
            )
        for ts in list(camera_by_timestamp.keys()):
            camera_by_timestamp[ts].sort(key=lambda img: int(img.name))

        ordered_images = []
        for timestamp in requested_timestamps:
            front_images = [None] * len(FRONT3_CAMERA_ORDER)
            for image in camera_by_timestamp[int(timestamp)]:
                front_images[FRONT3_CAMERA_TO_INDEX[int(image.name)]] = image.image
            ordered_images.append(front_images)
        return np.asarray(ordered_images, dtype=object)
    
    def decode_camera_images(self, front_camera_payloads):
        """Decode and stitch front camera payloads into one RGB image per timestamp.
        Input: payload array from _load_front_camera_images with shape (T, 3)
        ordered by timestamps and FRONT3_CAMERA_ORDER.
        Output: uint8 array with shape (T, 1280, 5760, 3).
        """
        stacked_by_timestamp = []
        for front_images in front_camera_payloads:
            decoded_images = []
            for image_value in front_images:
                bgr = cv2.imdecode(np.frombuffer(image_value, dtype=np.uint8), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                decoded_images.append(rgb)
            stacked_by_timestamp.append(np.concatenate(decoded_images, axis=1))
        return np.stack(stacked_by_timestamp, axis=0).astype(np.uint8)

    def _load_lidar_boxes(self, segment_id, timestamps):
        requested_timestamps = timestamps
        lidar_path = self._component_path("lidar_box", segment_id)
        lidar_columns = [
            FRAME_TIMESTAMP_COL,
            LIDAR_BOX_CENTER_X_COL,
            LIDAR_BOX_CENTER_Y_COL,
            LIDAR_BOX_SIZE_X_COL,
            LIDAR_BOX_SIZE_Y_COL,
            LIDAR_BOX_HEADING_COL,
            LIDAR_BOX_TYPE_COL,
        ]
        # not included: LIDAR_BOX_CENTER_Z_COL, LIDAR_BOX_SIZE_Z_COL, LIDAR_BOX_ID_COL
        lidar_table = pq.read_table(
            self._component_path("lidar_box", segment_id),
            filters=[(FRAME_TIMESTAMP_COL, "in", requested_timestamps)],
            columns=lidar_columns,
            use_threads=True,
            memory_map=True,
        )
        lidar_by_timestamp = {}
        lid_ts = np.asarray(lidar_table.column(FRAME_TIMESTAMP_COL).to_numpy(), dtype=np.int64)
        lid_x = lidar_table.column(LIDAR_BOX_CENTER_X_COL).to_pylist()
        lid_y = lidar_table.column(LIDAR_BOX_CENTER_Y_COL).to_pylist()
        lid_l = lidar_table.column(LIDAR_BOX_SIZE_X_COL).to_pylist()
        lid_w = lidar_table.column(LIDAR_BOX_SIZE_Y_COL).to_pylist()
        lid_heading = lidar_table.column(LIDAR_BOX_HEADING_COL).to_pylist()
        lid_type =  lidar_table.column(LIDAR_BOX_TYPE_COL).to_pylist()

        for idx, ts in enumerate(lid_ts.tolist()):
            box = SimpleNamespace(
                center_x=float(lid_x[idx]),
                center_y=float(lid_y[idx]),
                length=float(lid_l[idx]),
                width=float(lid_w[idx]),
                heading=float(lid_heading[idx]),
            )
            label = SimpleNamespace(box=box, type=int(lid_type[idx]))
            lidar_by_timestamp.setdefault(int(ts), []).append(label)

        ordered_lidar = []
        for timestamp in requested_timestamps:
            ordered_lidar.append(lidar_by_timestamp.get(timestamp, []))
        return np.asarray(ordered_lidar, dtype=object)

    def _load_segment_map_index(self, segment_id, timestamps=None):
        if timestamps is None:
            cached = self._map_index_cache.get(segment_id)
            if cached is not None:
                return cached

        map_columns = list(self._map_feature_columns(segment_id))
        map_path = self._component_path("map_features", segment_id)
        schema_names = set(pq.read_schema(map_path).names)
        use_timestamp_filter = timestamps is not None and FRAME_TIMESTAMP_COL in schema_names
        if use_timestamp_filter and FRAME_TIMESTAMP_COL not in map_columns:
            map_columns.append(FRAME_TIMESTAMP_COL)

        map_table = pq.read_table(
            map_path,
            columns=map_columns,
            use_threads=True,
            memory_map=True,
        )

        if use_timestamp_filter:
            requested = np.asarray(list(timestamps), dtype=np.int64)
            if requested.size == 0:
                map_table = map_table.slice(0, 0)
            else:
                table_timestamps = np.asarray(map_table.column(FRAME_TIMESTAMP_COL).to_numpy(), dtype=np.int64)
                keep_idx = np.flatnonzero(np.isin(table_timestamps, requested)).tolist()
                map_table = map_table.take(keep_idx) if keep_idx else map_table.slice(0, 0)

        map_index = self.build_map_index(map_table)
        if timestamps is None:
            self._map_index_cache[segment_id] = map_index
        return map_index

    def load_one_map_feature_by_segment_index(self, segment_index, frame_number=None):
        segment_id = self.segment_paths[int(segment_index)]
        map_columns = self._map_feature_columns(segment_id)
        map_table = pq.read_table(
            self._component_path("map_features", segment_id),
            columns=list(map_columns),
            use_threads=True,
            memory_map=True,
        )
        return {
            "segment_id": segment_id,
            "map_index": self.build_map_index(map_table),
        }

    def prefetch_map_features_by_segment_index(self):
        for segment_index in range(len(self.segment_paths)):
            data = self.load_one_map_feature_by_segment_index(segment_index)
            import pdb; pdb.set_trace()
            self._map_index_cache[data["segment_id"]] = data["map_index"]

    def derive_action(self, vehicle_poses, timestamps):
        """Derive per-step actions from a sequence of ego vehicle poses.
        Returns a dense array with shape (T, 2) in ACTION_KEYS order:
        [acceleration_mps2, steer_rad]. The first step is zeros.
        """
        poses = list(vehicle_poses)
        num_steps = len(poses)
        actions = np.zeros((num_steps, len(ACTION_KEYS)), dtype=np.float32)
        if num_steps <= 1:
            return actions

        prev_speed_mps = 0.0
        for idx in range(1, num_steps):
            previous_pose = np.asarray(poses[idx - 1], dtype=np.float64).reshape(4, 4)
            current_pose = np.asarray(poses[idx], dtype=np.float64).reshape(4, 4)

            current_translation = current_pose[:3, 3]
            previous_translation = previous_pose[:3, 3]
            delta_world = current_translation - previous_translation
            delta_vehicle = previous_pose[:3, :3].T @ delta_world

            yaw_delta = self.wrap_angle(
                self.yaw_from_pose(current_pose) - self.yaw_from_pose(previous_pose)
            )

            dt_sec = max((float(timestamps[idx]) - float(timestamps[idx - 1])) / 1e6, 1e-6)

            speed_mps = float(delta_vehicle[0] / dt_sec)
            acceleration_mps2 = float((speed_mps - prev_speed_mps) / dt_sec)
            yaw_rate_rps = float(yaw_delta / dt_sec)
            if abs(speed_mps) < 1e-4:
                steer_rad = 0.0
            else:
                steer_rad = float(np.arctan(self.wheelbase_m * yaw_rate_rps / speed_mps))
            steer_rad = float(np.clip(steer_rad, -self.max_steer_rad, self.max_steer_rad))

            actions[idx, 0] = acceleration_mps2
            actions[idx, 1] = steer_rad
            prev_speed_mps = speed_mps
        return actions

    def calculate_reward(self, map_feature, vehicle_poss, lidar_boxes=None):
        num_steps = len(vehicle_poss)
        rewards = np.zeros((num_steps, len(MAP_REWARD_KEYS)), dtype=np.float32)
        if num_steps == 0:
            return rewards

        if lidar_boxes is None:
            lidar_boxes = [None] * num_steps

        has_map_geometry = bool(map_feature.get("lanes")) or bool(map_feature.get("road_edges"))
        previous_frame = None
        for idx, pose_transform in enumerate(vehicle_poss):
            box_rows = []
            current_lidar_boxes = lidar_boxes[idx]
            if current_lidar_boxes is None:
                current_lidar_boxes = ()
            for label in current_lidar_boxes:
                if label is None or getattr(label, "box", None) is None:
                    continue
                box = label.box
                box_rows.append(
                    [
                        float(box.center_x),
                        float(box.center_y),
                        float(box.length),
                        float(box.width),
                        float(box.heading),
                    ]
                )

            frame = SimpleNamespace(
                pose_transform=np.asarray(pose_transform, dtype=np.float64).reshape(4, 4),
                lidar_boxes=np.asarray(box_rows, dtype=np.float64) if box_rows else np.zeros((0, 5), dtype=np.float64),
            )

            lane_info = None
            leaves_road_boundary = False
            if has_map_geometry:
                lane_info, leaves_road_boundary, _, _ = self.calculate_map_metrics(
                    frame,
                    map_index=map_feature,
                    previous_frame=previous_frame,
                )

            rewards[idx, 0] = 0.0 if lane_info is None else -(float(lane_info["alignment_angle"]) ** 2)
            rewards[idx, 1] = 0.0 if lane_info is None else -(float(lane_info["distance_to_center"]) ** 2)
            rewards[idx, 2] = -1.0 if self.detect_collision(frame) else 0.0
            rewards[idx, 3] = -1.0 if leaves_road_boundary else 0.0
            previous_frame = frame
        return rewards

    def compute_reward_row(self, frame, previous_frame, map_index):
        lane_info, leaves_road_boundary, collision, cross_stop_line = self.calculate_map_metrics(
            frame,
            map_index=map_index,
            previous_frame=previous_frame,
        )
        lane_alignment = 0.0 if lane_info is None else -abs(float(lane_info["alignment_angle"]))
        lane_center = 0.0 if lane_info is None else -abs(float(lane_info["distance_to_center"]))
        return np.asarray(
            [
                lane_alignment,
                lane_center,
                -1.0 if collision else 0.0,
                -1.0 if leaves_road_boundary else 0.0,
                -1.0 if cross_stop_line else 0.0,
            ],
            dtype=np.float32,
        )

    def build_map_index(self, map_table):
        map_data = map_table.to_pydict()
        feature_ids = np.asarray(map_data[MAP_FEATURE_ID_COL], dtype=np.int64)
        if feature_ids.size == 0:
            return {"lanes": {}, "road_edges": [], "stop_signs": []}

        feature_types = map_data[MAP_FEATURE_TYPE_COL]
        lane_x = map_data.get(MAP_LANE_POLYLINE_X_COL)
        lane_y = map_data.get(MAP_LANE_POLYLINE_Y_COL)
        road_edge_x = map_data.get(MAP_ROAD_EDGE_POLYLINE_X_COL)
        road_edge_y = map_data.get(MAP_ROAD_EDGE_POLYLINE_Y_COL)
        stop_sign_x = map_data.get(MAP_STOP_SIGN_POSITION_X_COL)
        stop_sign_y = map_data.get(MAP_STOP_SIGN_POSITION_Y_COL)
        stop_sign_lanes = map_data.get(MAP_STOP_SIGN_LANES_COL)

        lanes = {}
        road_edges = []
        stop_signs = []
        for row_idx in np.argsort(feature_ids, kind="stable").tolist():
            feature_type = feature_types[row_idx]
            feature_type = "" if feature_type is None else str(feature_type).lower()
            feature_id = int(feature_ids[row_idx])

            if feature_type == "lane":
                polyline = _polyline_from_lists(
                    lane_x[row_idx] if lane_x is not None else None,
                    lane_y[row_idx] if lane_y is not None else None,
                )
                if len(polyline) == 0:
                    continue
                lanes[feature_id] = {
                    "id": feature_id,
                    "polyline": polyline,
                }
                continue

            if feature_type == "road_edge":
                polyline = _polyline_from_lists(
                    road_edge_x[row_idx] if road_edge_x is not None else None,
                    road_edge_y[row_idx] if road_edge_y is not None else None,
                )
                if len(polyline) == 0:
                    continue
                road_edges.append({"id": feature_id, "polyline": polyline})
                continue

            if feature_type != "stop_sign" or stop_sign_x is None or stop_sign_y is None:
                continue

            pos_x = stop_sign_x[row_idx]
            pos_y = stop_sign_y[row_idx]
            if _is_missing(pos_x) or _is_missing(pos_y):
                continue
            lane_ids = [
                int(lane_id)
                for lane_id in _to_list(stop_sign_lanes[row_idx] if stop_sign_lanes is not None else None)
                if not _is_missing(lane_id)
            ]
            stop_signs.append(
                {
                    "id": feature_id,
                    "position": np.asarray([float(pos_x), float(pos_y)], dtype=np.float64),
                    "lane_ids": lane_ids,
                }
            )

        return {"lanes": lanes, "road_edges": road_edges, "stop_signs": stop_signs}

    def yaw_from_pose(self, transform):
        return float(np.arctan2(transform[1, 0], transform[0, 0]))

    def wrap_angle(self, angle):
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def closest_point_on_polyline(self, point_xy_world, polyline):
        if len(polyline) == 0:
            raise ValueError("Polyline must contain at least one point")
        if len(polyline) == 1:
            return float(np.linalg.norm(point_xy_world - polyline[0])), polyline[0], np.array([1.0, 0.0]), 0, 0.0

        segment_starts = polyline[:-1]
        segments = polyline[1:] - segment_starts
        segment_lengths_sq = np.einsum("ij,ij->i", segments, segments)
        valid_segment_idx = np.flatnonzero(segment_lengths_sq > 1e-9)
        if valid_segment_idx.size == 0:
            return float(np.linalg.norm(point_xy_world - polyline[0])), polyline[0], np.array([1.0, 0.0]), 0, 0.0

        valid_starts = segment_starts[valid_segment_idx]
        valid_segments = segments[valid_segment_idx]
        valid_lengths_sq = segment_lengths_sq[valid_segment_idx]
        offsets = point_xy_world - valid_starts
        u = np.clip(np.einsum("ij,ij->i", offsets, valid_segments) / valid_lengths_sq, 0.0, 1.0)
        candidates = valid_starts + valid_segments * u[:, None]
        deltas = candidates - point_xy_world
        distances_sq = np.einsum("ij,ij->i", deltas, deltas)

        best_local_idx = int(np.argmin(distances_sq))
        best_segment_idx = int(valid_segment_idx[best_local_idx])
        best_tangent = valid_segments[best_local_idx] / np.sqrt(valid_lengths_sq[best_local_idx])
        return (
            float(np.sqrt(distances_sq[best_local_idx])),
            candidates[best_local_idx],
            best_tangent,
            best_segment_idx,
            float(u[best_local_idx]),
        )

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

    def nearest_lane_info(self, ego_xy, ego_heading, map_index):
        best = None
        for lane_id, lane_info in map_index["lanes"].items():
            distance, _, tangent_xy, _, _ = self.closest_point_on_polyline(ego_xy, lane_info["polyline"])
            heading = float(np.arctan2(tangent_xy[1], tangent_xy[0]))
            candidate = {
                "lane_id": lane_id,
                "distance_to_center": distance,
                "tangent_xy": tangent_xy,
                "alignment_angle": self.wrap_angle(ego_heading - heading),
            }
            if best is None or candidate["distance_to_center"] < best["distance_to_center"]:
                best = candidate
        return best

    def nearest_road_edge_distance(self, ego_xy, map_index):
        best_distance = None
        for edge in map_index["road_edges"]:
            distance, _, _, _, _ = self.closest_point_on_polyline(ego_xy, edge["polyline"])
            if best_distance is None or distance < best_distance:
                best_distance = distance
        return best_distance

    def detect_collision(self, frame):
        if len(frame.lidar_boxes) == 0:
            return False

        for box in frame.lidar_boxes:
            other_poly = self.box_corners_2d(box[:2], float(box[2]), float(box[3]), float(box[4]))
            if self.polygons_intersect(self._ego_collision_poly, other_poly):
                return True
        return False

    def detect_cross_stop_line(self, current_xy, previous_xy, lane_info, map_index):
        if previous_xy is None or lane_info is None:
            return False

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
        current_xy = frame.pose_transform[:2, 3]
        ego_heading = self.yaw_from_pose(frame.pose_transform)
        lane_info = self.nearest_lane_info(current_xy, ego_heading, map_index) if map_index["lanes"] else None
        road_edge_distance = self.nearest_road_edge_distance(current_xy, map_index) if map_index["road_edges"] else None

        leaves_road_boundary = lane_info is None
        if lane_info is not None:
            leaves_road_boundary = lane_info["distance_to_center"] > 2.5
            if road_edge_distance is not None:
                leaves_road_boundary = leaves_road_boundary or road_edge_distance < 1.0

        previous_xy = None if previous_frame is None else previous_frame.pose_transform[:2, 3]
        return (
            lane_info,
            bool(leaves_road_boundary),
            self.detect_collision(frame),
            self.detect_cross_stop_line(current_xy, previous_xy, lane_info, map_index),
        )

def _flag(time_dim, first=False, last=False):
    value = np.zeros((time_dim, 1), dtype=bool)
    if first:
        value[0, 0] = True
    if last:
        value[-1, 0] = True
    return value

def _is_missing(value):
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False

def _to_list(value):
    if value is None or _is_missing(value):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]

def _polyline_from_lists(xs, ys):
    points = []
    for x, y in zip(_to_list(xs), _to_list(ys)):
        if _is_missing(x) or _is_missing(y):
            continue
        points.append((float(x), float(y)))
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)

def _normalize_image_size(value):
    if value is None:
        return None
    return tuple(value)
