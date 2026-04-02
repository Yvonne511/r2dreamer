import argparse
import sys
import types
from pathlib import Path

import cv2
import numpy as np

## Have issues
# https://github.com/waymo-research/waymo-open-dataset/pull/927
# https://github.com/waymo-research/waymo-open-dataset/issues/916
# install new version, copy python files, install old versions, place python files
    # end_to_end_driving_data_pb2.py
    # end_to_end_driving_submission_pb2.py
    # waymo-open-dataset-tf-2-12-0==1.6.4

# TensorFlow may import jax through tensorflow.lite; some environments have a
# broken jax/numpy combination that crashes before any Waymo code runs.
if not hasattr(np, "issubsctype"):
    np.issubsctype = np.issubdtype
if "jax" not in sys.modules:
    jax_stub = types.ModuleType("jax")
    jax_stub.jit = lambda fn=None, *args, **kwargs: fn
    sys.modules["jax"] = jax_stub

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
import tensorflow as tf


TRAIN_DIR = Path("/scratch/yw4142/datasets/ad/waymo_open_dataset_end_to_end_camera_v_1_0_0/train")
VALID_DIR = Path("/scratch/yw4142/datasets/ad/waymo_open_dataset_end_to_end_camera_v_1_0_0/valid")
TRAIN_SORTED_DIR = Path("/scratch/yw4142/datasets/ad/waymo_open_dataset_end_to_end_camera_v_1_0_0/train_sorted/segments")
VALID_SORTED_DIR = Path("/scratch/yw4142/datasets/ad/waymo_open_dataset_end_to_end_camera_v_1_0_0/valid_sorted/segments")
DEFAULT_OUTPUT_DIR = Path("/home/yw4142/ad/r2dreamer/test/waymo_vis_outputs")
FRONT3_CAMERA_ORDER = [2, 1, 3]
CAMERA_NAME_TO_LABEL = {
    1: "front",
    2: "front_left",
    3: "front_right",
}
SHORT_SEGMENT_LENGTH = 12


def list_tfrecords(split: str) -> list[str]:
    split_to_root = {
        "train": TRAIN_DIR,
        "valid": VALID_DIR,
        "train_sorted": TRAIN_SORTED_DIR,
        "valid_sorted": VALID_SORTED_DIR,
    }
    root = split_to_root[split]
    files = sorted(str(path) for path in root.glob("*.tfrecord*"))
    if not files:
        raise FileNotFoundError(f"No TFRecord files found in {root}")
    return files


def iter_examples(split: str):
    dataset = tf.data.TFRecordDataset(list_tfrecords(split), compression_type="")
    dataset_iter = dataset.as_numpy_iterator()
    return dataset_iter

def return_front3_cameras(data: wod_e2ed_pb2.E2EDFrame):
    """Return the front_left, front, and front_right cameras as a list of images"""
    image_list = []
    calibration_list = []
    # CameraName Enum reference:
    # https://github.com/waymo-research/waymo-open-dataset/blob/5f8a1cd42491210e7de629b6f8fc09b65e0cbe99/src/waymo_open_dataset/dataset.proto#L50
    order = [2, 1, 3]
    for camera_name in order:
        for index, image_content in enumerate(data.frame.images):
            if image_content.name == camera_name:
                # Decode the raw image string and convert to numpy type.
                calibration = data.frame.context.camera_calibrations[index]
                image = tf.io.decode_image(image_content.image).numpy()
                image_list.append(image)
                calibration_list.append(calibration)
                break

    return image_list, calibration_list

def return_all_cameras(data: wod_e2ed_pb2.E2EDFrame):
    """Returns all cameras in the frame."""
    image_list = []
    calibration_list = []
    order = [4, 2, 1, 3, 5, 6, 7, 8]
    for camera_name in order:
        for index, image_content in enumerate(data.frame.images):
            if image_content.name == camera_name:
                # Decode the raw image string and convert to numpy type.
                calibration = data.frame.context.camera_calibrations[index]
                image = tf.io.decode_image(image_content.image).numpy()
                image_list.append(image)
                calibration_list.append(calibration)
                break
    return image_list, calibration_list


def project_vehicle_to_image(vehicle_pose, calibration, points):
    """Projects from vehicle coordinate system to image with global shutter.

    Arguments:
        vehicle_pose: Vehicle pose transform from vehicle into world coordinate
        system.
        calibration: Camera calibration details (including intrinsics/extrinsics).
        points: Points to project of shape [N, 3] in vehicle coordinate system.

    Returns:
        Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
    """
    # Transform points from vehicle to world coordinate system (can be
    # vectorized).
    pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
    world_points = np.zeros_like(points)
    for i, point in enumerate(points):
        cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
        world_points[i] = (cx, cy, cz)

    # Populate camera image metadata. Velocity and latency stats are filled with
    # zeroes.
    extrinsic = tf.reshape(
        tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32),
        [4, 4])
    intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
    metadata = tf.constant([
        calibration.width,
        calibration.height,
        open_dataset.CameraCalibration.GLOBAL_SHUTTER,
    ],
                            dtype=tf.int32)
    camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10

    # Perform projection and return projected image coordinates (u, v, ok).
    return py_camera_model_ops.world_to_image(extrinsic, intrinsic, metadata,
                                                camera_image_metadata,
                                                world_points).numpy()

def draw_points_on_image(image, points, size):
    """Draws points on an image.

    Args:
    image: The image to draw on.
    points: A numpy array of shape (N, 2) representing the points to draw.
    """
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), size, (255, 0, 0), -1)
    return image


def format_action_summary(data: wod_e2ed_pb2.E2EDFrame) -> str:
    """Builds a short action summary from the ego driving logs."""
    past_states = data.past_states
    intent_name = wod_e2ed_pb2.EgoIntent.Intent.Name(data.intent)

    if len(past_states.vel_x) == 0:
        return f"intent={intent_name}"

    return (
        f"intent={intent_name} "
        f"vel=({past_states.vel_x[-1]:.2f}, {past_states.vel_y[-1]:.2f}) "
        f"accel=({past_states.accel_x[-1]:.2f}, {past_states.accel_y[-1]:.2f})"
    )


def format_timestep_info(data: wod_e2ed_pb2.E2EDFrame) -> dict:
    return {
        "context_name": data.frame.context.name,
        "timestamp_micros": data.frame.timestamp_micros,
        "frame_fields": [field.name for field, _ in data.frame.ListFields()],
        "e2ed_fields": [field.name for field, _ in data.ListFields()],
        "num_images": len(data.frame.images),
        "num_lasers": len(data.frame.lasers),
        "num_camera_labels": len(data.frame.camera_labels),
        "past_state_len": len(data.past_states.pos_x),
        "future_state_len": len(data.future_states.pos_x),
    }

import re
from collections import Counter, defaultdict
UUID_IDX_RE = re.compile(r"^([0-9a-fA-F]{32})[-_](\d+)$")

def main():
    ## Raw data Test
    # dataset_iter = iter_examples("valid")
    # DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # # Render Front Imgs with Points
    # for idx in range(40):
    #     bytes_example = next(dataset_iter)
    #     data = wod_e2ed_pb2.E2EDFrame()
    #     data.ParseFromString(bytes_example)

    #     front3_camera_image_list, front3_camera_calibration_list = return_front3_cameras(data)
    #     concatenated_image = np.concatenate(front3_camera_image_list, axis=1)
    #     future_waypoints_matrix = np.stack([data.future_states.pos_x, data.future_states.pos_y, data.future_states.pos_z], axis=1)
    #     vehicle_pose = data.frame.images[0].pose

    #     images_with_drawn_points = []
    #     for i in range(len(front3_camera_calibration_list)):
    #         waypoints_camera_space = project_vehicle_to_image(vehicle_pose, front3_camera_calibration_list[i], future_waypoints_matrix)
    #         images_with_drawn_points.append(draw_points_on_image(front3_camera_image_list[i], waypoints_camera_space, size=15))
    #         concatenated_image = np.concatenate(images_with_drawn_points, axis=1)

    #     output_path = DEFAULT_OUTPUT_DIR / f"waymo_rendered_{idx:02d}.png"
    #     cv2.imwrite(str(output_path), cv2.cvtColor(concatenated_image, cv2.COLOR_RGB2BGR))

    # Render All Images
    # for idx in range(40):
    #     bytes_example = next(dataset_iter)
    #     data = wod_e2ed_pb2.E2EDFrame()
    #     data.ParseFromString(bytes_example)

    #     import pdb; pdb.set_trace()

    #     all_camera_image_list, all_camera_calibration_list = return_all_cameras(data)
        
    #     max_h = max(img.shape[0] for img in all_camera_image_list)
    #     padded = []
    #     for img in all_camera_image_list:
    #         h, w, c = img.shape
    #         pad_h = max_h - h
    #         img_pad = np.pad(img, ((0, pad_h), (0, 0), (0, 0)), mode="constant")
    #         padded.append(img_pad)

    #     concatenated_image = np.concatenate(padded, axis=1)

    #     # concatenated_image = np.concatenate(all_camera_image_list, axis=1)
    #     output_path = DEFAULT_OUTPUT_DIR / f"waymo_full_{idx:02d}.png"
    #     cv2.imwrite(str(output_path), cv2.cvtColor(concatenated_image, cv2.COLOR_RGB2BGR))

    # Render short Front3 video segments and print actions
    # for idx in range(3):
    #     bytes_example = next(dataset_iter)
    #     data = wod_e2ed_pb2.E2EDFrame()
    #     data.ParseFromString(bytes_example)

    #     segment_id = data.frame.context.name
    #     video_frames = []
    #     action_summaries = [format_action_summary(data)]

    #     front3_camera_image_list, _ = return_front3_cameras(data)
    #     video_frames.append(np.concatenate(front3_camera_image_list, axis=1))

    #     for _ in range(SHORT_SEGMENT_LENGTH - 1):
    #         bytes_example = next(dataset_iter)
    #         next_data = wod_e2ed_pb2.E2EDFrame()
    #         next_data.ParseFromString(bytes_example)

    #         if next_data.frame.context.name != segment_id:
    #             break

    #         front3_camera_image_list, _ = return_front3_cameras(next_data)
    #         video_frames.append(np.concatenate(front3_camera_image_list, axis=1))
    #         action_summaries.append(format_action_summary(next_data))

    #     output_path = DEFAULT_OUTPUT_DIR / f"waymo_front3_segment_{idx:02d}.mp4"
    #     frame_h, frame_w = video_frames[0].shape[:2]
    #     writer = cv2.VideoWriter(
    #         str(output_path),
    #         cv2.VideoWriter_fourcc(*"mp4v"),
    #         10,
    #         (frame_w, frame_h),
    #     )
    #     for frame in video_frames:
    #         writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    #     writer.release()

    #     print(f"segment {idx} id={segment_id} frames={len(video_frames)}")
    #     for frame_idx, action_summary in enumerate(action_summaries):
    #         print(f"  frame {frame_idx:02d}: {action_summary}")

    # check counter
    # segment_counts = Counter()
    # segment_keys = defaultdict(list)

    # from tqdm import tqdm
    # for i in tqdm(range(5000)):
    #     bytes_example = next(dataset_iter)
    #     data = wod_e2ed_pb2.E2EDFrame()
    #     data.ParseFromString(bytes_example)
    #     frame_name = data.frame.context.name
    #     m = UUID_IDX_RE.match(frame_name)
    #     seg = m.group(1).lower()
    #     idx = int(m.group(2))
    #     segment_counts[seg] += 1
    #     segment_keys[seg].append(idx)

    # print("num unique segments:", len(segment_counts))
    # print("most common:", segment_counts.most_common(20))

    ## Sorted data test
    dataset_iter = iter_examples("valid_sorted")
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_frames = []
    for idx, bytes_example in enumerate(dataset_iter):
        data = wod_e2ed_pb2.E2EDFrame()
        data.ParseFromString(bytes_example)

        front3_camera_image_list, _ = return_front3_cameras(data)
        video_frames.append(np.concatenate(front3_camera_image_list, axis=1))

        print(f"timestep {idx:03d}")
        print(f"  action: {format_action_summary(data)}")
        print(f"  info: {format_timestep_info(data)}")

        if idx == 0:
            print(f"  all timestep fields: {[field.name for field, _ in data.ListFields()]}")

        if idx + 1 >= SHORT_SEGMENT_LENGTH:
            break

    output_path = DEFAULT_OUTPUT_DIR / "waymo_valid_sorted_front3.mp4"
    frame_h, frame_w = video_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (frame_w, frame_h),
    )
    for frame in video_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    print(f"saved video to {output_path}")

if __name__ == "__main__":
    main()
