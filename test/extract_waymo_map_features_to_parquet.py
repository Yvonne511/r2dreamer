#!/usr/bin/env python3
"""
Extract static Waymo map features from v1 TFRecords into per-segment parquet.

The output naming follows the existing Waymo v2-style folders, e.g.
`<segment_context_name>.parquet`, and the columns use flattened component-style
names such as `key.segment_context_name` and
`[MapFeatureComponent].lane.polyline.x`.

Example:
  /home/yw4142/container.sh
  mamba activate r2dreamer
  python /home/yw4142/ad/r2dreamer/test/extract_waymo_map_features_to_parquet.py \
      --input-dir /scratch/yw4142/datasets/ad/waymo_open_dataset_v_1_4_3/training \
      --output-dir /scratch/yw4142/datasets/ad/waymo_open_dataset_v_2_0_1/training/map_features \
      --workers 8
"""

import argparse
import dataclasses
import multiprocessing
import os
import re
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from waymo_open_dataset.v2 import column_types
from waymo_open_dataset.v2 import component
from waymo_open_dataset.v2.perception import base


DEFAULT_INPUT_DIR = Path("/scratch/yw4142/datasets/ad/waymo_open_dataset_v_1_4_3/training")
DEFAULT_OUTPUT_DIR = Path("/scratch/yw4142/datasets/ad/waymo_open_dataset_v_2_0_1/training/map_features")

SEGMENT_NAME_RE = re.compile(
    r"^segment-(?P<segment_id>.+?)(?:_with_camera_labels)?(?:\.tfrecord(?:s)?(?:\.(?:gz|gzip))?)$"
)

_column = component.create_column
_TF = None
_OPEN_DATASET = None


@dataclasses.dataclass(frozen=True)
class MapFeatureKey(base.SegmentKey):
    map_feature_id: int = _column(arrow_type=pa.int64())


@dataclasses.dataclass(frozen=True)
class BoundarySegmentList:
    lane_start_index: list[int] = _column(arrow_type=pa.list_(pa.int32()), default_factory=list)
    lane_end_index: list[int] = _column(arrow_type=pa.list_(pa.int32()), default_factory=list)
    boundary_feature_id: list[int] = _column(arrow_type=pa.list_(pa.int64()), default_factory=list)
    boundary_type: list[int] = _column(arrow_type=pa.list_(pa.int8()), default_factory=list)


@dataclasses.dataclass(frozen=True)
class NestedBoundarySegmentList:
    lane_start_index: list[list[int]] = _column(arrow_type=pa.list_(pa.list_(pa.int32())), default_factory=list)
    lane_end_index: list[list[int]] = _column(arrow_type=pa.list_(pa.list_(pa.int32())), default_factory=list)
    boundary_feature_id: list[list[int]] = _column(arrow_type=pa.list_(pa.list_(pa.int64())), default_factory=list)
    boundary_type: list[list[int]] = _column(arrow_type=pa.list_(pa.list_(pa.int8())), default_factory=list)


@dataclasses.dataclass(frozen=True)
class LaneNeighborList:
    feature_id: list[int] = _column(arrow_type=pa.list_(pa.int64()), default_factory=list)
    self_start_index: list[int] = _column(arrow_type=pa.list_(pa.int32()), default_factory=list)
    self_end_index: list[int] = _column(arrow_type=pa.list_(pa.int32()), default_factory=list)
    neighbor_start_index: list[int] = _column(arrow_type=pa.list_(pa.int32()), default_factory=list)
    neighbor_end_index: list[int] = _column(arrow_type=pa.list_(pa.int32()), default_factory=list)
    boundaries: NestedBoundarySegmentList = _column(default_factory=NestedBoundarySegmentList)


@dataclasses.dataclass(frozen=True)
class LaneFeatureData:
    speed_limit_mph: float = _column(arrow_type=pa.float64())
    type: int = _column(arrow_type=pa.int8())
    interpolating: bool = _column(arrow_type=pa.bool_())
    polyline: column_types.Vec3dList = _column()
    entry_lanes: list[int] = _column(arrow_type=pa.list_(pa.int64()), default_factory=list)
    exit_lanes: list[int] = _column(arrow_type=pa.list_(pa.int64()), default_factory=list)
    left_boundaries: BoundarySegmentList = _column(default_factory=BoundarySegmentList)
    right_boundaries: BoundarySegmentList = _column(default_factory=BoundarySegmentList)
    left_neighbors: LaneNeighborList = _column(default_factory=LaneNeighborList)
    right_neighbors: LaneNeighborList = _column(default_factory=LaneNeighborList)


@dataclasses.dataclass(frozen=True)
class RoadLineFeatureData:
    type: int = _column(arrow_type=pa.int8())
    polyline: column_types.Vec3dList = _column()


@dataclasses.dataclass(frozen=True)
class RoadEdgeFeatureData:
    type: int = _column(arrow_type=pa.int8())
    polyline: column_types.Vec3dList = _column()


@dataclasses.dataclass(frozen=True)
class StopSignFeatureData:
    position: column_types.Vec3d = _column()
    lane: list[int] = _column(arrow_type=pa.list_(pa.int64()), default_factory=list)


@dataclasses.dataclass(frozen=True)
class PolygonFeatureData:
    polygon: column_types.Vec3dList = _column()


@dataclasses.dataclass(frozen=True)
class MapFeatureComponent(component.Component):
    key: MapFeatureKey
    feature_type: str = _column(arrow_type=pa.string())
    lane: Optional[LaneFeatureData] = _column(default=None)
    road_line: Optional[RoadLineFeatureData] = _column(default=None)
    road_edge: Optional[RoadEdgeFeatureData] = _column(default=None)
    stop_sign: Optional[StopSignFeatureData] = _column(default=None)
    crosswalk: Optional[PolygonFeatureData] = _column(default=None)
    speed_bump: Optional[PolygonFeatureData] = _column(default=None)
    driveway: Optional[PolygonFeatureData] = _column(default=None)


FULL_SCHEMA = pa.schema([pa.field("index", pa.string()), *list(MapFeatureComponent.schema())])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract static map features from Waymo v1 TFRecords into per-segment parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=1, help="Number of TFRecord shards to process in parallel.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N TFRecord files.")
    parser.add_argument(
        "--compression-type",
        choices=("auto", "none", "gzip"),
        default="auto",
        help="How to open the TFRecord source files.",
    )
    parser.add_argument(
        "--parquet-compression",
        choices=("snappy", "gzip", "brotli", "zstd", "none"),
        default="snappy",
        help="Compression codec for the output parquet files.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing parquet files.")
    return parser.parse_args()


def import_waymo_runtime():
    global _TF, _OPEN_DATASET
    if _TF is not None and _OPEN_DATASET is not None:
        return _TF, _OPEN_DATASET

    import tensorflow.compat.v1 as tf
    from waymo_open_dataset import dataset_pb2 as open_dataset

    if hasattr(tf, "executing_eagerly") and not tf.executing_eagerly():
        tf.enable_eager_execution()
    _TF = tf
    _OPEN_DATASET = open_dataset
    return _TF, _OPEN_DATASET


def list_tfrecord_files(input_dir: Path) -> list[Path]:
    files = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and ".tfrecord" in path.name:
            files.append(path)
    if not files:
        raise FileNotFoundError(f"No TFRecord files found under {input_dir}")
    return files


def segment_id_from_path(path: Path) -> str:
    match = SEGMENT_NAME_RE.match(path.name)
    if match:
        return match.group("segment_id")

    name = path.name
    for suffix in (".tfrecord.gz", ".tfrecord.gzip", ".tfrecords", ".tfrecord"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = name.removeprefix("segment-")
    return name.removesuffix("_with_camera_labels")


def tfrecord_compression(path: Path, compression_type: str) -> str:
    if compression_type == "gzip":
        return "GZIP"
    if compression_type == "none":
        return ""
    name = path.name.lower()
    return "GZIP" if name.endswith(".gz") or name.endswith(".gzip") else ""


def parquet_compression(codec: str):
    return None if codec == "none" else codec


def vec3d(point) -> column_types.Vec3d:
    return column_types.Vec3d(x=float(point.x), y=float(point.y), z=float(point.z))


def vec3d_list(points) -> column_types.Vec3dList:
    return column_types.Vec3dList(
        x=[float(point.x) for point in points],
        y=[float(point.y) for point in points],
        z=[float(point.z) for point in points],
    )


def boundary_segment_list(segments) -> BoundarySegmentList:
    return BoundarySegmentList(
        lane_start_index=[int(segment.lane_start_index) for segment in segments],
        lane_end_index=[int(segment.lane_end_index) for segment in segments],
        boundary_feature_id=[int(segment.boundary_feature_id) for segment in segments],
        boundary_type=[int(segment.boundary_type) for segment in segments],
    )


def nested_boundary_segment_list(neighbors) -> NestedBoundarySegmentList:
    return NestedBoundarySegmentList(
        lane_start_index=[[int(boundary.lane_start_index) for boundary in neighbor.boundaries] for neighbor in neighbors],
        lane_end_index=[[int(boundary.lane_end_index) for boundary in neighbor.boundaries] for neighbor in neighbors],
        boundary_feature_id=[[int(boundary.boundary_feature_id) for boundary in neighbor.boundaries] for neighbor in neighbors],
        boundary_type=[[int(boundary.boundary_type) for boundary in neighbor.boundaries] for neighbor in neighbors],
    )


def lane_neighbor_list(neighbors) -> LaneNeighborList:
    return LaneNeighborList(
        feature_id=[int(neighbor.feature_id) for neighbor in neighbors],
        self_start_index=[int(neighbor.self_start_index) for neighbor in neighbors],
        self_end_index=[int(neighbor.self_end_index) for neighbor in neighbors],
        neighbor_start_index=[int(neighbor.neighbor_start_index) for neighbor in neighbors],
        neighbor_end_index=[int(neighbor.neighbor_end_index) for neighbor in neighbors],
        boundaries=nested_boundary_segment_list(neighbors),
    )


def lane_feature_data(lane) -> LaneFeatureData:
    return LaneFeatureData(
        speed_limit_mph=float(lane.speed_limit_mph),
        type=int(lane.type),
        interpolating=bool(lane.interpolating),
        polyline=vec3d_list(lane.polyline),
        entry_lanes=[int(lane_id) for lane_id in lane.entry_lanes],
        exit_lanes=[int(lane_id) for lane_id in lane.exit_lanes],
        left_boundaries=boundary_segment_list(lane.left_boundaries),
        right_boundaries=boundary_segment_list(lane.right_boundaries),
        left_neighbors=lane_neighbor_list(lane.left_neighbors),
        right_neighbors=lane_neighbor_list(lane.right_neighbors),
    )


def road_line_feature_data(road_line) -> RoadLineFeatureData:
    return RoadLineFeatureData(
        type=int(road_line.type),
        polyline=vec3d_list(road_line.polyline),
    )


def road_edge_feature_data(road_edge) -> RoadEdgeFeatureData:
    return RoadEdgeFeatureData(
        type=int(road_edge.type),
        polyline=vec3d_list(road_edge.polyline),
    )


def stop_sign_feature_data(stop_sign) -> StopSignFeatureData:
    return StopSignFeatureData(
        lane=[int(lane_id) for lane_id in stop_sign.lane],
        position=vec3d(stop_sign.position),
    )


def polygon_feature_data(polygon_points) -> PolygonFeatureData:
    return PolygonFeatureData(polygon=vec3d_list(polygon_points))


def feature_to_component(segment_context_name: str, feature) -> MapFeatureComponent:
    feature_type = feature.WhichOneof("feature_data") or "unknown"
    kwargs = {
        "key": MapFeatureKey(
            segment_context_name=segment_context_name,
            map_feature_id=int(feature.id),
        ),
        "feature_type": feature_type,
    }
    if feature_type == "lane":
        kwargs["lane"] = lane_feature_data(feature.lane)
    elif feature_type == "road_line":
        kwargs["road_line"] = road_line_feature_data(feature.road_line)
    elif feature_type == "road_edge":
        kwargs["road_edge"] = road_edge_feature_data(feature.road_edge)
    elif feature_type == "stop_sign":
        kwargs["stop_sign"] = stop_sign_feature_data(feature.stop_sign)
    elif feature_type == "crosswalk":
        kwargs["crosswalk"] = polygon_feature_data(feature.crosswalk.polygon)
    elif feature_type == "speed_bump":
        kwargs["speed_bump"] = polygon_feature_data(feature.speed_bump.polygon)
    elif feature_type == "driveway":
        kwargs["driveway"] = polygon_feature_data(feature.driveway.polygon)
    return MapFeatureComponent(**kwargs)


def component_to_row(map_feature: MapFeatureComponent) -> dict:
    row = map_feature.to_flatten_dict()
    row["index"] = f"{row['key.segment_context_name']};{row['key.map_feature_id']}"
    return row


def extract_map_feature_rows(segment_path: Path, compression_type: str):
    tf, open_dataset = import_waymo_runtime()
    dataset = tf.data.TFRecordDataset(
        [str(segment_path)],
        compression_type=tfrecord_compression(segment_path, compression_type),
        buffer_size=8 << 20,
        num_parallel_reads=1,
    )
    dataset = dataset.prefetch(1)

    fallback_segment_id = segment_id_from_path(segment_path)
    segment_context_name = fallback_segment_id
    frames_scanned = 0

    for record in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(record.numpy())
        frames_scanned += 1

        frame_context_name = getattr(frame.context, "name", "")
        if frame_context_name:
            segment_context_name = frame_context_name
        if not frame.map_features:
            continue

        rows = []
        feature_counts = Counter()
        for feature in frame.map_features:
            feature_type = feature.WhichOneof("feature_data") or "unknown"
            feature_counts[feature_type] += 1
            rows.append(component_to_row(feature_to_component(segment_context_name, feature)))
        return segment_context_name, rows, feature_counts, frames_scanned

    return segment_context_name, [], Counter(), frames_scanned


def write_rows(output_path: Path, rows: list[dict], compression: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=FULL_SCHEMA)
    temp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        pq.write_table(table, temp_path, compression=compression)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def process_segment(segment_path_str: str, output_dir_str: str, compression_type: str, parquet_codec: str, overwrite: bool):
    segment_path = Path(segment_path_str)
    derived_segment_id = segment_id_from_path(segment_path)
    default_output_path = Path(output_dir_str) / f"{derived_segment_id}.parquet"
    if default_output_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "segment_id": derived_segment_id,
            "segment_path": str(segment_path),
            "output_path": str(default_output_path),
            "feature_count": 0,
            "frames_scanned": 0,
            "type_counts": {},
            "elapsed_sec": 0.0,
        }

    started_at = time.perf_counter()
    try:
        segment_context_name, rows, feature_counts, frames_scanned = extract_map_feature_rows(
            segment_path=segment_path,
            compression_type=compression_type,
        )
        output_path = Path(output_dir_str) / f"{segment_context_name}.parquet"
        if output_path.exists() and not overwrite:
            return {
                "status": "skipped",
                "segment_id": segment_context_name,
                "segment_path": str(segment_path),
                "output_path": str(output_path),
                "feature_count": 0,
                "frames_scanned": frames_scanned,
                "type_counts": {},
                "elapsed_sec": time.perf_counter() - started_at,
            }

        write_rows(output_path, rows, parquet_codec)
        return {
            "status": "written",
            "segment_id": segment_context_name,
            "segment_path": str(segment_path),
            "output_path": str(output_path),
            "feature_count": len(rows),
            "frames_scanned": frames_scanned,
            "type_counts": dict(feature_counts),
            "elapsed_sec": time.perf_counter() - started_at,
        }
    except Exception as exc:
        return {
            "status": "error",
            "segment_id": derived_segment_id,
            "segment_path": str(segment_path),
            "output_path": str(default_output_path),
            "feature_count": 0,
            "frames_scanned": 0,
            "type_counts": {},
            "elapsed_sec": time.perf_counter() - started_at,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def print_summary(results: list[dict], output_dir: Path) -> None:
    status_counts = Counter(result["status"] for result in results)
    feature_type_totals = Counter()
    total_features = 0
    total_frames_scanned = 0
    empty_segments = 0

    for result in results:
        feature_type_totals.update(result.get("type_counts", {}))
        total_features += int(result.get("feature_count", 0))
        total_frames_scanned += int(result.get("frames_scanned", 0))
        if result.get("status") == "written" and result.get("feature_count", 0) == 0:
            empty_segments += 1

    print(f"Wrote {status_counts['written']} parquet files to {output_dir}")
    print(f"Skipped {status_counts['skipped']} existing files")
    print(f"Errors: {status_counts['error']}")
    print(f"Total extracted map features: {total_features}")
    print(f"Total TFRecord frames scanned: {total_frames_scanned}")
    if empty_segments:
        print(f"Segments with no map features found: {empty_segments}")
    if feature_type_totals:
        parts = [f"{feature_type}={count}" for feature_type, count in sorted(feature_type_totals.items())]
        print("Feature type totals: " + ", ".join(parts))

    error_results = [result for result in results if result["status"] == "error"]
    if error_results:
        print("\nFirst errors:")
        for result in error_results[:5]:
            print(f"- {result['segment_path']}: {result['error']}")


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    segment_paths = list_tfrecord_files(args.input_dir)
    if args.limit is not None:
        segment_paths = segment_paths[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    worker_args = (
        str(args.output_dir),
        args.compression_type,
        parquet_compression(args.parquet_compression),
        args.overwrite,
    )

    results = []
    if args.workers == 1:
        progress = tqdm(segment_paths, desc="extract map features", unit="segment")
        for segment_path in progress:
            result = process_segment(str(segment_path), *worker_args)
            results.append(result)
            progress.set_postfix(
                status=result["status"],
                features=result.get("feature_count", 0),
            )
    else:
        mp_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp_context) as executor:
            futures = [
                executor.submit(process_segment, str(segment_path), *worker_args)
                for segment_path in segment_paths
            ]
            progress = tqdm(as_completed(futures), total=len(futures), desc="extract map features", unit="segment")
            for future in progress:
                result = future.result()
                results.append(result)
                progress.set_postfix(
                    status=result["status"],
                    features=result.get("feature_count", 0),
                )

    print_summary(results, args.output_dir)
    return 1 if any(result["status"] == "error" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
