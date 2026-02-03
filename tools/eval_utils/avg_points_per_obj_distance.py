import argparse
import pickle
import numpy as np
from pathlib import Path


def load_pkl_local(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def iter_original_infos(pkl_obj):
    if isinstance(pkl_obj, dict):
        yield pkl_obj
        return
    if not isinstance(pkl_obj, (list, tuple)):
        raise TypeError(f"Unexpected PKL root type: {type(pkl_obj)}")

    for i, sample in enumerate(pkl_obj):
        if isinstance(sample, dict):
            yield sample
            continue
        if isinstance(sample, (list, tuple)) and len(sample) > 0 and isinstance(sample[0], dict):
            yield sample[0]  # original
            continue
        raise TypeError(f"Unexpected sample format at index {i}: {type(sample)}")


def _as_boxes7(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if boxes.size == 0:
        return boxes.reshape(0, 7).astype(np.float32)

    if boxes.shape[1] < 7:
        raise ValueError(f"gt_boxes must have >=7 dims, got shape {boxes.shape}")

    return boxes[:, :7].astype(np.float32, copy=False)


def _count_points_in_boxes_cpu(points_xyz: np.ndarray, boxes7: np.ndarray) -> np.ndarray:
    if boxes7.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if points_xyz.shape[0] == 0:
        return np.zeros((boxes7.shape[0],), dtype=np.int64)

    try:
        import torch
        from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    except Exception as e:
        raise RuntimeError(
            "Failed to import OpenPCDet roiaware_pool3d (needed to count points in boxes). "
            "Make sure OpenPCDet ops are built and torch is available."
        ) from e

    pts = torch.from_numpy(points_xyz[:, :3].astype(np.float32, copy=False))
    boxes = torch.from_numpy(boxes7.astype(np.float32, copy=False))
    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(pts, boxes).numpy()  # (Nboxes, Npoints)
    return point_indices.sum(axis=1).astype(np.int64)


def _extract_boxes_and_num_points(info: dict):
    # Case A: classic infos with annos['num_points_in_gt']
    annos = info.get("annos", None)
    if isinstance(annos, dict):
        boxes = annos.get("gt_boxes_lidar", None)
        if boxes is None:
            boxes = annos.get("gt_boxes", None)
        num_pts = annos.get("num_points_in_gt", None)
        if boxes is not None and num_pts is not None:
            boxes7 = _as_boxes7(boxes)
            num_pts = np.asarray(num_pts).reshape(-1)[: boxes7.shape[0]]
            names = annos.get("name", None)
            if names is not None:
                names = np.asarray(names).reshape(-1)[: boxes7.shape[0]]
            index = annos.get("index", None)
            if index is not None:
                index = np.asarray(index).reshape(-1)[: boxes7.shape[0]]
            return boxes7, num_pts, names, index, "annos.num_points_in_gt"

    # Case B: your train_dataset.pkl style: gt_boxes + points
    boxes = info.get("gt_boxes", None)
    if boxes is None:
        boxes = info.get("gt_boxes_lidar", None)
    points = info.get("points", None)
    if boxes is None or points is None:
        return None

    boxes7 = _as_boxes7(boxes)
    points = np.asarray(points)
    points_xyz = points[:, :3].astype(np.float32, copy=False)

    num_pts = _count_points_in_boxes_cpu(points_xyz, boxes7)

    names = info.get("gt_names", None)  # in your sample
    if names is not None:
        names = np.asarray(names).reshape(-1)[: boxes7.shape[0]]

    # no "index" in this format; treat all as valid
    return boxes7, num_pts, names, None, "computed_from_points"


def compute_avg_points_per_object_over_distance(
    pkl_path: str,
    max_dist: float = 60.0,
    bin_size: float = 1.0,
    distance_mode: str = "x",
    enforce_forward_range: bool = True
):
    infos_raw = load_pkl_local(pkl_path)
    n_bins = int(np.ceil(max_dist / bin_size))
    bin_edges = np.arange(0.0, n_bins * bin_size + 1e-9, bin_size, dtype=np.float32)

    points_per_bin = [[] for _ in range(n_bins)]

    stats = {
        "pkl_path": str(pkl_path),
        "max_dist": float(max_dist),
        "bin_size": float(bin_size),
        "distance_mode": str(distance_mode),
        "enforce_forward_range": bool(enforce_forward_range),
        "n_samples": 0,
        "n_total_objs_seen": 0,
        "n_total_objs_used": 0,
    }

    for info in iter_original_infos(infos_raw):
        stats["n_samples"] += 1
        extracted = _extract_boxes_and_num_points(info)
        if extracted is None:
            continue
        boxes7, num_pts, names, index, source = extracted
        if boxes7.shape[0] == 0:
            continue

        stats["n_total_objs_seen"] += int(boxes7.shape[0])

        x = boxes7[:, 0]
        y = boxes7[:, 1]
        z = boxes7[:, 2]
        if distance_mode == "x":
            dist = x
        elif distance_mode == "xy":
            dist = np.sqrt(x * x + y * y)
        elif distance_mode == "xyz":
            dist = np.sqrt(x * x + y * y + z * z)
        else:
            raise ValueError("distance_mode must be one of: x, xy, xyz")

        mask = np.ones(boxes7.shape[0], dtype=bool)
        if index is not None:
            mask &= index >= 0
        if names is not None:
            mask &= (names != "DontCare")
        mask &= (num_pts >= 0)
        mask &= (dist >= 0) & (dist <= max_dist)
        if enforce_forward_range:
            mask &= (x >= 0) & (x <= max_dist)
        if not np.any(mask):
            continue

        dist_sel = dist[mask]
        pts_sel = np.asarray(num_pts[mask], dtype=np.float64)
        bin_idx = np.floor(dist_sel / bin_size).astype(np.int64)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        for b, pts in zip(bin_idx, pts_sel):
            points_per_bin[b].append(pts)
        stats["n_total_objs_used"] += int(mask.sum())

    avg_points = np.array([np.mean(bin) if bin else np.nan for bin in points_per_bin])
    std_points = np.array([np.std(bin) if bin else np.nan for bin in points_per_bin])
    obj_counts = np.array([len(bin) for bin in points_per_bin])

    return bin_edges, avg_points, std_points, obj_counts, stats

def save_csv(csv_path: str, bin_edges, avg_points, std_points, obj_counts):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("bin_start,bin_end,avg_points_per_object,std_points_per_object,num_objects\n")
        for i in range(len(avg_points)):
            avg = avg_points[i]
            std = std_points[i]
            count = obj_counts[i]
            avg_str = "" if not np.isfinite(avg) else f"{avg:.2f}"
            std_str = "" if not np.isfinite(std) else f"{std:.2f}"
            f.write(f"{bin_edges[i]:.1f},{bin_edges[i+1]:.1f},{avg_str},{std_str},{count}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--max-dist", type=float, default=60.0)
    ap.add_argument("--bin-size", type=float, default=1.0)
    ap.add_argument("--distance-mode", type=str, default="x", choices=["x", "xy", "xyz"])
    ap.add_argument("--no-enforce-forward-range", action="store_true")
    ap.add_argument("--out-csv", type=str, default=None)
    args = ap.parse_args()

    bin_edges, avg_points, std_points, obj_counts, stats = compute_avg_points_per_object_over_distance(
        pkl_path=args.pkl,
        max_dist=args.max_dist,
        bin_size=args.bin_size,
        distance_mode=args.distance_mode,
        enforce_forward_range=not args.no_enforce_forward_range,
    )

    print("\n=== Avg points per object over distance ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    if stats["n_total_objs_used"] == 0:
        print(
            "\nWARNING: No objects were counted. "
            "Most likely the PKL doesn't contain points/gt_boxes in the expected format "
            "or roiaware_pool3d is not available."
        )

    print("\nbin_start  bin_end  num_objects  avg_points_per_object")
    for i in range(len(avg_points)):
        a = bin_edges[i]
        b = bin_edges[i + 1]
        c = int(obj_counts[i])
        m = avg_points[i]
        m_str = f"{m:.3f}" if np.isfinite(m) else "nan"
        print(f"{a:8.2f}  {b:7.2f}  {c:11d}  {m_str}")

    if args.out_csv:
        save_csv(args.out_csv, bin_edges, avg_points, std_points, obj_counts)
        print(f"Saved CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()