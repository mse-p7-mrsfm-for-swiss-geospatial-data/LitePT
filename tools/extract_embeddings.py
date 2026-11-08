"""Export NuScenes LitePT-S encoder embeddings from a local COPC file."""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


def read_copc(path, bounds=None, max_points=200000):
    from laspy import Bounds, CopcReader

    if bounds is not None:
        bounds = np.asarray(bounds, dtype=np.float64)
        if not np.isfinite(bounds).all() or np.any(bounds[:3] >= bounds[3:]):
            raise ValueError("Bounds must have finite min values smaller than max values.")
    with CopcReader.open(str(path)) as reader:
        if bounds is None and reader.header.point_count > max_points:
            raise ValueError(
                f"File contains {reader.header.point_count} points. Select a crop with "
                "--bounds xmin ymin zmin xmax ymax zmax, or increase --max-points."
            )
        points = reader.query(
            bounds=None if bounds is None else Bounds(bounds[:3], bounds[3:])
        )
        if not 0 < len(points) <= max_points:
            raise ValueError(f"Crop has {len(points)} points; expected 1..{max_points}.")
        coord = np.column_stack((points.x, points.y, points.z)).astype(np.float64)
        intensity = np.asarray(points.intensity, dtype=np.float32).copy()
    return coord, intensity


def prepare_cloud(coord, intensity, grid_size, intensity_scale):
    from datasets.transform import Compose

    # Shift in float64 before conversion: Swiss/projected CRS coordinates lose
    # centimeter precision if converted directly to float32.
    origin = coord.min(axis=0)
    origin[:2] = (coord.min(axis=0)[:2] + coord.max(axis=0)[:2]) / 2
    local = (coord - origin).astype(np.float32)
    transform = Compose([
        dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train",
             return_grid_coord=True, return_inverse=True),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "inverse"),
             feat_keys=("coord", "strength")),
    ])
    point = transform(dict(coord=local, strength=intensity[:, None] / intensity_scale))
    return point, origin


def export_stages(output, input_inverse, origin, directory, pca_components=0, seed=0):
    """Map input points through each pooling level without expanding embeddings."""
    stages = [output]
    while "pooling_parent" in stages[-1]:
        stages.append(stages[-1]["pooling_parent"])
    stages.reverse()
    mapping = np.asarray(input_inverse, dtype=np.int64)
    summary = []
    for i, stage in enumerate(stages):
        if i:
            parent_to_child = stage["pooling_inverse"].detach().cpu().numpy()
            mapping = parent_to_child[mapping]
        feat = stage["feat"].detach().float().cpu().numpy()
        coord = stage["coord"].detach().float().cpu().numpy().astype(np.float64) + origin
        arrays = dict(feat=feat, coord=coord, input_to_stage=mapping.copy())
        if pca_components:
            from sklearn.decomposition import PCA

            if pca_components > min(feat.shape):
                raise ValueError(f"enc{i} is too small for {pca_components} PCA components.")
            pca = PCA(n_components=pca_components, random_state=seed)
            arrays.update(
                pca=pca.fit_transform(feat),
                pca_components=pca.components_,
                pca_mean=pca.mean_,
                pca_explained_variance_ratio=pca.explained_variance_ratio_,
            )
        np.savez(directory / f"enc{i}.npz", **arrays)
        summary.append(dict(name=f"enc{i}", points=len(feat), channels=feat.shape[1]))
        print(f"enc{i}: {feat.shape} -> {directory / f'enc{i}.npz'}", flush=True)
    return summary


def offload_pooling_parent(module, inputs, output):
    """Completed encoder stages are needed for export, not further inference."""
    # Point(dict) copies nested dictionaries, so the saved parent can differ
    # from the actual input still retained by outer forward calls. Offload that
    # actual input too, and reuse it as the export parent instead of a copy.
    parent = inputs[0] if inputs else output["pooling_parent"]
    output["pooling_parent"] = parent
    # SparseConvTensor retains large indice caches. The decoder would need it,
    # but encoder-only export needs only the parent features and point mappings.
    parent.pop("sparse_conv_feat", None)
    for key, value in list(parent.items()):
        if isinstance(value, torch.Tensor):
            parent[key] = value.detach().cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Local trusted NuScenes LitePT-S checkpoint")
    parser.add_argument("--bounds", nargs=6, type=float, metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"))
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument("--grid-size", type=float, default=0.05, help="Voxel size in meters")
    parser.add_argument("--intensity-scale", type=float, default=65535., help="Divisor for raw LAS intensity; use 255 for 8-bit data")
    parser.add_argument("--pca-components", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.grid_size <= 0 or args.intensity_scale <= 0 or args.max_points <= 0 or args.pca_components < 0:
        parser.error("Sizes/scales must be positive and PCA components nonnegative.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty (avoid mixing exports).")
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required.")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    coord, intensity = read_copc(args.input, args.bounds, args.max_points)
    print(f"Read {len(coord)} points; intensity range {intensity.min()}..{intensity.max()}", flush=True)
    point, origin = prepare_cloud(coord, intensity, args.grid_size, args.intensity_scale)
    inverse = point["inverse"].numpy().copy()

    from litept.model import LitePT

    checkpoint = args.checkpoint
    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download(
            repo_id="prs-eth/LitePT",
            filename="nuscenes-semseg-litept-small-v1m1/model/model_best.pth",
            cache_dir=Path(__file__).resolve().parents[1] / ".cache/huggingface/hub",
        )
    # Official training checkpoints contain objects beyond tensors.
    # A custom checkpoint passed here must also be from a trusted source.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    prefix = "module.backbone."
    weights = OrderedDict((k[len(prefix):], v) for k, v in state.items() if k.startswith(prefix))
    model = LitePT()
    model.load_state_dict(weights, strict=True)
    # Keep all encoder parent features intact; the decoder mutates those parents.
    model.enc_mode = True
    model.eval().cuda()
    # A pooling output no longer reads its parent. Preserve the export chain on
    # CPU instead of retaining every stage's features/indice caches on the GPU.
    for stage in model.enc.children():
        if hasattr(stage, "down"):
            stage.down.register_forward_hook(offload_pooling_parent)
    point = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in point.items()}
    with torch.inference_mode():
        output = model(point)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "input_coord.npy", coord)
    stages = export_stages(output, inverse, origin, args.output, args.pca_components, args.seed)
    metadata = dict(
        input=str(args.input.resolve()), checkpoint=str(checkpoint), bounds=args.bounds,
        grid_size=args.grid_size, intensity_scale=args.intensity_scale,
        origin=origin.tolist(), seed=args.seed, stages=stages,
        mapping="input_to_stage indexes rows of feat for rows of input_coord.npy",
        pca="Fitted independently on native points of each encoder stage; no feature standardization",
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
