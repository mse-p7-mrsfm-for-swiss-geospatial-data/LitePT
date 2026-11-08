"""Map encoder embeddings to original COPC points and export PCA-colored LAZ."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import laspy
import numpy as np
from laspy.vlrs.vlrlist import VLRList


def read_original_points(directory, source=None):
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    source = Path(source or metadata["input"])
    bounds = metadata.get("bounds")
    with laspy.CopcReader.open(str(source)) as reader:
        points = reader.query(bounds=None if bounds is None else laspy.Bounds(bounds[:3], bounds[3:]))
        if reader.header.evlrs is None:
            reader.header.read_evlrs(reader.source)
        header = deepcopy(reader.header)
    coord = np.load(directory / "input_coord.npy", mmap_mode="r")
    if coord.shape != (len(points), 3) or any(
        not np.array_equal(np.asarray(points[axis]), coord[:, i])
        for i, axis in enumerate(("x", "y", "z"))
    ):
        raise ValueError("COPC point order/coordinates do not match input_coord.npy. Check source and bounds.")
    return points, header, source


def embedding_fields(channels):
    # A VLR can hold only 65535 bytes; 504 scalar Extra Bytes definitions would
    # exceed that. LAS 1.4 array fields store up to three float32 components per
    # definition. PDAL expands them to separate attributes using suffixes 0..2.
    return [(f"emb_{start:03d}_{min(start + 2, channels - 1):03d}", start, min(3, channels - start))
            for start in range(0, channels, 3)]


def output_header(source, channels, pca_only=False, rgb_only=False):
    header = deepcopy(source)
    header.vlrs = VLRList(v for v in header.vlrs if v.user_id.lower() != "copc")
    header.evlrs = VLRList(v for v in (header.evlrs or []) if v.user_id.lower() != "copc")
    if "red" not in header.point_format.dimension_names:
        fmt = laspy.PointFormat(7)
        fmt.dimensions.extend(deepcopy(list(header.point_format.extra_dimensions)))
        header.set_version_and_point_format(laspy.header.Version(1, 4), fmt)
    if rgb_only:
        # QGIS 3.44 / OSGeo4W has a reproducible PDAL Extra Bytes import bug.
        # Standard RGB fields remain directly readable; keep a separate full
        # attribute file so the visualization workaround loses no embeddings.
        header.remove_extra_dims(list(header.point_format.extra_dimension_names))
        return header, []
    fields = [] if pca_only else embedding_fields(channels)
    dimensions = [laspy.ExtraBytesParams(name="litept_index", type=np.uint64,
                                       description="Row in input_coord.npy"),
                  laspy.ExtraBytesParams(name="litept_voxel", type=np.uint64,
                                       description="Row in encoder feat array")]
    dimensions += [laspy.ExtraBytesParams(name=f"pca_{i + 1}", type=np.float32)
                   for i in range(3)]
    dimensions += [laspy.ExtraBytesParams(
        name=name, type=np.dtype((np.float32, (count,))) if count > 1 else np.float32,
        description=f"Encoder channels {start}..{start + count - 1}",
    ) for name, start, count in fields]
    names = set(header.point_format.dimension_names)
    if names.intersection(d.name for d in dimensions):
        raise ValueError("Source already has attributes reserved by the embedding exporter.")
    header.add_extra_dims(dimensions)
    if any(len(v.record_data_bytes()) > 65535 for v in header.vlrs):
        raise ValueError("LAS VLR metadata exceeds 65535 bytes; try --pca-only.")
    return header, fields


def pca_colors(features, cached_scores=None, percentiles=(2., 98.)):
    if cached_scores is not None and cached_scores.ndim == 2 and cached_scores.shape[1] >= 3:
        scores = np.asarray(cached_scores[:, :3], dtype=np.float32)
        origin = "cached NPZ PCA"
    else:
        from sklearn.decomposition import PCA

        count = min(3, *features.shape)
        scores = np.zeros((len(features), 3), dtype=np.float32)
        scores[:, :count] = PCA(n_components=count, random_state=0).fit_transform(features)
        origin = "PCA fitted on native stage features"
    if scores.shape != (len(features), 3) or not np.isfinite(scores).all():
        raise ValueError("PCA scores must be finite and match the native stage points.")
    low, high = np.percentile(scores, percentiles, axis=0)
    span = high - low
    span[span == 0] = 1.
    rgb = np.rint(np.clip((scores - low) / span, 0, 1) * 65535).astype(np.uint16)
    return scores, rgb, dict(source=origin, percentiles=list(percentiles),
                             rgb_low=low.tolist(), rgb_high=high.tolist())


def write_stage(npz_path, destination, source_points, source_header, chunk_size=10000,
                pca_only=False, percentiles=(2., 98.), rgb_only=False):
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(destination.stem + ".partial.laz")
    if temporary.exists():
        raise FileExistsError(temporary)
    with np.load(npz_path, allow_pickle=False) as stage:
        features = stage["feat"]
        mapping = stage["input_to_stage"]
        if features.ndim != 2 or min(features.shape) == 0 or not np.isfinite(features).all():
            raise ValueError(f"Invalid feature array in {npz_path}")
        if (mapping.shape != (len(source_points),) or mapping.dtype.kind not in "iu"
                or mapping.min() < 0 or mapping.max() >= len(features)):
            raise ValueError(f"Invalid input_to_stage mapping in {npz_path}")
        scores, rgb, color_info = pca_colors(features, stage.get("pca"), percentiles)
    header, fields = output_header(source_header, features.shape[1], pca_only, rgb_only)
    available = set(header.point_format.dimension_names)
    source_dimensions = [n for n in source_points.point_format.dimension_names
                         if n in available and n not in ("red", "green", "blue")]
    try:
        with laspy.open(temporary, mode="w", header=header, do_compress=True) as writer:
            for start in range(0, len(source_points), chunk_size):
                stop = min(start + chunk_size, len(source_points))
                indices = mapping[start:stop]
                points = laspy.ScaleAwarePointRecord.zeros(stop - start, header=header)
                original = source_points[start:stop]
                for name in source_dimensions:
                    points[name] = original[name]
                if not rgb_only:
                    points["litept_index"] = np.arange(start, stop, dtype=np.uint64)
                    points["litept_voxel"] = indices
                for i, name in enumerate(("red", "green", "blue")):
                    points[name] = rgb[indices, i]
                    if not rgb_only:
                        points[f"pca_{i + 1}"] = scores[indices, i]
                for name, channel, count in fields:
                    values = features[indices, channel:channel + count]
                    points[name] = values if count > 1 else values[:, 0]
                writer.write_points(points)
            if header.evlrs:
                writer.write_evlrs(header.evlrs)
        temporary.rename(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return dict(
        file=destination.name, points=len(source_points), channels=features.shape[1],
        embedding_attributes=not (pca_only or rgb_only), pca=color_info,
        pca_attributes=not rgb_only, rgb_preview=rgb_only,
        embedding_fields=[dict(las_name=name, first_channel=start, components=count,
                               pdal_names=[name + str(i) for i in range(count)] if count > 1 else [name])
                          for name, start, count in fields],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Directory containing enc*.npz and metadata.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path, help="Override COPC path from metadata if the file was moved")
    parser.add_argument("--stages", nargs="+", help="Optional stage names, e.g. enc0 enc4")
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--pca-only", action="store_true", help="Only write QGIS RGB preview files, without Extra Bytes attributes")
    parser.add_argument("--color-percentiles", type=float, nargs=2, default=(2., 98.))
    args = parser.parse_args()
    if args.chunk_size < 1 or not 0 <= args.color_percentiles[0] < args.color_percentiles[1] <= 100:
        parser.error("Chunk size must be positive and color percentiles satisfy 0 <= low < high <= 100.")
    files = sorted(args.input.glob("enc[0-9]*.npz"))
    if args.stages:
        available = {p.stem for p in files}
        if set(args.stages) - available:
            parser.error(f"Unknown stages: {set(args.stages) - available}")
        files = [p for p in files if p.stem in args.stages]
    if not files:
        parser.error("No encoder NPZ files found.")
    targets = [args.output / (p.stem + "_pca.laz") for p in files]
    if not args.pca_only:
        targets += [args.output / (p.stem + ".laz") for p in files]
    if any(p.exists() or p.with_name(p.stem + ".partial.laz").exists() for p in targets) or (args.output / "manifest.json").exists():
        parser.error("Output files already exist; select another output directory.")
    points, header, source = read_original_points(args.input, args.source)
    print(f"Verified {len(points):,} original points against input_coord.npy", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    stages = []
    for path in files:
        info = None
        if not args.pca_only:
            target = args.output / (path.stem + ".laz")
            print(f"Writing {target} ...", flush=True)
            info = write_stage(path, target, points, header, args.chunk_size,
                               percentiles=args.color_percentiles)
            print(f"Finished {target} ({target.stat().st_size / 1e6:.1f} MB)", flush=True)
        target = args.output / (path.stem + "_pca.laz")
        print(f"Writing QGIS preview {target} ...", flush=True)
        preview = write_stage(path, target, points, header, args.chunk_size,
                              percentiles=args.color_percentiles, rgb_only=True)
        info = info or preview
        info["qgis_preview"] = target.name
        stages.append(info)
        print(f"Finished {target} ({target.stat().st_size / 1e6:.1f} MB)", flush=True)
    manifest = dict(source=str(source.resolve()), embeddings=str(args.input.resolve()),
                    points=len(points), format="LAS 1.4 / LAZ, not COPC",
                    colors="PCA1/2/3 -> red/green/blue; independently normalized per stage",
                    stages=stages)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
