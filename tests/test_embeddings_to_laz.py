from pathlib import Path
import tempfile
import unittest

import laspy
import numpy as np
from laspy.vlrs.known import WktCoordinateSystemVlr

from tools.embeddings_to_laz import pca_colors, write_stage


class LazExportTest(unittest.TestCase):
    def source(self):
        header = laspy.LasHeader(point_format=6, version="1.4")
        header.scales = [0.01] * 3
        header.offsets = [2664000., 1254000., 0.]
        header.add_extra_dim(laspy.ExtraBytesParams(name="echo_width", type=np.uint16))
        header.vlrs.append(laspy.VLR(user_id="copc", record_id=1, record_data=b"not copied"))
        header.vlrs.append(WktCoordinateSystemVlr('LOCAL_CS["test",UNIT["metre",1]]'))
        header.global_encoding.wkt = True
        points = laspy.ScaleAwarePointRecord.zeros(5, header=header)
        points.X = [0, 3, 5, 7, 9]
        points.Y = [2, 1, 6, 8, 10]
        points.Z = [40000, 40020, 40060, 40100, 40120]
        points.intensity = [120, 220, 320, 420, 520]
        points.classification = [2, 6, 5, 2, 6]
        points.return_number = [1, 2, 1, 3, 2]
        points.number_of_returns = [2, 2, 1, 3, 2]
        points.overlap = [1, 0, 0, 1, 0]
        points.echo_width = [10, 20, 30, 40, 50]
        return points, header

    def test_504_channels_roundtrip_preserves_points_attributes_and_mapping(self):
        original, header = self.source()
        features = np.arange(4 * 504, dtype=np.float32).reshape(4, 504) / 17
        mapping = np.array([2, 0, 2, 3, 1], dtype=np.int64)
        scores = np.array([[0., -2., 1.], [1., 0., 2.], [2., 3., -1.], [4., 4., 0.]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "enc4.npz"
            target = Path(temporary) / "enc4.laz"
            np.savez(source, feat=features, input_to_stage=mapping, pca=scores)
            info = write_stage(source, target, original, header, chunk_size=2, percentiles=(0, 100))
            loaded = laspy.read(target)
            self.assertEqual(len(loaded.points), 5)
            self.assertEqual(loaded.header.point_format.id, 7)
            np.testing.assert_array_equal(loaded.header.scales, header.scales)
            np.testing.assert_array_equal(loaded.header.offsets, header.offsets)
            for name in original.point_format.dimension_names:
                np.testing.assert_array_equal(loaded[name], original[name])
            np.testing.assert_array_equal(loaded.litept_index, np.arange(5))
            np.testing.assert_array_equal(loaded.litept_voxel, mapping)
            for field in info["embedding_fields"]:
                start = field["first_channel"]
                count = field["components"]
                np.testing.assert_array_equal(loaded[field["las_name"]], features[mapping, start:start + count])
            expected_rgb = pca_colors(features, scores, (0, 100))[1]
            for i, name in enumerate(("red", "green", "blue")):
                np.testing.assert_array_equal(loaded[name], expected_rgb[mapping, i])
                np.testing.assert_array_equal(loaded[f"pca_{i + 1}"], scores[mapping, i])
            self.assertFalse(any(v.user_id == "copc" for v in loaded.header.vlrs))
            self.assertEqual(loaded.header.vlrs.get("WktCoordinateSystemVlr")[0].string,
                             header.vlrs.get("WktCoordinateSystemVlr")[0].string)
            self.assertLess(len(loaded.header.vlrs.get("ExtraBytesVlr")[0].record_data_bytes()), 65536)

    def test_pca_only_and_invalid_mapping(self):
        points, header = self.source()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "enc0.npz"
            np.savez(source, feat=np.eye(4, dtype=np.float32),
                     input_to_stage=np.array([0, 1, 2, 3, 0]), pca=np.ones((4, 3), dtype=np.float32))
            target = directory / "enc0.laz"
            write_stage(source, target, points, header, pca_only=True)
            loaded = laspy.read(target)
            self.assertFalse(any(n.startswith("emb_") for n in loaded.point_format.dimension_names))
            np.testing.assert_array_equal(loaded.red, 0)
            with self.assertRaises(FileExistsError):
                write_stage(source, target, points, header)
            np.savez(source, feat=np.eye(4, dtype=np.float32), input_to_stage=np.array([0, 1, 2, 3, 99]))
            with self.assertRaisesRegex(ValueError, "mapping"):
                write_stage(source, directory / "bad.laz", points, header)
            self.assertFalse((directory / "bad.laz").exists())

    def test_rgb_preview_has_no_extra_bytes_and_same_original_coordinates(self):
        points, header = self.source()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez(root / "enc0.npz", feat=np.eye(4, dtype=np.float32),
                     input_to_stage=np.array([0, 1, 2, 3, 0]),
                     pca=np.arange(12, dtype=np.float32).reshape(4, 3))
            write_stage(root / "enc0.npz", root / "preview.laz", points, header,
                        chunk_size=2, rgb_only=True)
            loaded = laspy.read(root / "preview.laz")
            self.assertEqual(loaded.header.point_format.num_extra_bytes, 0)
            self.assertEqual(loaded.header.vlrs.get("ExtraBytesVlr"), [])
            for axis in ("X", "Y", "Z", "intensity", "classification"):
                np.testing.assert_array_equal(loaded[axis], points[axis])
            self.assertGreater(loaded.red.max(), 0)


if __name__ == "__main__":
    unittest.main()
