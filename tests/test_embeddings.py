import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.extract_embeddings import export_stages, offload_pooling_parent, prepare_cloud


class EmbeddingExportTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_offload_preserves_all_encoder_features(self):
        from litept.model import LitePT

        model = LitePT().eval().cuda()
        model.enc_mode = True
        grid = torch.arange(512, device="cuda")
        grid = torch.stack((grid // 64, grid // 8 % 8, grid % 8), dim=1)
        data = dict(coord=grid.float() * 0.05, grid_coord=grid,
                    feat=torch.randn(512, 4, device="cuda"),
                    offset=torch.tensor([512], device="cuda"))

        def snapshot():
            torch.manual_seed(23)
            with torch.inference_mode():
                point = model(data)
            result = []
            while True:
                result.append(point["feat"].cpu().clone())
                if "pooling_parent" not in point:
                    return result
                point = point["pooling_parent"]

        expected = snapshot()
        handles = [stage.down.register_forward_hook(offload_pooling_parent)
                   for stage in model.enc.children() if hasattr(stage, "down")]
        try:
            actual = snapshot()
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(len(actual), 5)
        for original, offloaded in zip(expected, actual):
            torch.testing.assert_close(original, offloaded)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_offload_keeps_current_stage_on_gpu_and_preserves_parent_data(self):
        parent = dict(
            feat=torch.randn(4, 36, device="cuda"),
            coord=torch.randn(4, 3, device="cuda"),
            pooling_inverse=torch.tensor([0, 1, 1, 2], device="cuda"),
            sparse_conv_feat=object(),
        )
        expected = parent["feat"].cpu().clone()
        output = dict(feat=torch.randn(3, 72, device="cuda"), pooling_parent=dict(parent))
        offload_pooling_parent(None, (parent,), output)
        self.assertIs(output["pooling_parent"], parent)
        self.assertFalse(parent["feat"].is_cuda)
        self.assertFalse(parent["pooling_inverse"].is_cuda)
        self.assertTrue(output["feat"].is_cuda)
        self.assertNotIn("sparse_conv_feat", parent)
        torch.testing.assert_close(parent["feat"], expected)

    def test_mapping_composes_voxelization_and_pooling(self):
        first = dict(feat=torch.arange(12).reshape(4, 3), coord=torch.zeros(4, 3))
        second = dict(
            feat=torch.tensor([[10., 20., 30.], [40., 50., 60.]]),
            coord=torch.ones(2, 3), pooling_parent=first,
            pooling_inverse=torch.tensor([1, 0, 1, 0]),
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            export_stages(second, [2, 0, 1, 3, 2], np.array([100., 200., 300.]), directory)
            with np.load(directory / "enc1.npz") as result:
                np.testing.assert_array_equal(result["input_to_stage"], [1, 1, 0, 0, 1])
                np.testing.assert_allclose(result["coord"], [[101., 201., 301.]] * 2)
                np.testing.assert_allclose(result["feat"][result["input_to_stage"]][2], [10., 20., 30.])

    def test_world_coordinates_shift_before_float32_and_voxels_map_back(self):
        # Keep points off exact voxel boundaries; floor() can move a boundary
        # point into an adjacent voxel after float32 rounding.
        coord = np.array([[2600000.02, 1200000., 500.],
                          [2600000.03, 1200000., 500.],
                          [2600000.20, 1200000., 500.]])
        point, origin = prepare_cloud(coord, np.array([0., 255., 127.]), 0.05, 255.)
        self.assertEqual(len(point["coord"]), 2)
        inverse = point["inverse"].numpy()
        self.assertEqual(inverse[0], inverse[1])
        self.assertNotEqual(inverse[0], inverse[2])
        self.assertGreater(origin[0], 2000000)
        self.assertLess(point["coord"].abs().max(), 1.)
        self.assertTrue(torch.all((point["feat"][:, 3] >= 0) & (point["feat"][:, 3] <= 1)))


if __name__ == "__main__":
    unittest.main()
