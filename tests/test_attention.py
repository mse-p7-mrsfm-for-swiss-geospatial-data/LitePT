import unittest
from unittest.mock import patch

import torch

from libs.attention import varlen_attention


class AttentionTest(unittest.TestCase):
    def test_default_scale_uses_original_head_dimension(self):
        qkv = torch.randn(4, 3, 2, 6)
        with patch("libs.attention._flash_attention", None):
            result = varlen_attention(qkv, torch.tensor([0, 4]), 4)
        q, k, v = qkv.permute(1, 2, 0, 3).unbind(0)
        expected = ((q @ k.transpose(-2, -1) / 6**0.5).softmax(-1) @ v).transpose(0, 1)
        torch.testing.assert_close(result, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_head_dim_18_supports_memory_efficient_cuda_kernel(self):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        qkv = torch.randn(8, 3, 2, 18, device="cuda", dtype=torch.float16)
        offsets = torch.tensor([0, 4, 8], device="cuda", dtype=torch.int32)
        with patch("libs.attention._flash_attention", None):
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                result = varlen_attention(qkv, offsets, 4)
        q, k, v = qkv.float().reshape(2, 4, 3, 2, 18).permute(2, 0, 3, 1, 4).unbind(0)
        expected = ((q @ k.transpose(-2, -1) / 18**0.5).softmax(-1) @ v)
        expected = expected.transpose(1, 2).reshape(8, 2, 18)
        torch.testing.assert_close(result.float(), expected, atol=2e-3, rtol=2e-3)

    def test_variable_lengths_and_gradients(self):
        torch.manual_seed(7)
        qkv = torch.randn(7, 3, 2, 6, requires_grad=True)
        offsets = torch.tensor([0, 4, 7], dtype=torch.int32)
        scale = 0.3
        with patch("libs.attention._flash_attention", None):
            result = varlen_attention(qkv, offsets, 4, softmax_scale=scale)
        expected = []
        for start, end in [(0, 4), (4, 7)]:
            q, k, v = qkv[start:end].permute(1, 2, 0, 3).unbind(0)
            weights = (q @ k.transpose(-2, -1) * scale).softmax(-1)
            expected.append((weights @ v).transpose(0, 1))
        expected = torch.cat(expected)
        torch.testing.assert_close(result, expected)
        expected_grad = torch.autograd.grad(expected.square().sum(), qkv)[0]
        result.square().sum().backward()
        torch.testing.assert_close(qkv.grad, expected_grad)

    def test_sequences_are_independent(self):
        qkv = torch.randn(8, 3, 2, 6)
        offsets = torch.tensor([0, 4, 8], dtype=torch.int32)
        with patch("libs.attention._flash_attention", None):
            original = varlen_attention(qkv, offsets, 4)
            qkv[4:] += 10
            changed = varlen_attention(qkv, offsets, 4)
        torch.testing.assert_close(original[:4], changed[:4])


if __name__ == "__main__":
    unittest.main()
