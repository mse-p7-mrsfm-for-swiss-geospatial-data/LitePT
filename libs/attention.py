"""Variable-length attention with an optional FlashAttention acceleration."""

import torch
from torch.nn.functional import pad, scaled_dot_product_attention

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func as _flash_attention
except (ImportError, OSError):
    _flash_attention = None


def varlen_attention(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None):
    if _flash_attention is not None and qkv.is_cuda:
        return _flash_attention(
            qkv, cu_seqlens, max_seqlen,
            dropout_p=dropout_p, softmax_scale=softmax_scale,
        )

    # LitePT uses head_dim=18. Efficient CUDA SDPA kernels require multiples
    # of eight; zero padding preserves QK dot products and the original scale.
    head_dim = qkv.shape[-1]
    scale = softmax_scale if softmax_scale is not None else head_dim ** -0.5
    if head_dim % 8:
        qkv = pad(qkv, (0, -head_dim % 8))

    # Each patch attends only to itself. Group consecutive patches of equal
    # length into a batch, avoiding a separate kernel launch for every patch.
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu()
    sizes, counts = torch.unique_consecutive(lengths, return_counts=True)
    outputs = []
    start = 0
    for size, count in zip(sizes.tolist(), counts.tolist()):
        end = start + size * count
        q, k, v = qkv[start:end].reshape(count, size, 3, *qkv.shape[2:]).permute(
            2, 0, 3, 1, 4
        ).unbind(0)
        output = scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, scale=scale
        )
        outputs.append(output.transpose(1, 2).reshape(end - start, *qkv.shape[2:])[..., :head_dim])
        start = end
    return torch.cat(outputs, dim=0)
