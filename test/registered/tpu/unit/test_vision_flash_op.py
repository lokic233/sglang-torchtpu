"""CPU-runnable unit tests for the TPU vision-encoder flash-attention wrapper.

These cover the pure-PyTorch logic in ``VisionTpuAttention.forward``
(``layers/attention/vision.py``): rearrange to ``[b, h, s, d]``, padding the
sequence up to ``_BLOCK_SIZE``, extending ``segment_ids`` with a dedicated
padding segment so pad tokens never attend to real ones, and slicing the output
back to the real length.

The actual pallas ``flash_attention`` kernel numerics (JAX + TPU) are out of
scope here and covered by the kernel tests; we substitute a pure-torch
segment-masked attention reference for ``_flash_op`` so this runs on CPU. The
contract verified is: the wrapper, with padding + a pad segment, returns exactly
the block-diagonal (per-segment) attention of the *unpadded* inputs.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

import math
import unittest
from unittest.mock import patch

import torch
from einops import rearrange

from sglang.srt.layers.attention.vision import VisionTpuAttention
from sglang.test.test_utils import CustomTestCase


def _ref_segment_attention(q, k, v, seg_ids, scale):
    """Pure-torch non-causal attention with block-diagonal (per-segment) mask.

    q, k, v: [b, h, s, d]; seg_ids: [b, s]. A query attends to a key iff they
    share a segment id. Mirrors the pallas SegmentIds semantics.
    """
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * scale
    same = seg_ids[:, None, :, None] == seg_ids[:, None, None, :]  # [b,1,s,s]
    scores = scores.masked_fill(~same, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", attn, v.float()).to(q.dtype)


class _CpuVisionAttn(VisionTpuAttention):
    """``VisionTpuAttention`` with the pallas op swapped for a CPU reference.

    The production ``__init__`` builds the pallas op (imports JAX + torch_tpu);
    we bypass it and inject ``_flash_op`` so ``forward`` runs on CPU.
    """

    def __init__(self, head_dim, num_heads, scale):
        torch.nn.Module.__init__(self)
        self.head_size = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.flatten_batch = True
        self.scale = scale
        self._flash_op = lambda q, k, v, seg: _ref_segment_attention(
            q, k, v, seg, scale
        )


class TestVisionFlashOpWrapper(CustomTestCase):
    def _run(self, seq_lens, h=2, d=16, seed=0):
        torch.manual_seed(seed)
        s = sum(seq_lens)
        scale = 1.0 / math.sqrt(d)
        q = torch.randn(s, h, d)
        k = torch.randn(s, h, d)
        v = torch.randn(s, h, d)
        seg = torch.cat(
            [torch.full((L,), i, dtype=torch.int32) for i, L in enumerate(seq_lens)]
        ).unsqueeze(0)

        attn = _CpuVisionAttn(d, h, scale)
        out = attn.forward(q, k, v, bsz=1, segment_ids=seg.clone())

        qr, kr, vr = (rearrange(x, "(b s) h d -> b h s d", b=1) for x in (q, k, v))
        ref = rearrange(
            _ref_segment_attention(qr, kr, vr, seg, scale), "b h s d -> (b s) h d"
        )
        return out, ref

    def test_single_segment_with_padding(self):
        # 100 -> padded to 128; the 28 pad tokens form their own segment.
        out, ref = self._run([100])
        self.assertEqual(tuple(out.shape), tuple(ref.shape))
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)

    def test_multi_segment_block_diagonal(self):
        # Three images packed in one sequence (block-diagonal masking).
        out, ref = self._run([30, 50, 20])
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)

    def test_exact_block_multiple_no_padding(self):
        # 128 is a multiple of _BLOCK_SIZE -> no padding branch.
        out, ref = self._run([128])
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)

    def test_pad_segment_does_not_leak(self):
        # Two unequal segments; correctness here means pad tokens (added to a
        # fresh segment) did not bleed into real tokens' softmax.
        out, ref = self._run([10, 90], h=4, d=32, seed=3)
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)


class TestVisionFlashTiling(CustomTestCase):
    """The vision op's JAX body, with the kernel swapped for the JAX reference:
    shapes on the single KV step call it unchanged; tiled shapes are padded to
    4,096, tiled, and sliced back to the per-segment attention of the real tokens."""

    def _run(self, seq_lens, tiles, h=2, d=16):
        import jax.numpy as jnp
        import numpy as np

        from sglang_tpu import attention_pallas
        from sglang_tpu.kernels.flash_attention import mha_reference

        calls = []

        def reference(q, k, v, *, segment_ids, causal, sm_scale, block_sizes, **_):
            calls.append((q.shape[2], block_sizes))
            return mha_reference(
                q, k, v, None, segment_ids=segment_ids, causal=causal, sm_scale=sm_scale
            )

        torch.manual_seed(0)
        s, scale = sum(seq_lens), 1.0 / math.sqrt(d)
        q, k, v = (torch.randn(1, h, s, d) for _ in range(3))
        seg = torch.cat(
            [torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(seq_lens)]
        ).unsqueeze(0)
        with patch.object(
            attention_pallas, "_vision_flash_tiles", lambda seq_len, heads: tiles
        ):
            out = attention_pallas._vision_flash(
                *(jnp.asarray(x.numpy()) for x in (q, k, v, seg)),
                sm_scale=scale,
                attention=reference,
            )
        ref = _ref_segment_attention(q, k, v, seg, scale)
        torch.testing.assert_close(
            torch.from_numpy(np.asarray(out)), ref, atol=1e-4, rtol=1e-4
        )
        return calls

    def test_single_step_shape_calls_kernel_unchanged(self):
        self.assertEqual(self._run([300, 212], tiles=False), [(512, None)])

    def test_tiled_shape_is_padded_tiled_and_sliced(self):
        from sglang_tpu.attention_pallas import _VISION_FLASH_TILED_BLOCKS

        # 1,100 tokens -> padded to 4,096 for the tiled blocks.
        calls = self._run([700, 400], tiles=True)
        self.assertEqual(calls, [(4096, _VISION_FLASH_TILED_BLOCKS)])

    def test_tiles_only_where_the_single_step_runs_out_of_vmem(self):
        from sglang_tpu.attention_pallas import _vision_flash_tiles

        for seq_len, heads, tiles in (
            (12288, 16, False),
            (15872, 8, False),
            (16384, 1, False),
            (20480, 1, False),
            (16256, 2, True),
            (16384, 8, True),
            (24576, 1, True),
        ):
            self.assertEqual(
                _vision_flash_tiles(seq_len, heads), tiles, (seq_len, heads)
            )


if __name__ == "__main__":
    unittest.main()
