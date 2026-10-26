"""Tests for segment-local vision attention over gathered rows (``VisionTpuRowsAttention``).

* On CPU, ``forward(rows=...)`` with the Pallas op swapped for a torch reference:
  the output equals the per-segment attention of the token-order inputs, and bad
  layouts are rejected. The op's JAX body runs with the JAX reference attention,
  covering the gather, head transposes and inverse gather.
* On the TPU in bf16, the registered op through ``VisionTpuRowsAttention``, eager
  and compiled, against an fp32 CPU reference: rows of 256 slots shared by
  several segments, segments interleaved in token order, pad slots, an all-pad row.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=60,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "infra", "multimodal", "torch-compile"],
)

import math
import unittest

import torch
from einops import rearrange

from sglang.srt.utils import is_tpu
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.vision_rows import VisionAttentionRows, VisionTpuRowsAttention


def _ref_segment_attention(q, k, v, seg_ids, scale):
    """Non-causal attention within a segment id, in fp32. q, k, v: [b, h, s, d]; seg_ids: [b, s]."""
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * scale
    same = seg_ids[:, None, :, None] == seg_ids[:, None, None, :]
    scores = scores.masked_fill(~same, float("-inf"))
    return torch.einsum("bhqk,bhkd->bhqd", torch.softmax(scores, dim=-1), v.float())


def _ref_token_attention(q, k, v, seg, scale):
    """Per-segment attention of token-order q, k, v [s, h, d] with segments seg [s]."""
    qr, kr, vr = (rearrange(x, "s h d -> 1 h s d") for x in (q, k, v))
    out = _ref_segment_attention(qr, kr, vr, seg[None], scale)
    return rearrange(out, "1 h s d -> s h d")


def _ref_rows_attention(q, k, v, row_index, row_segment_ids, inverse_index, scale):
    """Torch reference for the rows op: gather, per-row segment attention, gather back."""
    num_rows = row_segment_ids.shape[0]
    rows = [
        rearrange(
            x[row_index.long().clamp(0, x.shape[0] - 1)],
            "(r l) h d -> r h l d",
            r=num_rows,
        )
        for x in (q, k, v)
    ]
    out = _ref_segment_attention(*rows, row_segment_ids, scale).to(q.dtype)
    return rearrange(out, "r h l d -> (r l) h d")[inverse_index.long()]


def _rows_layout(seg_ids, row_len):
    """First-fit the segments of ``seg_ids`` [s] into rows of ``row_len`` slots, plus one all-pad row."""
    segments = {}
    for t, sid in enumerate(seg_ids.tolist()):
        segments.setdefault(sid, []).append(t)
    rows = []  # each row: list of (segment id, token)
    for sid, toks in sorted(segments.items(), key=lambda kv: -len(kv[1])):
        assert len(toks) <= row_len
        for row in rows:
            if len(row) + len(toks) <= row_len:
                row.extend((sid, t) for t in toks)
                break
        else:
            rows.append([(sid, t) for t in toks])
    rows.append([])
    row_index = torch.zeros(len(rows) * row_len, dtype=torch.int32)
    row_seg = torch.full((len(rows), row_len), -1, dtype=torch.int32)
    inverse = torch.empty(seg_ids.shape[0], dtype=torch.int32)
    for r, row in enumerate(rows):
        for c, (sid, t) in enumerate(row):
            row_index[r * row_len + c] = t
            row_seg[r, c] = sid
            inverse[t] = r * row_len + c
    return VisionAttentionRows(row_index, row_seg, inverse)


def _segments(lengths):
    return torch.cat(
        [torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(lengths)]
    )


def _tiles(grid_h, grid_w, tile_h, tile_w):
    """Row-major patch sequence of a grid cut into tiles: tile ids interleave in token order."""
    r = torch.arange(grid_h * grid_w) // grid_w
    c = torch.arange(grid_h * grid_w) % grid_w
    return ((r // tile_h) * (grid_w // tile_w) + c // tile_w).to(torch.int32)


class _CpuRowsAttn(VisionTpuRowsAttention):
    """``VisionTpuRowsAttention`` with both Pallas ops swapped for torch references."""

    def __init__(self, head_dim, num_heads, num_kv_heads, scale):
        torch.nn.Module.__init__(self)
        self.head_size = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.flatten_batch = True
        self.scale = scale
        self._flash_op = lambda q, k, v, seg: _ref_segment_attention(
            q, k, v, seg, scale
        ).to(q.dtype)
        self._flash_rows_op = lambda q, k, v, idx, seg, inv: _ref_rows_attention(
            q, k, v, idx, seg, inv, scale
        )


class TestVisionTpuRowsAttentionCpu(CustomTestCase):
    """``forward(rows=...)`` returns per-segment attention of the token-order inputs."""

    def _check(self, seg, row_len, h=2, kv_h=None, d=16, seed=0):
        torch.manual_seed(seed)
        kv_h = kv_h or h
        scale = 1.0 / math.sqrt(d)
        s = seg.shape[0]
        q = torch.randn(s, h, d)
        k, v = (torch.randn(s, kv_h, d) for _ in range(2))
        attn = _CpuRowsAttn(d, h, kv_h, scale)
        out = attn.forward(q, k, v, bsz=1, rows=_rows_layout(seg, row_len))
        rep = h // kv_h
        ref = _ref_token_attention(
            q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), seg, scale
        )
        torch.testing.assert_close(out, ref, atol=2e-5, rtol=2e-5)

    def test_one_segment_per_row(self):
        self._check(_segments([128, 128, 128]), row_len=128)

    def test_segments_share_rows_with_padding(self):
        self._check(_segments([100, 20, 60, 128, 7, 1]), row_len=128, h=4, d=32)

    def test_interleaved_segments(self):
        # 8x16 tiles of a 32x32 grid: 128 tokens each, not contiguous in token order.
        self._check(_tiles(32, 32, 8, 16), row_len=256, seed=1)

    def test_kv_heads_expand_to_query_heads(self):
        self._check(_segments([100, 60, 90]), row_len=128, h=4, kv_h=2, seed=2)

    def test_without_rows_is_the_segment_id_path(self):
        torch.manual_seed(3)
        seg = _segments([30, 50, 20])
        q, k, v = (torch.randn(100, 2, 16) for _ in range(3))
        attn = _CpuRowsAttn(16, 2, 2, 0.25)
        out = attn.forward(q, k, v, bsz=1, segment_ids=seg[None].clone())
        torch.testing.assert_close(
            out, _ref_token_attention(q, k, v, seg, 0.25), atol=2e-5, rtol=2e-5
        )

    def test_rejects_other_masks(self):
        attn = _CpuRowsAttn(16, 2, 2, 0.25)
        q = torch.randn(128, 2, 16)
        rows = _rows_layout(torch.zeros(128, dtype=torch.int32), 128)
        with self.assertRaises(ValueError):
            attn.forward(
                q,
                q,
                q,
                bsz=1,
                rows=rows,
                segment_ids=torch.zeros(1, 128, dtype=torch.int32),
            )
        with self.assertRaises(ValueError):
            attn.forward(q, q, q, bsz=2, rows=rows)

    def test_rejects_bad_layout_shapes(self):
        attn = _CpuRowsAttn(16, 2, 2, 0.25)
        q = torch.randn(128, 2, 16)
        rows = _rows_layout(torch.zeros(128, dtype=torch.int32), 128)
        cases = (
            (q, rows._replace(inverse_index=rows.inverse_index[:64])),
            (q, rows._replace(row_index=rows.row_index[:64])),
            # row_len 96 is not a multiple of 128.
            (q[:96], _rows_layout(torch.zeros(96, dtype=torch.int32), 96)),
        )
        for x, bad in cases:
            with self.subTest(shapes=[tuple(t.shape) for t in bad]):
                with self.assertRaises(ValueError):
                    attn.forward(x, x, x, bsz=1, rows=bad)

    def test_rejects_v_heads_unlike_k(self):
        attn = _CpuRowsAttn(16, 2, 2, 0.25)
        q = torch.randn(128, 2, 16)
        rows = _rows_layout(torch.zeros(128, dtype=torch.int32), 128)
        with self.assertRaises(ValueError):
            attn.forward(q, q, torch.randn(128, 1, 16), bsz=1, rows=rows)


class TestVisionFlashRowsJax(CustomTestCase):
    """The op's JAX body, with the Pallas kernel swapped for the JAX reference attention."""

    def _check(self, seg, row_len, h, d, seed, pad_index=0):
        import jax
        import jax.numpy as jnp
        import numpy as np

        from sglang_tpu.attention_pallas import _vision_flash_rows
        from sglang_tpu.kernels.flash_attention import mha_reference

        def reference_attention(q, k, v, *, segment_ids, causal, sm_scale):
            return mha_reference(
                q, k, v, None, segment_ids=segment_ids, causal=causal, sm_scale=sm_scale
            )

        torch.manual_seed(seed)
        scale = 1.0 / math.sqrt(d)
        q, k, v = (torch.randn(seg.shape[0], h, d) for _ in range(3))
        rows = _rows_layout(seg, row_len)
        pads = rows.row_segment_ids.flatten() == -1
        rows = rows._replace(
            row_index=torch.where(
                pads, torch.tensor(pad_index, dtype=torch.int32), rows.row_index
            )
        )
        # On the JAX CPU device, so the fp32 reference is exact on a TPU host too
        # (TPU matmuls default to bf16 passes).
        with jax.default_device(jax.devices("cpu")[0]):
            out = _vision_flash_rows(
                *(jnp.asarray(x.numpy()) for x in (q, k, v)),
                *(jnp.asarray(x.numpy()) for x in rows),
                sm_scale=scale,
                attention=reference_attention,
            )
        torch.testing.assert_close(
            torch.from_numpy(np.asarray(out)),
            _ref_token_attention(q, k, v, seg, scale),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_segments_share_rows_with_padding(self):
        self._check(_segments([100, 20, 60, 128, 7, 1]), row_len=128, h=4, d=32, seed=3)

    def test_interleaved_segments(self):
        self._check(_tiles(32, 32, 8, 16), row_len=256, h=2, d=16, seed=1)

    def test_out_of_range_pad_indices_are_clamped(self):
        # Pad slots pointing past either end gather a clamped token, not NaN.
        seg = _segments([100, 20, 60, 128, 7, 1])
        for pad_index in (seg.shape[0], 1 << 20, -5):
            with self.subTest(pad_index=pad_index):
                self._check(seg, row_len=128, h=2, d=16, seed=4, pad_index=pad_index)


def _tpu_available() -> bool:
    """The TPU platform (what VisionTpuAttention requires) with a usable device.

    torch.accelerator alone is also true on CUDA hosts.
    """
    return (
        is_tpu() and hasattr(torch, "accelerator") and torch.accelerator.is_available()
    )


@unittest.skipUnless(_tpu_available(), "requires the TPU platform and a TPU device")
class TestVisionFlashRowsOnTpu(CustomTestCase):
    """The registered Pallas op in bf16, eager and compiled, against fp32 on the CPU."""

    HEADS, HEAD_DIM, ROW_LEN = 2, 96, 256
    # bf16 inputs and outputs against an fp32 reference of the same bf16 inputs.
    BF16_TOL = 2e-2

    def test_backend_registration(self):
        from sglang.srt.layers.attention.vision import QKV_BACKEND_IMPL
        from sglang_tpu.tpu_platform import _register_attention_backends

        _register_attention_backends()
        self.assertIs(QKV_BACKEND_IMPL["tpu"], VisionTpuRowsAttention)

    def test_rows_op_matches_reference_eager_and_compiled(self):
        # Four 8x32 tiles of a 16x64 grid (128 tokens each, interleaved in token
        # order) plus a 12-, a 100- and a 140-token segment: rows of 256 slots hold
        # one or several segments, one row ends in pads, and one row is all pads.
        seg = torch.cat([_tiles(16, 64, 8, 32), _segments([12, 100, 140]) + 4])
        rows = _rows_layout(seg, self.ROW_LEN)
        g = torch.Generator().manual_seed(0)
        q, k, v = (
            torch.randn(seg.shape[0], self.HEADS, self.HEAD_DIM, generator=g).bfloat16()
            for _ in range(3)
        )
        attn = VisionTpuRowsAttention(self.HEAD_DIM, self.HEADS, self.HEADS)
        ref = _ref_token_attention(q, k, v, seg, attn.scale)

        dq, dk, dv = (x.to("tpu") for x in (q, k, v))
        drows = VisionAttentionRows(*(t.to("tpu") for t in rows))
        eager = attn(dq, dk, dv, 1, rows=drows)

        def run(q, k, v, row_index, row_segment_ids, inverse_index):
            layout = VisionAttentionRows(row_index, row_segment_ids, inverse_index)
            return attn(q, k, v, 1, rows=layout)

        compiled = torch.compile(run, backend="tpu", fullgraph=True, dynamic=False)(
            dq, dk, dv, *drows
        )

        self.assertEqual(eager.dtype, torch.bfloat16)
        self.assertEqual(tuple(eager.shape), tuple(q.shape))
        torch.testing.assert_close(
            eager.float().cpu(), ref, atol=self.BF16_TOL, rtol=self.BF16_TOL
        )
        torch.testing.assert_close(compiled.cpu(), eager.cpu(), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
