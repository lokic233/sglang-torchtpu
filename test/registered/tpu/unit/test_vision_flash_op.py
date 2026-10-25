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

import torch
from einops import rearrange

from sglang.srt.layers.attention.vision import VisionAttentionRows, VisionTpuAttention
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


def _ref_rows_attention(q, k, v, row_index, row_segment_ids, inverse_index, scale):
    """Pure-torch reference for the rows op: gather, per-row segment attention, gather back.

    q, k, v: [s, h, d]. Mirrors ``_get_vision_flash_rows_op``.
    """
    num_rows, row_len = row_segment_ids.shape
    rows = [
        rearrange(x[row_index.long()], "(r l) h d -> r h l d", r=num_rows)
        for x in (q, k, v)
    ]
    out = _ref_segment_attention(*rows, row_segment_ids, scale)
    return rearrange(out, "r h l d -> (r l) h d")[inverse_index.long()]


def _rows_layout(seg_ids, row_len):
    """First-fit the segments of ``seg_ids`` [s] into rows of ``row_len`` slots."""
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
    rows.append([])  # one all-pad row, so pad slots exist
    pad_id = -1
    row_index = torch.zeros(len(rows) * row_len, dtype=torch.int32)
    row_seg = torch.full((len(rows), row_len), pad_id, dtype=torch.int32)
    inverse = torch.empty(seg_ids.shape[0], dtype=torch.int32)
    for r, row in enumerate(rows):
        for c, (sid, t) in enumerate(row):
            row_index[r * row_len + c] = t
            row_seg[r, c] = sid
            inverse[t] = r * row_len + c
    return VisionAttentionRows(row_index, row_seg, inverse)


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
        self._flash_rows_op = lambda q, k, v, idx, seg, inv: _ref_rows_attention(
            q, k, v, idx, seg, inv, scale
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


class TestVisionFlashRowsWrapper(CustomTestCase):
    """``rows=`` returns per-segment attention of the token-order inputs."""

    def _run(self, seq_lens, row_len, h=2, d=16, seed=0):
        torch.manual_seed(seed)
        s = sum(seq_lens)
        scale = 1.0 / math.sqrt(d)
        q, k, v = (torch.randn(s, h, d) for _ in range(3))
        seg = torch.cat(
            [torch.full((L,), i, dtype=torch.int32) for i, L in enumerate(seq_lens)]
        )
        attn = _CpuVisionAttn(d, h, scale)
        out = attn.forward(q, k, v, bsz=1, rows=_rows_layout(seg, row_len))
        qr, kr, vr = (rearrange(x, "(b s) h d -> b h s d", b=1) for x in (q, k, v))
        ref = rearrange(
            _ref_segment_attention(qr, kr, vr, seg[None], scale),
            "b h s d -> (b s) h d",
        )
        return out, ref

    def test_one_segment_per_row(self):
        out, ref = self._run([32, 32, 32], row_len=32)
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)

    def test_segments_share_rows_with_padding(self):
        # Short segments share rows; leftover slots are pads.
        out, ref = self._run([20, 7, 13, 32, 5, 1], row_len=32, h=4, d=32, seed=3)
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)

    def test_interleaved_segments(self):
        # Segment tokens need not be contiguous in token order (2D tiles in a
        # row-major patch sequence are not).
        torch.manual_seed(1)
        s, h, d, scale = 48, 2, 16, 0.25
        q, k, v = (torch.randn(s, h, d) for _ in range(3))
        seg = torch.tensor(
            [(t % 12) // 4 + 3 * (t // 24) for t in range(s)], dtype=torch.int32
        )
        attn = _CpuVisionAttn(d, h, scale)
        out = attn.forward(q, k, v, bsz=1, rows=_rows_layout(seg, 16))
        qr, kr, vr = (rearrange(x, "(b s) h d -> b h s d", b=1) for x in (q, k, v))
        ref = rearrange(
            _ref_segment_attention(qr, kr, vr, seg[None], scale), "b h s d -> (b s) h d"
        )
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)

    def test_rows_reject_other_masks(self):
        attn = _CpuVisionAttn(16, 2, 0.25)
        q = torch.randn(8, 2, 16)
        rows = _rows_layout(torch.zeros(8, dtype=torch.int32), 8)
        with self.assertRaises(ValueError):
            attn.forward(
                q,
                q,
                q,
                bsz=1,
                rows=rows,
                segment_ids=torch.zeros(1, 8, dtype=torch.int32),
            )
        with self.assertRaises(ValueError):
            attn.forward(q, q, q, bsz=2, rows=rows)


if __name__ == "__main__":
    unittest.main()
