"""On-TPU regression test for ``VisionTpuTiledAttention`` (explicit flash block sizes).

Under a 24 MiB scoped-VMEM limit, the default vision flash op cannot compile
16,384 keys x 8 heads x head_dim 96 (one KV step over the whole sequence).
Through the real constructor and the registered Pallas op, in bf16:

* the default ``VisionTpuAttention`` fails at that shape, which shows the limit
  is in force;
* ``VisionTpuTiledAttention`` with ``BlockSizes(512, 4096, 1024)`` runs it, eager
  and under ``torch.compile(backend="tpu")``, and matches an fp32 per-segment
  reference with segments that cross KV-block boundaries;
* a 15,000-token sequence is padded to the 4,096-key tile and sliced back.
"""

import os
import re

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=120,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "infra", "multimodal", "torch-compile"],
)

# The serving limit. libtpu reads LIBTPU_INIT_ARGS when it loads, so this must
# precede the first TPU import below; CI runs each file in its own process.
_SCOPED_VMEM_FLAG = "--xla_tpu_scoped_vmem_limit_kib=24576"
os.environ["LIBTPU_INIT_ARGS"] = " ".join(
    [
        f
        for f in os.environ.get("LIBTPU_INIT_ARGS", "").split()
        if not f.startswith("--xla_tpu_scoped_vmem_limit_kib=")
    ]
    + [_SCOPED_VMEM_FLAG]
)

import unittest

import torch

from sglang.srt.utils import is_tpu
from sglang.test.test_utils import CustomTestCase


def _tpu_available() -> bool:
    """The TPU platform (what VisionTpuAttention requires) with a usable device.

    torch.accelerator alone is also true on CUDA hosts.
    """
    return (
        is_tpu() and hasattr(torch, "accelerator") and torch.accelerator.is_available()
    )


def _segments(lengths):
    """Segment ids [s] for consecutive segments of ``lengths`` tokens."""
    return torch.cat(
        [torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(lengths)]
    )


def _ref_segment_attention(q, k, v, lengths, scale):
    """Non-causal attention within each segment in fp32, one segment and head at a time.

    q, k, v: [s, h, d]. Returns [s, h, d].
    """
    out = torch.empty(q.shape, dtype=torch.float32)
    start = 0
    for n in lengths:
        for h in range(q.shape[1]):
            qs, ks, vs = (x[start : start + n, h].float() for x in (q, k, v))
            p = torch.softmax(qs @ ks.T * scale, dim=-1)
            out[start : start + n, h] = p @ vs
        start += n
    return out


@unittest.skipUnless(_tpu_available(), "requires the TPU platform and a TPU device")
class TestVisionTpuTiledAttentionOnTpu(CustomTestCase):
    HEADS, HEAD_DIM = 8, 96
    # Segments cross the 512-query and 4,096-key block boundaries; the longest
    # spans a whole KV block.
    LENGTHS = [1000, 3500, 100, 4096, 2700, 4096, 892]  # 16,384 tokens
    PADDED_LENGTHS = [1000, 3500, 100, 4096, 2700, 3604]  # 15,000 tokens
    # bf16 inputs and outputs against an fp32 reference of the same bf16 inputs.
    BF16_TOL = 2e-2

    @classmethod
    def setUpClass(cls):
        from sglang_tpu.kernels.flash_attention import BlockSizes

        cls.blocks = BlockSizes(
            block_q=512, block_k_major=4096, block_k=1024, block_b=1
        )

    def _inputs(self, lengths):
        g = torch.Generator().manual_seed(0)
        s = sum(lengths)
        q, k, v = (
            torch.randn(s, self.HEADS, self.HEAD_DIM, generator=g).bfloat16()
            for _ in range(3)
        )
        return q, k, v, _segments(lengths)[None]

    def _tiled(self):
        from sglang_tpu.attention.vision_tiled import VisionTpuTiledAttention

        return VisionTpuTiledAttention(
            self.HEAD_DIM,
            self.HEADS,
            self.HEADS,
            flatten_batch=True,
            flash_block_sizes=self.blocks,
        )

    def test_rejects_head_dim_the_tiled_kernel_cannot_run(self):
        from sglang_tpu.attention.vision_tiled import VisionTpuTiledAttention

        with self.assertRaisesRegex(ValueError, "head_dim"):
            VisionTpuTiledAttention(160, 2, 2, flash_block_sizes=self.blocks)

    def test_default_blocks_exceed_the_limit(self):
        from sglang.srt.layers.attention.vision import VisionTpuAttention

        q, k, v, seg = (x.to("tpu") for x in self._inputs(self.LENGTHS))
        attn = VisionTpuAttention(
            self.HEAD_DIM, self.HEADS, self.HEADS, flatten_batch=True
        )
        with self.assertRaisesRegex(Exception, re.compile("vmem", re.IGNORECASE)):
            attn(q, k, v, 1, segment_ids=seg).cpu()

    def test_tiled_matches_reference_eager_and_compiled(self):
        q, k, v, seg = self._inputs(self.LENGTHS)
        attn = self._tiled()
        ref = _ref_segment_attention(q, k, v, self.LENGTHS, attn.scale)

        dq, dk, dv, dseg = (x.to("tpu") for x in (q, k, v, seg))
        eager = attn(dq, dk, dv, 1, segment_ids=dseg)

        def run(q, k, v, segment_ids):
            return attn(q, k, v, 1, segment_ids=segment_ids)

        compiled = torch.compile(run, backend="tpu", fullgraph=True, dynamic=False)(
            dq, dk, dv, dseg
        )

        self.assertEqual(eager.dtype, torch.bfloat16)
        self.assertEqual(tuple(eager.shape), tuple(q.shape))
        torch.testing.assert_close(
            eager.float().cpu(), ref, atol=self.BF16_TOL, rtol=self.BF16_TOL
        )
        torch.testing.assert_close(compiled.cpu(), eager.cpu(), atol=0, rtol=0)

    def test_pads_to_the_tile(self):
        q, k, v, seg = self._inputs(self.PADDED_LENGTHS)
        attn = self._tiled()
        self.assertEqual(attn._BLOCK_SIZE, 4096)
        ref = _ref_segment_attention(q, k, v, self.PADDED_LENGTHS, attn.scale)

        out = attn(*(x.to("tpu") for x in (q, k, v)), 1, segment_ids=seg.to("tpu"))

        self.assertEqual(tuple(out.shape), tuple(q.shape))
        torch.testing.assert_close(
            out.float().cpu(), ref, atol=self.BF16_TOL, rtol=self.BF16_TOL
        )


if __name__ == "__main__":
    unittest.main()
