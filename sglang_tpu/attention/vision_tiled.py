# SPDX-License-Identifier: Apache-2.0
"""Vision flash attention with explicit block sizes, which tile the KV sequence.

``VisionTpuAttention`` runs the flash kernel with its default blocks: up to
92,800 keys, one KV step over the whole sequence, so VMEM use grows with the
sequence. On TPU7x under a 24 MiB scoped-VMEM limit, 16,384 keys with 2+ local
heads, or 24,576 keys with any head count, fail to compile.
``VisionTpuTiledAttention`` passes explicit ``BlockSizes``: the kernel then
loops over KV blocks with an online softmax, and VMEM is bounded by the blocks.

Callers that know their sequence lengths construct it directly. Use large
tiles: on TPU7x (head_dim 72 and 96),
``BlockSizes(block_q=512, block_k_major=4096, block_k=1024, block_b=1)`` is
6-13% faster than the default single step at multiples of 4,096 keys, while
128-row query blocks are 4-17x slower and 1,024-key KV blocks about 40% slower.
The sequence is padded to a multiple of ``block_q`` and ``block_k_major``;
lengths that already are one avoid that padding work.
"""

from __future__ import annotations

import math

from sglang.srt.layers.attention.vision import VisionTpuAttention
from sglang_tpu.attention_pallas import _get_vision_flash_op
from sglang_tpu.kernels.flash_attention import MIN_BLOCK_SIZE, BlockSizes


class VisionTpuTiledAttention(VisionTpuAttention):
    """``VisionTpuAttention`` whose flash kernel runs ``flash_block_sizes``."""

    def __init__(
        self,
        head_dim,
        num_heads,
        num_kv_heads,
        *,
        flash_block_sizes: BlockSizes,
        **kwargs,
    ):
        # The tiled kernel broadcasts its softmax state over head_dim in
        # 128-lane tiles.
        if head_dim > MIN_BLOCK_SIZE and head_dim % MIN_BLOCK_SIZE:
            raise ValueError(
                f"tiled vision flash attention needs head_dim <= {MIN_BLOCK_SIZE} "
                f"or a multiple of {MIN_BLOCK_SIZE}, got {head_dim}"
            )
        super().__init__(head_dim, num_heads, num_kv_heads, **kwargs)
        b = flash_block_sizes
        # forward() pads the sequence to a multiple of _BLOCK_SIZE, which the
        # kernel's query and KV blocks must divide.
        self._BLOCK_SIZE = math.lcm(
            VisionTpuAttention._BLOCK_SIZE, b.block_q, b.block_k_major
        )
        self._flash_op = _get_vision_flash_op(float(self.scale), b)
