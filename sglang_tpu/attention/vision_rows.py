"""Segment-local vision attention over gathered rows.

ViTs with window- or tile-local layers (each token attends only within its own
window or tile) run those layers through ``VisionTpuAttention`` with
``segment_ids``. The Pallas flash kernel masks by segment but still computes
every query-key block of the whole sequence, so a local layer costs as much as
full attention. ``VisionTpuRowsAttention`` additionally takes a host-built
layout (``rows=``) that gathers each segment's tokens into short fixed-length
rows and attends within a row only (``_get_vision_flash_rows_op``).

``_register_attention_backends`` installs it as the ``"tpu"`` vision attention
backend; without ``rows=`` it is ``VisionTpuAttention`` unchanged.

Callers that pass ``rows=`` must call this class directly. ``VisionAttention``
does not forward ``rows=``, and a plain ``VisionTpuAttention`` (the default of
other vision runners, or the ``"tpu"`` entry before registration) accepts it
through ``**kwargs`` and ignores it, running full attention without an error.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from sglang.srt.layers.attention.vision import VisionTpuAttention
from sglang_tpu.attention_pallas import _get_vision_flash_rows_op

# The flash kernel runs a row as one KV block, which must be a multiple of 128.
ROW_LEN_MULTIPLE = 128


class VisionAttentionRows(NamedTuple):
    """Segment-local attention layout for ``VisionTpuRowsAttention(rows=...)``.

    Tokens of each short segment are gathered into rows of equal length, and
    attention runs within a row only.

    row_index: [num_rows * row_len] int32, the token held by each row slot. A pad
        slot may hold any index: the gather clamps it into range, and its segment
        ID keeps it out of every real token's attention.
    row_segment_ids: [num_rows, row_len] int32, the segment of each slot. A query
        attends only to keys of its own segment in its row; pad slots use an ID
        that no segment in their row uses.
    inverse_index: [seq_len] int32, the row slot whose output each token takes.

    The caller builds the layout on the host and keeps it static per shape. A
    segment must fit in one row (a split segment attends only within each part),
    and ``row_len`` must be a multiple of 128.
    """

    row_index: torch.Tensor
    row_segment_ids: torch.Tensor
    inverse_index: torch.Tensor


class VisionTpuRowsAttention(VisionTpuAttention):
    """``VisionTpuAttention`` that also runs segment-local attention over rows.

    Call it directly with ``rows=``; see the module docstring for the paths that
    would drop the layout silently.
    """

    def __init__(self, head_dim, num_heads, num_kv_heads, **kwargs):
        super().__init__(head_dim, num_heads, num_kv_heads, **kwargs)
        self._flash_rows_op = _get_vision_flash_rows_op(float(self.scale))

    def forward(
        self,
        q,
        k,
        v,
        bsz,
        cu_seqlens=None,
        attention_mask=None,
        segment_ids=None,
        rows: VisionAttentionRows | None = None,
        **kwargs,
    ):
        if rows is None:
            return super().forward(
                q,
                k,
                v,
                bsz,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                **kwargs,
            )
        if (
            attention_mask is not None
            or segment_ids is not None
            or cu_seqlens is not None
            or bsz != 1
        ):
            raise ValueError(
                "TPU vision row attention takes no attention_mask, segment_ids "
                "or cu_seqlens, and needs bsz == 1"
            )
        if (
            rows.inverse_index.shape != (q.shape[0],)
            or rows.row_segment_ids.dim() != 2
            or rows.row_index.shape != (rows.row_segment_ids.numel(),)
            or rows.row_segment_ids.shape[1] % ROW_LEN_MULTIPLE != 0
        ):
            raise ValueError(
                "TPU vision row attention needs inverse_index [seq_len], "
                "row_segment_ids [num_rows, row_len] with row_len a multiple of "
                f"{ROW_LEN_MULTIPLE}, and row_index [num_rows * row_len]; got "
                f"{tuple(rows.inverse_index.shape)}, "
                f"{tuple(rows.row_segment_ids.shape)}, "
                f"{tuple(rows.row_index.shape)} for seq_len {q.shape[0]}"
            )
        # Same KV-head checks and expansion as the segment-id path.
        if k.shape[1] != v.shape[1] or q.shape[1] % k.shape[1] != 0:
            raise ValueError(
                "TPU vision attention requires matching KV heads and an "
                "integer query-to-KV head ratio"
            )
        if k.shape[1] != q.shape[1]:
            repeats = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        return self._flash_rows_op(
            q,
            k,
            v,
            rows.row_index.to(device=q.device, dtype=torch.int32),
            rows.row_segment_ids.to(device=q.device, dtype=torch.int32),
            rows.inverse_index.to(device=q.device, dtype=torch.int32),
        )
