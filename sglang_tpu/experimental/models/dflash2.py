"""Dense DFlash2 selector: K target-head candidates per slot, one path walked over their lattice.

score[b, j, p, c] = unary[b, j, c] + <A[cand[b, j-1, p]] * proj(h[b, j]), B[cand[b, j, c]]>,
with the anchor as slot 0's only predecessor; slot j draws from score[b, j, i_{j-1}].
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.models.dflash import CandidateSelector, DFlash2DraftModel
from sglang_tpu.experimental.speculative.spec_ops import pick

# (row_scores [pbs, K], slot) -> chosen candidate index [pbs].
CandidateChooser = Callable[[torch.Tensor, int], torch.Tensor]
# (candidate_ids [pbs, gamma, K], scores [pbs, gamma, K, K]).
DraftLattice = tuple[torch.Tensor, torch.Tensor]


def build_lattice(
    model: DFlash2DraftModel, draft_hidden: torch.Tensor, anchor_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Candidate ids [pbs, gamma, K] and scores [pbs, gamma, K, K] from [pbs, gamma, H] hidden."""
    bs, gamma = draft_hidden.shape[:2]
    candidate_ids, unary_logits = model.compute_candidates(
        draft_hidden.reshape(bs * gamma, -1)
    )
    candidate_ids = candidate_ids.view(bs, gamma, -1)
    selector = model.candidate_selector
    assert isinstance(selector, CandidateSelector)
    # Core's _score_edges without its torch.compile / mark_static, which the TPU trace
    # cannot take; table lookups go through embedding, not advanced indexing.
    hidden = selector.hidden_projection(draft_hidden)
    predecessor_ids = torch.cat(
        [
            anchor_tokens[:, None, None].expand(-1, 1, selector.top_k),
            candidate_ids[:, :-1],
        ],
        dim=1,
    )
    predecessors = F.embedding(predecessor_ids, selector.predecessor_codebook)
    successors = F.embedding(candidate_ids, selector.successor_codebook)
    scores = unary_logits.view(bs, gamma, 1, -1) + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )
    return candidate_ids, scores


def walk_lattice(
    *, candidate_ids: torch.Tensor, scores: torch.Tensor, choose: CandidateChooser
) -> torch.Tensor:
    """Drafts [pbs, gamma]: each slot chooses from the row its predecessor selects."""
    # Slot 0's K rows all score from the anchor, so any row will do.
    index = torch.zeros(
        candidate_ids.shape[0], dtype=torch.int64, device=candidate_ids.device
    )
    tokens = []
    for slot in range(candidate_ids.shape[1]):
        index = choose(pick(scores[:, slot], index), slot)
        tokens.append(pick(candidate_ids[:, slot], index))
    return torch.stack(tokens, dim=1)


def walk_lattice_sampled(
    *,
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    uniforms: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
) -> torch.Tensor:
    """Drafts [pbs, gamma] by core's sample_path inverse-CDF walk over the K candidates.

    Every predecessor row draws at once; the walk only follows indices, via one-hot picks.
    """
    top_k = scores.shape[-1]
    temps = temperatures.view(-1).float().clamp_min(1e-5)[:, None, None, None]
    probs = torch.softmax(scores.float() / temps, dim=-1)
    maps = (uniforms[:, :, None, None] >= probs.cumsum(dim=-1)).sum(dim=-1)
    maps = torch.where(
        greedy_mask.view(-1, 1, 1), scores.argmax(dim=-1), maps.clamp_max(top_k - 1)
    )
    # Slot 0's K rows all score from the anchor, so any row will do.
    index = torch.zeros(
        candidate_ids.shape[0], dtype=torch.int64, device=candidate_ids.device
    )
    tokens = []
    for slot in range(candidate_ids.shape[1]):
        index = pick(maps[:, slot], index)
        tokens.append(pick(candidate_ids[:, slot], index))
    return torch.stack(tokens, dim=1)


@compiled_op(uses_rng=True)
def sample_lattice_path(
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
) -> torch.Tensor:
    """Sampled drafts [pbs, gamma] in one program; rows with top_k <= 1 walk the argmax."""
    uniforms = torch.rand(scores.shape[:2], dtype=torch.float32, device=scores.device)
    return walk_lattice_sampled(
        candidate_ids=candidate_ids,
        scores=scores,
        uniforms=uniforms,
        temperatures=temperatures,
        greedy_mask=top_ks.view(-1) <= 1,
    )


@compiled_op()
def lattice_path_logits(
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    draft_tokens: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Dense [pbs, gamma, V] rows of draft_tokens' path, rebuilt from the synced drafts."""
    index = torch.zeros(
        candidate_ids.shape[0], dtype=torch.int64, device=candidate_ids.device
    )
    rows = []
    for slot in range(candidate_ids.shape[1]):
        rows.append(pick(scores[:, slot], index).float())
        slot_ids = candidate_ids[:, slot]
        index = (slot_ids == draft_tokens[:, slot, None]).to(torch.int32).argmax(dim=-1)
    rows_3d = torch.stack(rows, dim=1)
    # -inf off the candidates, so softmax(dense / T) is the selector's q.
    return torch.full(
        (*rows_3d.shape[:2], vocab_size),
        float("-inf"),
        dtype=rows_3d.dtype,
        device=rows_3d.device,
    ).scatter(-1, candidate_ids, rows_3d)


def sample_drafts(
    *,
    draft_lattice: DraftLattice,
    vocab_size: int,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    sync: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sampled drafts [pbs, gamma] and the dense [pbs, gamma, V] logits q is drawn from."""
    candidate_ids, scores = draft_lattice
    # Ranks draw from their own generators: sync once, then every rank rebuilds q
    # along the synced path.
    draft_tokens = sync(
        sample_lattice_path(candidate_ids, scores, temperatures, top_ks)
    )
    return draft_tokens, lattice_path_logits(
        candidate_ids, scores, draft_tokens, vocab_size
    )


def sample_draft_tokens(
    self: DFlash2DraftModel, hidden_states: torch.Tensor, input_ids: torch.Tensor
) -> tuple[torch.Tensor, DraftLattice]:
    """Greedy selector path [bs, gamma] and its lattice, traced into the TPU draft graph."""
    block = self.block_size
    hidden_3d = hidden_states.view(-1, block, hidden_states.shape[-1])
    candidate_ids, scores = build_lattice(
        self, hidden_3d[:, 1:], input_ids.view(-1, block)[:, 0]
    )
    tokens = walk_lattice(
        candidate_ids=candidate_ids,
        scores=scores,
        choose=lambda row, slot: torch.argmax(row, dim=-1),
    )
    return tokens, (candidate_ids, scores)
