"""Unit tests for the response-token slicing under left-padding.

This is the most error-prone part of the batched extraction backend. We
verify it against synthetic activations where each token's value equals its
position index, so the mean-pool result is a known integer arithmetic mean.
"""

from __future__ import annotations

import torch

from src.extraction.backend_hf import slice_response_tokens_left_padded


def _make_position_activations(B: int, S: int, D: int) -> torch.Tensor:
    """Activations[b, s, d] = s for every (b, d). Lets us verify mean-pool slicing."""
    base = torch.arange(S, dtype=torch.float32).reshape(1, S, 1).expand(B, S, D)
    return base.contiguous()


def test_left_padded_response_slice_basic() -> None:
    """3 rows; left padding; response tokens are the last (full_n - prefix_n) positions."""
    B, S, D = 3, 10, 4
    acts = _make_position_activations(B, S, D)

    # Row 0: full=10, prefix=4 → response = positions [4..9], mean = 6.5
    # Row 1: full=8 (padded by 2), prefix=3 → response = positions [5..9], mean = 7
    # Row 2: full=6 (padded by 4), prefix=2 → response = positions [6..9], mean = 7.5
    full_lens = [10, 8, 6]
    prefix_lens = [4, 3, 2]
    attn = torch.zeros(B, S, dtype=torch.long)
    for i, n in enumerate(full_lens):
        attn[i, S - n : S] = 1

    pooled = slice_response_tokens_left_padded(acts, attn, full_lens, prefix_lens)
    assert len(pooled) == B
    # Row 0: response slice = [last 6 positions] = positions 4..9 → mean 6.5
    assert torch.allclose(pooled[0], torch.full((D,), 6.5, dtype=torch.float32))
    # Row 1: response_n = 8 - 3 = 5 → positions [5..9] → mean 7.0
    assert torch.allclose(pooled[1], torch.full((D,), 7.0, dtype=torch.float32))
    # Row 2: response_n = 6 - 2 = 4 → positions [6..9] → mean 7.5
    assert torch.allclose(pooled[2], torch.full((D,), 7.5, dtype=torch.float32))


def test_no_response_pools_all_real_tokens() -> None:
    """prefix_n=0 means no response → mean over all real tokens."""
    B, S, D = 2, 8, 2
    acts = _make_position_activations(B, S, D)
    full_lens = [8, 5]  # row 1 is padded by 3
    prefix_lens = [0, 0]
    attn = torch.zeros(B, S, dtype=torch.long)
    attn[0, :] = 1  # all 8 real
    attn[1, S - 5 :] = 1  # last 5 real, positions 3..7

    pooled = slice_response_tokens_left_padded(acts, attn, full_lens, prefix_lens)
    # Row 0: positions 0..7 → mean 3.5
    assert torch.allclose(pooled[0], torch.full((D,), 3.5, dtype=torch.float32))
    # Row 1: positions 3..7 → mean 5.0
    assert torch.allclose(pooled[1], torch.full((D,), 5.0, dtype=torch.float32))


def test_truncation_falls_back_to_real_tokens() -> None:
    """If tokenizer truncated such that prefix > actual, fall back to all real."""
    B, S, D = 1, 6, 1
    acts = _make_position_activations(B, S, D)
    # full_n=10 but actual=6 (truncated by 4); prefix_n=8
    # → response_n would be -2 (degenerate); fall back to mean of all real.
    full_lens = [10]
    prefix_lens = [8]
    attn = torch.ones(B, S, dtype=torch.long)
    pooled = slice_response_tokens_left_padded(acts, attn, full_lens, prefix_lens)
    # All real → positions 0..5 → mean 2.5
    assert torch.allclose(pooled[0], torch.full((D,), 2.5, dtype=torch.float32))


def test_response_n_exceeds_actual_falls_back() -> None:
    """If full - prefix > actual_count (impossible normally), fall back gracefully."""
    B, S, D = 1, 5, 1
    acts = _make_position_activations(B, S, D)
    # actual=3, prefix=0, full=10 → response_n = 10, but only 3 real tokens.
    full_lens = [10]
    prefix_lens = [0]
    attn = torch.zeros(B, S, dtype=torch.long)
    attn[0, S - 3 :] = 1  # last 3 real
    pooled = slice_response_tokens_left_padded(acts, attn, full_lens, prefix_lens)
    # mean over real = positions 2..4 → mean 3.0
    assert torch.allclose(pooled[0], torch.full((D,), 3.0, dtype=torch.float32))


def test_minimal_response_one_token() -> None:
    """Response = 1 token at the very end."""
    B, S, D = 1, 6, 3
    acts = _make_position_activations(B, S, D)
    full_lens = [6]
    prefix_lens = [5]  # response_n = 1
    attn = torch.ones(B, S, dtype=torch.long)
    pooled = slice_response_tokens_left_padded(acts, attn, full_lens, prefix_lens)
    # Only position 5 → value 5
    assert torch.allclose(pooled[0], torch.full((D,), 5.0, dtype=torch.float32))


def test_per_token_dim_independence() -> None:
    """Different d-values per row must be averaged independently."""
    B, S, D = 1, 4, 3
    acts = torch.zeros(B, S, D, dtype=torch.float32)
    # position 0: d=[1,2,3]
    # position 1: d=[4,5,6]
    # position 2: d=[7,8,9]
    # position 3: d=[10,11,12]
    for s in range(S):
        for d in range(D):
            acts[0, s, d] = float(s * D + d + 1)

    full_lens = [4]
    prefix_lens = [2]  # response = positions [2, 3]
    attn = torch.ones(B, S, dtype=torch.long)
    pooled = slice_response_tokens_left_padded(acts, attn, full_lens, prefix_lens)
    # mean of positions 2 and 3 per d-channel:
    # d=0: (7 + 10) / 2 = 8.5
    # d=1: (8 + 11) / 2 = 9.5
    # d=2: (9 + 12) / 2 = 10.5
    expected = torch.tensor([8.5, 9.5, 10.5])
    assert torch.allclose(pooled[0], expected)
