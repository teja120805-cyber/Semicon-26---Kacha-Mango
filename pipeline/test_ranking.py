"""Unit tests for pipeline/ranking.py::apply_center_tiebreak - the Applied
Materials spec's "closest to Search-image centre" tie-break for multiple
valid/equivalent matches.

These test the tie-break in isolation against synthetic Candidate objects
(no images, no correlation, no generator) so the tie/equivalence logic and
the centre-distance geometry can be verified independently of the rest of
the pipeline.
"""
from __future__ import annotations

from pipeline.candidate_generation import Candidate
from pipeline.ranking import (
    MULTIWAY_MAX_SPREAD_PX,
    MULTIWAY_MIN_GROUP_SIZE,
    MULTIWAY_TIE_SCORE_EPSILON,
    TIE_SCORE_EPSILON,
    apply_center_tiebreak,
)


def _cand(x, y, score):
    return Candidate(x=x, y=y, score=score, scale=10.0, rotation_deg=0.0, template_size=100)


def test_unique_winner_is_kept_regardless_of_center_distance():
    """TEST 1: A clearly better than B -> A wins regardless of centre distance."""
    a = _cand(x=900.0, y=900.0, score=0.90)   # far from centre, clearly best
    b = _cand(x=500.0, y=500.0, score=0.50)   # exactly at centre, far worse
    ranked = [a, b]
    result = apply_center_tiebreak(ranked, search_shape=(1000, 1000))
    assert result[0] is a


def test_exact_tie_prefers_candidate_closer_to_center():
    """TEST 2: A and B have equal score; B is closer to centre -> B wins."""
    a = _cand(x=950.0, y=950.0, score=0.80)   # far from centre
    b = _cand(x=520.0, y=510.0, score=0.80)   # close to centre
    ranked = [a, b]
    result = apply_center_tiebreak(ranked, search_shape=(1000, 1000))
    assert result[0] is b


def test_multiple_tied_candidates_closest_to_center_wins():
    """TEST 3: Three candidates with equivalent (genuinely equal, within
    float precision) scores -> closest to centre wins."""
    far = _cand(x=50.0, y=50.0, score=0.80)
    mid = _cand(x=700.0, y=700.0, score=0.80)
    closest = _cand(x=505.0, y=495.0, score=0.80)
    ranked = [far, mid, closest]  # already sorted by score, descending
    result = apply_center_tiebreak(ranked, search_shape=(1000, 1000))
    assert result[0] is closest


def test_center_computed_dynamically_from_arbitrary_image_size():
    """TEST 4: A non-1000x1000 image -> centre must come from actual dimensions,
    never hardcoded (500, 500)."""
    # 400 (width) x 200 (height) image -> true centre is (200, 100), not (500, 500).
    near_true_center = _cand(x=205.0, y=98.0, score=0.85)
    near_hardcoded_center = _cand(x=505.0, y=505.0, score=0.85)  # outside this image entirely
    ranked = [near_hardcoded_center, near_true_center]
    result = apply_center_tiebreak(ranked, search_shape=(200, 400))
    assert result[0] is near_true_center


def test_tie_break_does_not_alter_genuinely_better_winner():
    """TEST 5: One candidate is genuinely better (score gap far above the
    tie epsilon) but farther from centre -> it must still win; the
    tie-break must never override a real score-based winner."""
    genuinely_better = _cand(x=950.0, y=950.0, score=1.00)   # far from centre
    worse_but_central = _cand(x=500.0, y=500.0, score=0.50)  # centred, but not tied
    ranked = [genuinely_better, worse_but_central]
    result = apply_center_tiebreak(ranked, search_shape=(1000, 1000))
    assert result[0] is genuinely_better


def test_tie_break_is_deterministic():
    """TEST 6: Same candidates/scores, repeated evaluation -> same result every time."""
    a = _cand(x=900.0, y=900.0, score=0.80)
    b = _cand(x=520.0, y=510.0, score=0.80)
    c = _cand(x=100.0, y=100.0, score=0.80)
    ranked = [a, b, c]
    results = [
        [cand.x for cand in apply_center_tiebreak(list(ranked), search_shape=(1000, 1000))]
        for _ in range(10)
    ]
    assert all(r == results[0] for r in results)


def test_no_op_below_two_candidates():
    """A single candidate (or an empty pool) must pass through unchanged -
    there is nothing to tie-break."""
    a = _cand(x=900.0, y=900.0, score=0.80)
    assert apply_center_tiebreak([a], search_shape=(1000, 1000)) == [a]
    assert apply_center_tiebreak([], search_shape=(1000, 1000)) == []


def test_rank_classical_unchanged_pure_arg_max():
    """Regression guard: rank_classical itself must remain untouched (pure
    score-descending sort) - the tie-break is a separate, later step."""
    from pipeline.ranking import rank_classical
    a = _cand(x=10.0, y=10.0, score=0.50)
    b = _cand(x=990.0, y=990.0, score=0.90)
    result = rank_classical([a, b])
    assert result[0] is b and result[1] is a


def test_similar_but_not_equal_scores_are_not_treated_as_tied():
    """Guard against the failure mode found during implementation: scores
    that are merely close (a realistic gap between two genuinely different
    candidate locations) must NOT be treated as tied - only near-exact
    equality (within TIE_SCORE_EPSILON) qualifies. Reusing a loose
    percentage-based tolerance here previously collapsed benchmark
    accuracy from 71.2% to 33.3%@5px because ZNCC scores decay gradually
    across distinct wrong-location candidates."""
    better = _cand(x=900.0, y=900.0, score=0.80)
    close_but_distinct = _cand(x=500.0, y=500.0, score=0.80 - TIE_SCORE_EPSILON * 100)
    ranked = [better, close_but_distinct]
    result = apply_center_tiebreak(ranked, search_shape=(1000, 1000))
    assert result[0] is better


# --- Multiway tier (integrated 2026-08-15) - see pipeline/ranking.py's
# MULTIWAY_* constants for the full rationale and experiments/multiway_tiebreak_v1/
# for the empirical derivation. These tests exercise the tier in isolation,
# same style as the tests above. ---

def test_multiway_tier_fires_with_three_tied_candidates_and_small_spread():
    """A group of >= MULTIWAY_MIN_GROUP_SIZE candidates within
    MULTIWAY_TIE_SCORE_EPSILON of the top score, spatially close together
    (a stand-in for a genuine periodicity pileup) -> closest to centre wins,
    exactly like the tight tier, even though the gap is far above
    TIE_SCORE_EPSILON."""
    gap = MULTIWAY_TIE_SCORE_EPSILON / 4  # keep every member's gap-from-best strictly under the epsilon
    far = _cand(x=600.0, y=600.0, score=0.80)
    mid = _cand(x=580.0, y=560.0, score=0.80 - gap)
    closest = _cand(x=520.0, y=480.0, score=0.80 - 2 * gap)
    ranked = [far, mid, closest]  # already sorted by score, descending
    assert _max_spread(ranked) <= MULTIWAY_MAX_SPREAD_PX  # sanity: this is the "small spread" case
    result = apply_center_tiebreak(ranked, search_shape=(1000, 1000))
    assert result[0] is closest


def test_multiway_tier_does_not_fire_with_only_two_candidates():
    """The same wide-but-not-tight score gap, but only 2 candidates -
    exactly the pairwise-coincidence pattern both rejected prior attempts
    exhibited. Must NOT fire (group size < MULTIWAY_MIN_GROUP_SIZE), even
    though a lone pairwise threshold would have treated this as tied."""
    assert MULTIWAY_MIN_GROUP_SIZE >= 3, "test assumes the validated minimum of 3"
    gap = MULTIWAY_TIE_SCORE_EPSILON / 2
    winner = _cand(x=900.0, y=900.0, score=0.80)
    close_but_only_pair = _cand(x=500.0, y=500.0, score=0.80 - gap)
    ranked = [winner, close_but_only_pair]
    result = apply_center_tiebreak(ranked, search_shape=(1000, 1000))
    assert result[0] is winner


def test_multiway_tier_does_not_fire_when_spread_too_large():
    """>= MULTIWAY_MIN_GROUP_SIZE candidates tied by score, but scattered
    across the image far beyond MULTIWAY_MAX_SPREAD_PX - the signature of
    the single worst regression in the rejected center_tiebreak_v2
    experiment (a 525px outlier). Must NOT fire even though the group-size
    condition alone is satisfied."""
    gap = MULTIWAY_TIE_SCORE_EPSILON / 2
    winner = _cand(x=900.0, y=900.0, score=0.80)
    scattered_a = _cand(x=100.0, y=100.0, score=0.80 - gap)
    scattered_b = _cand(x=900.0, y=100.0, score=0.80 - 2 * gap)
    ranked = [winner, scattered_a, scattered_b]
    assert _max_spread(ranked) > MULTIWAY_MAX_SPREAD_PX  # sanity: this is the "too spread out" case
    result = apply_center_tiebreak(ranked, search_shape=(1000, 1000))
    assert result[0] is winner


def test_multiway_tier_does_not_fire_beyond_its_own_epsilon():
    """A gap wider than MULTIWAY_TIE_SCORE_EPSILON itself is a genuine
    ranking decision, not a tie at any tier - must never fire regardless of
    group size or spread."""
    winner = _cand(x=900.0, y=900.0, score=0.80)
    b = _cand(x=520.0, y=510.0, score=0.80 - MULTIWAY_TIE_SCORE_EPSILON * 2)
    c = _cand(x=515.0, y=505.0, score=0.80 - MULTIWAY_TIE_SCORE_EPSILON * 3)
    ranked = [winner, b, c]
    result = apply_center_tiebreak(ranked, search_shape=(1000, 1000))
    assert result[0] is winner


def test_tight_tier_takes_precedence_and_is_never_weakened_by_multiway_params():
    """When the tight tier already fires (a genuine near-exact tie), the
    multiway tier's group-size/spread gates must never veto it - the
    original, provably-safe behavior is unconditional and must not regress
    now that a second tier exists alongside it."""
    a = _cand(x=950.0, y=950.0, score=0.80)   # far from centre
    b = _cand(x=520.0, y=510.0, score=0.80)   # tight tie with a, close to centre
    ranked = [a, b]  # only 2 candidates - below multiway_min_group_size
    result = apply_center_tiebreak(ranked, search_shape=(1000, 1000))
    assert result[0] is b


def _max_spread(ranked) -> float:
    import math
    max_d = 0.0
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            d = math.hypot(ranked[i].x - ranked[j].x, ranked[i].y - ranked[j].y)
            max_d = max(max_d, d)
    return max_d
