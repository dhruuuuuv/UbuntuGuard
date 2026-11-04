import math

import pytest

from evaluate import auroc, p_fail_from_logprobs

VERDICT_IDS = {"PASS": 1, "FAIL": 2}


def test_p_fail_normalises_over_the_two_labels():
    top = {1: math.log(0.25), 2: math.log(0.70), 3: math.log(0.05)}
    assert p_fail_from_logprobs(top, VERDICT_IDS) == (pytest.approx(0.70 / 0.95), False)


def test_p_fail_missing_labels():
    # One label missing: clamped to the lowest returned log-probability
    p_fail, clamped = p_fail_from_logprobs({2: -0.01, 3: -9.0}, VERDICT_IDS)
    assert p_fail == pytest.approx(math.exp(-0.01) / (math.exp(-0.01) + math.exp(-9.0)))
    assert clamped
    # Both missing: no score
    assert p_fail_from_logprobs({3: -0.1}, VERDICT_IDS) == (None, True)


def test_auroc():
    assert auroc([False, False, True, True], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([False, True, False, True], [0.5, 0.5, 0.5, 0.5]) == 0.5
    # Pairs (FAIL, PASS): win, tie, win, win -> (3 + 0.5) / 4
    assert auroc([True, True, False, False], [0.6, 0.9, 0.2, 0.6]) == 0.875
