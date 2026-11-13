import math

import pytest

from evaluate import auroc, verdict_scores


def test_verdict_scores_normalise_over_the_two_labels():
    s = verdict_scores(math.log(0.25), math.log(0.70))
    assert s["p_fail"] == pytest.approx(0.70 / 0.95)
    assert s["fail_log_odds"] == pytest.approx(math.log(0.70 / 0.25))
    assert s["verdict_mass"] == pytest.approx(0.95)


def test_verdict_scores_keep_order_when_very_confident():
    # p_fail rounds to 1.0 for both, but log-odds still rank them
    a = verdict_scores(-50.0, -1e-6)
    b = verdict_scores(-80.0, -1e-6)
    assert a["p_fail"] == b["p_fail"] == 1.0
    assert b["fail_log_odds"] > a["fail_log_odds"]
    assert verdict_scores(-1e-6, -900.0)["p_fail"] == 0.0  # no overflow


def test_auroc():
    assert auroc([False, False, True, True], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([False, True, False, True], [0.5, 0.5, 0.5, 0.5]) == 0.5
    # Pairs (FAIL, PASS): win, tie, win, win -> (3 + 0.5) / 4
    assert auroc([True, True, False, False], [0.6, 0.9, 0.2, 0.6]) == 0.875
