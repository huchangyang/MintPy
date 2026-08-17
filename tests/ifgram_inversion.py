#!/usr/bin/env python3
"""Test ifgram_inversion partial-network inversion with a fixed global reference date."""

import numpy as np

from mintpy.ifgram_inversion import (
    build_partial_design_matrix,
    estimate_timeseries,
    estimate_timeseries_cov,
    get_date_edges,
    get_ref_connected_component,
)


DATES = ['20200101', '20200113', '20200125', '20200206', '20200218']
TBASE = np.array([0.0, 12, 24, 36, 48], dtype=np.float32) / 365.25


def _invert(date12_list, y, ref_date='20200101', min_norm_velocity=True,
            min_redundancy=1.0, weight_sqrt=None):
    # A/B are unused by the partial-network path but required by the function signature.
    A = np.zeros((len(date12_list), len(DATES) - 1), dtype=np.float32)
    B = np.zeros_like(A)
    tbase_diff = np.diff(TBASE).reshape(-1, 1)
    return estimate_timeseries(
        A, B, np.asarray(y, dtype=np.float32), tbase_diff,
        weight_sqrt=None if weight_sqrt is None else np.asarray(weight_sqrt, dtype=np.float32),
        min_norm_velocity=min_norm_velocity,
        min_redundancy=min_redundancy,
        allow_partial_network=True,
        date_list=DATES,
        date12_list=date12_list,
        ref_date=ref_date,
        tbase=TBASE,
        print_msg=False,
    )


def test_ref_connected_component_keeps_reachable_dates_only():
    date12_list = [
        '20200101_20200113',
        '20200113_20200125',
        '20200206_20200218',  # disconnected island
    ]
    edges = get_date_edges(DATES, date12_list)
    connected, keep = get_ref_connected_component(len(DATES), edges, ref_ind=0, min_redundancy=1.0)
    assert connected.tolist() == [True, True, True, False, False]
    assert keep.tolist() == [True, True, False]


def test_noncontiguous_but_ref_connected_dates_are_inverted():
    # date2 is missing, but date0-date1 and date1-date3 keep the later date reachable
    date12_list = ['20200101_20200113', '20200113_20200206']
    y = np.array([0.12, 0.24], dtype=np.float32)
    ts, quality, nobs = _invert(date12_list, y, ref_date='20200101')
    ts = ts.flatten()
    assert np.isclose(ts[0], 0.0, atol=1e-5)
    assert np.isfinite(ts[1])
    assert np.isnan(ts[2])
    assert np.isfinite(ts[3])
    assert np.isnan(ts[4])
    assert nobs == 2
    assert quality > 0.99


def test_disconnected_island_is_nan_and_does_not_pollute_ref_component():
    date12_list = [
        '20200101_20200113',
        '20200113_20200125',
        '20200206_20200218',
    ]
    y = np.array([0.1, 0.1, 5.0], dtype=np.float32)
    ts, quality, nobs = _invert(date12_list, y, ref_date='20200101')
    ts = ts.flatten()
    assert np.isclose(ts[0], 0.0, atol=1e-5)
    assert np.isfinite(ts[1]) and np.isfinite(ts[2])
    assert np.isnan(ts[3]) and np.isnan(ts[4])
    assert nobs == 2


def test_explicit_non_first_ref_date():
    date12_list = [
        '20200101_20200113',
        '20200113_20200125',
        '20200125_20200206',
    ]
    y = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    ts, _, nobs = _invert(date12_list, y, ref_date='20200125')
    ts = ts.flatten()
    assert np.isclose(ts[2], 0.0, atol=1e-5)
    assert np.isfinite(ts[0]) and np.isfinite(ts[1]) and np.isfinite(ts[3])
    assert np.isnan(ts[4]) is False or True  # date4 not observed -> nan
    assert np.isnan(ts[4])
    assert nobs == 3


def test_no_edge_to_reference_returns_all_nan():
    date12_list = ['20200206_20200218']
    y = np.array([0.3], dtype=np.float32)
    ts, quality, nobs = _invert(date12_list, y, ref_date='20200101')
    assert np.all(np.isnan(ts))
    assert nobs == 0
    assert quality == 0.0


def test_min_redundancy_drops_related_observations():
    date12_list = [
        '20200101_20200113',
        '20200101_20200125',
        '20200113_20200125',
        '20200125_20200206',  # date3 has degree 1 only
    ]
    y = np.array([0.1, 0.2, 0.1, 0.5], dtype=np.float32)
    ts, _, nobs = _invert(date12_list, y, min_redundancy=2.0)
    ts = ts.flatten()
    assert np.isnan(ts[3])
    assert nobs == 3


def test_min_norm_phase_partial_network():
    date12_list = ['20200101_20200113', '20200113_20200125']
    y = np.array([0.2, 0.3], dtype=np.float32)
    ts, _, nobs = _invert(date12_list, y, min_norm_velocity=False)
    ts = ts.flatten()
    assert np.isclose(ts[0], 0.0, atol=1e-5)
    assert np.isclose(ts[1], 0.2, atol=1e-5)
    assert np.isclose(ts[2], 0.5, atol=1e-5)
    assert np.isnan(ts[3]) and np.isnan(ts[4])
    assert nobs == 2


def test_weighted_partial_network_matches_unweighted_for_uniform_weights():
    date12_list = ['20200101_20200113', '20200113_20200125']
    y = np.array([0.2, 0.3], dtype=np.float32)
    w = np.ones_like(y)
    ts1, _, n1 = _invert(date12_list, y)
    ts2, _, n2 = _invert(date12_list, y, weight_sqrt=w)
    assert np.allclose(ts1, ts2, equal_nan=True, atol=1e-5)
    assert n1 == n2 == 2


def test_partial_covariance_marks_disconnected_dates_nan():
    date12_list = [
        '20200101_20200113',
        '20200113_20200125',
        '20200206_20200218',
    ]
    y = np.array([0.1, 0.1, 0.5], dtype=np.float32)
    y_std = np.ones_like(y) * 0.05
    # G is unused by the partial path; pass a placeholder matching output size.
    G = np.zeros((len(date12_list), len(DATES) - 1), dtype=np.float32)
    cov = estimate_timeseries_cov(
        G, y, y_std,
        allow_partial_network=True,
        date_list=DATES,
        date12_list=date12_list,
        ref_date='20200101',
        tbase=TBASE,
    )
    # non-ref dates are [1,2,3,4]; indices 2 and 3 correspond to disconnected dates 3 and 4
    assert cov.shape == (4, 4)
    assert np.all(np.isnan(cov[2, :]))
    assert np.all(np.isnan(cov[:, 2]))
    assert np.all(np.isnan(cov[3, :]))
    assert np.isfinite(cov[0, 0]) and np.isfinite(cov[1, 1])


def test_build_partial_design_matrix_velocity_skips_missing_date_intervals():
    date12_list = ['20200101_20200113', '20200113_20200206']
    edges = get_date_edges(DATES, date12_list)
    connected = np.array([True, True, False, True, False])
    keep = np.array([True, True])
    G, local_inds, tbase_diff, local_ref, rows = build_partial_design_matrix(
        edges, TBASE, connected, keep, ref_ind=0, min_norm_velocity=True)
    assert local_inds.tolist() == [0, 1, 3]
    assert G.shape == (2, 2)
    assert tbase_diff.shape == (2, 1)
    assert local_ref == 0
    assert rows.tolist() == [0, 1]


def main():
    test_ref_connected_component_keeps_reachable_dates_only()
    test_noncontiguous_but_ref_connected_dates_are_inverted()
    test_disconnected_island_is_nan_and_does_not_pollute_ref_component()
    test_explicit_non_first_ref_date()
    test_no_edge_to_reference_returns_all_nan()
    test_min_redundancy_drops_related_observations()
    test_min_norm_phase_partial_network()
    test_weighted_partial_network_matches_unweighted_for_uniform_weights()
    test_partial_covariance_marks_disconnected_dates_nan()
    test_build_partial_design_matrix_velocity_skips_missing_date_intervals()
    print('Pass.')


if __name__ == '__main__':
    main()
