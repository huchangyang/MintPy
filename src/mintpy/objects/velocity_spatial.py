"""Spatially constrained linear velocity from per-pixel normal equations.

Used by timeseries2velocity when allowPartialDate and spatialSmooth are on.
Four-neighbor (azimuth + range) smoothness acts only on velocity; intercept
is unconstrained. Neighbor weights grow with coverage-set difference so
coverage-step stripes are pulled together while same-coverage gradients stay.
"""
############################################################
# Program is part of MintPy
############################################################

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg


def coverage_delta_n(valid_i, valid_j):
    """Relative symmetric-difference of two observation masks.

    ΔN = |S_i △ S_j| / max(|S_i ∪ S_j|, 1)
    """
    valid_i = np.asarray(valid_i, dtype=bool)
    valid_j = np.asarray(valid_j, dtype=bool)
    union = np.count_nonzero(valid_i | valid_j)
    symdiff = np.count_nonzero(valid_i != valid_j)
    return float(symdiff) / float(max(union, 1))


def neighbor_weight(valid_i, valid_j, alpha=1.0):
    """Coverage-difference weight: 1 + alpha * ΔN."""
    return 1.0 + float(alpha) * coverage_delta_n(valid_i, valid_j)


def four_neighbor_pairs(ys, xs, length, width):
    """Unique 4-neighbor estimable pairs (down and right), no diagonals.

    Parameters: ys, xs - 1D int arrays of estimable pixel coordinates
    Returns:    i_idx, j_idx - 1D int arrays of pair indices into ys/xs
    """
    ys = np.asarray(ys, dtype=np.int32)
    xs = np.asarray(xs, dtype=np.int32)
    n = ys.size
    if n == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    loc = np.full((int(length), int(width)), -1, dtype=np.int32)
    loc[ys, xs] = np.arange(n, dtype=np.int32)

    i_parts = []
    j_parts = []
    if length > 1:
        a = loc[:-1, :]
        b = loc[1:, :]
        ok = (a >= 0) & (b >= 0)
        i_parts.append(a[ok])
        j_parts.append(b[ok])
    if width > 1:
        a = loc[:, :-1]
        b = loc[:, 1:]
        ok = (a >= 0) & (b >= 0)
        i_parts.append(a[ok])
        j_parts.append(b[ok])

    if not i_parts:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    return np.concatenate(i_parts), np.concatenate(j_parts)


def _edge_weights(valid, i_idx, j_idx, alpha=1.0, chunk=50000):
    """Vectorized 1 + alpha * ΔN for neighbor pairs, chunked for memory."""
    n_edge = int(i_idx.size)
    w = np.empty(n_edge, dtype=np.float64)
    if n_edge == 0:
        return w
    for start in range(0, n_edge, chunk):
        stop = min(start + chunk, n_edge)
        vi = valid[i_idx[start:stop]]
        vj = valid[j_idx[start:stop]]
        sym = np.count_nonzero(vi != vj, axis=1)
        uni = np.count_nonzero(vi | vj, axis=1)
        delta = sym / np.maximum(uni, 1)
        w[start:stop] = 1.0 + float(alpha) * delta
    return w


def _cg_solve(H, rhs, x0):
    """Conjugate gradient with scipy API compatibility (rtol vs tol)."""
    n = int(rhs.size)
    maxiter = max(400, 8 * n)
    try:
        x, info = cg(H, rhs, x0=x0, atol=0.0, rtol=1e-8, maxiter=maxiter)
    except TypeError:
        x, info = cg(H, rhs, x0=x0, tol=1e-8, maxiter=maxiter)
    return x, info


def accumulate_linear_normal_eq(G, ts_data, pixel_y, pixel_x, min_obs=3):
    """Per-pixel 2x2 normal equations for a + v t, grouped by coverage mask.

    Parameters: G        - ndarray (num_date, 2), full-date design matrix
                ts_data  - ndarray (num_date, num_pixel)
                pixel_y  - ndarray (num_pixel,)
                pixel_x  - ndarray (num_pixel,)
                min_obs  - int, minimum finite dates (need > 2 for residue STD)
    Returns:    ys, xs, N, b, valid, e2, m_std
                N     - (n, 2, 2) float64
                b     - (n, 2) float64
                valid - (n, num_date) bool
                e2    - (n,) unconstrained residual sum of squares
                m_std - (n, 2) unconstrained parameter STD
    """
    G = np.asarray(G, dtype=np.float64)
    ts_data = np.asarray(ts_data)
    pixel_y = np.asarray(pixel_y, dtype=np.int32)
    pixel_x = np.asarray(pixel_x, dtype=np.int32)
    num_date = G.shape[0]

    empty = (
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.int32),
        np.zeros((0, 2, 2), dtype=np.float64),
        np.zeros((0, 2), dtype=np.float64),
        np.zeros((0, num_date), dtype=bool),
        np.zeros(0, dtype=np.float64),
        np.zeros((0, 2), dtype=np.float64),
    )
    if ts_data.size == 0 or G.shape[1] != 2:
        return empty

    valid = np.isfinite(ts_data)
    nobs = np.sum(valid, axis=0)
    keep = nobs >= int(min_obs)
    if not np.any(keep):
        return empty

    ts_data = np.asarray(ts_data[:, keep], dtype=np.float64)
    valid = valid[:, keep]
    pixel_y = pixel_y[keep]
    pixel_x = pixel_x[keep]

    packed = np.packbits(np.ascontiguousarray(valid.T), axis=1)
    _, inv = np.unique(packed, axis=0, return_inverse=True)

    ys_out, xs_out, N_out, b_out, valid_out, e2_out, std_out = [], [], [], [], [], [], []
    for cid in range(int(inv.max()) + 1):
        sel = inv == cid
        valid_i = valid[:, sel][:, 0]
        Gv = G[valid_i]
        if Gv.shape[0] < int(min_obs) or np.linalg.matrix_rank(Gv) < 2:
            continue
        N = Gv.T @ Gv
        d = ts_data[np.ix_(valid_i, sel)]
        b = Gv.T @ d
        try:
            mu = np.linalg.solve(N, b)
        except np.linalg.LinAlgError:
            continue
        resid = d - Gv @ mu
        e2 = np.sum(resid * resid, axis=0)
        dof = Gv.shape[0] - 2
        if dof <= 0:
            continue
        ginv_diag = np.diag(np.linalg.inv(N))
        m_std = np.sqrt(ginv_diag[:, None] * (e2 / dof)[None, :])

        ng = int(np.sum(sel))
        ys_out.append(pixel_y[sel])
        xs_out.append(pixel_x[sel])
        N_out.append(np.broadcast_to(N, (ng, 2, 2)).copy())
        b_out.append(b.T)
        valid_out.append(np.broadcast_to(valid_i, (ng, num_date)).copy())
        e2_out.append(e2)
        std_out.append(m_std.T)

    if not ys_out:
        return empty

    return (
        np.concatenate(ys_out),
        np.concatenate(xs_out),
        np.concatenate(N_out, axis=0),
        np.concatenate(b_out, axis=0),
        np.concatenate(valid_out, axis=0),
        np.concatenate(e2_out),
        np.concatenate(std_out, axis=0),
    )


def solve_constrained_velocity(N, b, valid, ys, xs, length, width,
                               strength=1.0, alpha=1.0, print_msg=True):
    """Solve blkdiag(N_i) + λ L^T W L on velocity, intercept unconstrained.

    λ = strength * median(N_vv). L is the 4-neighbor difference operator
    on the velocity component only (azimuth and range, same weights).

    Parameters: N, b, valid, ys, xs - from accumulate_linear_normal_eq
                strength - relative spatialSmoothStrength (default 1)
                alpha    - coverage-difference weight scale (default 1)
    Returns:    m     - (n, 2) intercept, velocity
                info  - scipy.cg info code
                lam   - scalar λ used
    """
    N = np.asarray(N, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    ys = np.asarray(ys, dtype=np.int32)
    xs = np.asarray(xs, dtype=np.int32)
    n = int(N.shape[0])
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64), 0, 0.0

    try:
        m0 = np.linalg.solve(N, b)
    except np.linalg.LinAlgError:
        m0 = np.zeros((n, 2), dtype=np.float64)
        for i in range(n):
            m0[i] = np.linalg.lstsq(N[i], b[i], rcond=None)[0]

    nvv = N[:, 1, 1]
    ok = np.isfinite(nvv) & (nvv > 0)
    med = float(np.median(nvv[ok])) if np.any(ok) else 1.0
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    lam = float(strength) * med

    i_idx, j_idx = four_neighbor_pairs(ys, xs, length, width)
    w = _edge_weights(valid, i_idx, j_idx, alpha=alpha)

    # blkdiag(N_i): four entries per pixel, C-order [N00, N01, N10, N11]
    pix = np.arange(n, dtype=np.int32)
    rows_n = np.repeat(pix * 2, 4) + np.tile(np.array([0, 0, 1, 1], dtype=np.int32), n)
    cols_n = np.repeat(pix * 2, 4) + np.tile(np.array([0, 1, 0, 1], dtype=np.int32), n)
    data_n = N.reshape(n, 4).ravel()

    if i_idx.size:
        iv = 2 * i_idx + 1
        jv = 2 * j_idx + 1
        val = lam * w
        rows_l = np.concatenate([iv, jv, iv, jv])
        cols_l = np.concatenate([iv, jv, jv, iv])
        data_l = np.concatenate([val, val, -val, -val])
        rows = np.concatenate([rows_n, rows_l])
        cols = np.concatenate([cols_n, cols_l])
        data = np.concatenate([data_n, data_l])
    else:
        rows, cols, data = rows_n, cols_n, data_n

    n_param = 2 * n
    H = coo_matrix((data, (rows, cols)), shape=(n_param, n_param)).tocsr()
    rhs = b.reshape(-1)
    x, info = _cg_solve(H, rhs, m0.reshape(-1))
    if print_msg:
        print(f'spatial smoothness: lambda={lam:.6g} (strength={strength} * median N_vv={med:.6g})')
        if info == 0:
            print('spatial smoothness: conjugate gradient converged')
        else:
            print(f'WARNING: conjugate gradient info={info} (0=converged); using last iterate')
        print('spatially constrained velocity is biased; '
              'velocityStd uses unconstrained residual approximation')

    m = np.asarray(x, dtype=np.float64).reshape(n, 2)
    return m, int(info), lam
