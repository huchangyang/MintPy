"""Correct InSAR velocity with a GNSS residual ramp (Franklin and Huang, 2022).

At GNSS sites, fit a 2-D ramp to (InSAR - GNSS) LOS velocity, then subtract
that ramp from the whole InSAR velocity field so InSAR is in the GNSS frame.

Recommend import:
    from mintpy.correct_gnss_ramp import correct_velocity_with_gnss_ramp
"""

import os

import numpy as np

from mintpy.objects.coord import coordinate
from mintpy.objects.ramp import RAMP_LIST
from mintpy.utils import readfile, writefile


def design_matrix_ramp(ys, xs, ramp_type='linear'):
    """Design matrix at sample locations. Column order matches mintpy.objects.ramp.deramp."""
    yy = np.asarray(ys, dtype=np.float32).reshape(-1, 1)
    xx = np.asarray(xs, dtype=np.float32).reshape(-1, 1)
    ones = np.ones(xx.shape, dtype=np.float32)
    if ramp_type == 'linear':
        G = np.hstack((yy, xx, ones))
    elif ramp_type == 'quadratic':
        G = np.hstack((yy**2, xx**2, yy * xx, yy, xx, ones))
    elif ramp_type == 'linear_range':
        G = np.hstack((xx, ones))
    elif ramp_type == 'linear_azimuth':
        G = np.hstack((yy, ones))
    elif ramp_type == 'quadratic_range':
        G = np.hstack((xx**2, xx, ones))
    elif ramp_type == 'quadratic_azimuth':
        G = np.hstack((yy**2, yy, ones))
    elif ramp_type == 'quadratic_azimuth_linear_range':
        G = np.hstack((yy**2, yy, xx, ones))
    elif ramp_type == 'cubic_azimuth_linear_range':
        G = np.hstack((yy**3, yy**2, yy, xx, ones))
    elif ramp_type == 'cubic':
        G = np.hstack((yy**3, yy**2, yy, xx**3, xx**2, xx, ones))
    else:
        raise ValueError(f'un-recognized ramp type: {ramp_type}. Choose from {RAMP_LIST}')
    return G


def fit_ramp_coeff(ys, xs, residual, ramp_type='linear'):
    """Least-squares ramp coefficients from residuals at sample pixels."""
    G = design_matrix_ramp(ys, xs, ramp_type)
    d = np.asarray(residual, dtype=np.float32).reshape(-1, 1)
    if G.shape[0] < G.shape[1]:
        raise ValueError(
            f'Need at least {G.shape[1]} GNSS sites to fit {ramp_type} ramp, got {G.shape[0]}'
        )
    return np.dot(np.linalg.pinv(G, rcond=1e-15), d).flatten()


def apply_ramp(data, coeff, ramp_type='linear'):
    """Subtract a ramp (defined by coeff) from a 2-D field."""
    length, width = data.shape[-2:]
    xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, length))
    G = design_matrix_ramp(yy.ravel(), xx.ravel(), ramp_type)
    ramp = np.dot(G, np.asarray(coeff, dtype=np.float32).reshape(-1, 1))
    ramp = np.array(ramp.reshape(length, width), dtype=data.dtype)
    return data - ramp


def pixel_size_m(atr, lat):
    """Return (y_size_m, x_size_m) from metadata."""
    x_step = abs(float(atr['X_STEP']))
    y_step = abs(float(atr['Y_STEP']))
    if x_step < 0.1:
        return y_step * 111.32e3, x_step * 111.32e3 * np.cos(np.deg2rad(lat))
    return y_step, x_step


def sample_insar_at_sites(data, msk, ys, xs, radius_pix=0):
    """Mean InSAR value in a circular window; radius_pix=0 uses the nearest pixel."""
    length, width = data.shape
    out = np.full(len(ys), np.nan, dtype=np.float32)
    r = int(radius_pix)
    for i, (y, x) in enumerate(zip(ys, xs)):
        y, x = int(y), int(x)
        if r <= 0:
            if 0 <= y < length and 0 <= x < width and msk[y, x] and np.isfinite(data[y, x]):
                out[i] = data[y, x]
            continue
        y0, y1 = max(0, y - r), min(length, y + r + 1)
        x0, x1 = max(0, x - r), min(width, x + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        circ = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
        patch = data[y0:y1, x0:x1]
        maskp = msk[y0:y1, x0:x1] != 0
        vals = patch[circ & maskp & np.isfinite(patch)]
        if vals.size:
            out[i] = np.mean(vals)
    return out


def _rmse(a, b):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size <= 1:
        return np.nan
    return float(np.sqrt(np.sum(d ** 2) / (d.size - 1)))


def correct_velocity_with_gnss_ramp(
        vel_file,
        csv_file,
        msk_file=None,
        outfile=None,
        ramp_type='linear',
        radius=500.0,
        ex_gnss_sites=None,
        dset='velocity',
        print_msg=True):
    """Fit a ramp to InSAR-GNSS LOS residuals and write a corrected velocity file.

    Parameters: vel_file      - str, InSAR velocity HDF5
                csv_file      - str, GNSS CSV from view.py --gnss-comp (Site,Lon,Lat,Disp,Vel in m/yr)
                msk_file      - str or None, mask for valid InSAR pixels
                outfile       - str or None, default: add _gnssRamp before suffix
                ramp_type     - str, default linear (constant + x + y), same names as remove_ramp.py
                radius        - float, averaging radius in meters around each GNSS site
                                (Franklin and Huang 2022 used 500 m). 0 = nearest pixel
                ex_gnss_sites - list of str, sites excluded from the ramp fit
                dset          - str, dataset name in vel_file
    Returns:    outfile       - str, path of the corrected velocity file
    """
    vprint = print if print_msg else lambda *args, **kwargs: None
    if ramp_type not in RAMP_LIST:
        raise ValueError(f'un-recognized ramp type: {ramp_type}. Choose from {RAMP_LIST}')

    # GNSS
    col_types = ['U10'] + ['f8'] * 4
    vprint(f'read GNSS LOS velocity from {csv_file}')
    fc = np.genfromtxt(csv_file, dtype=col_types, delimiter=',', names=True)
    sites = np.array(fc['Site'])
    lats = np.array(fc['Lat'], dtype=np.float64)
    lons = np.array(fc['Lon'], dtype=np.float64)
    gnss = np.array(fc['Velocity'], dtype=np.float32)

    if ex_gnss_sites:
        ex_flag = np.array([s in ex_gnss_sites for s in sites], dtype=bool)
        if np.sum(ex_flag) > 0:
            vprint(f'exclude {np.sum(ex_flag)} GNSS sites from ramp fit: {sites[ex_flag]}')
            keep = ~ex_flag
            sites, lats, lons, gnss = sites[keep], lats[keep], lons[keep], gnss[keep]

    # InSAR
    vprint(f'read InSAR velocity from {vel_file}')
    vel, atr = readfile.read(vel_file, datasetName=dset)
    length, width = int(atr['LENGTH']), int(atr['WIDTH'])
    msk = readfile.read(msk_file)[0] if msk_file else np.ones((length, width), dtype=np.bool_)
    msk = np.array(msk, dtype=bool)

    coord = coordinate(atr)
    ys, xs = coord.geo2radar(lats, lons, print_msg=False)[:2]
    ys = np.asarray(ys)
    xs = np.asarray(xs)

    lat0 = float(np.nanmean(lats)) if lats.size else 0.0
    y_m, x_m = pixel_size_m(atr, lat0)
    pix_m = float(np.sqrt(y_m * x_m))
    radius_pix = 0 if (radius is None or radius <= 0) else int(round(float(radius) / pix_m))
    vprint(f'GNSS sampling radius: {float(radius or 0):.0f} m ({radius_pix} pixel)')

    insar = sample_insar_at_sites(vel, msk, ys, xs, radius_pix=radius_pix)
    flag = np.isfinite(insar) & np.isfinite(gnss)
    n_drop = int(np.sum(~flag))
    if n_drop:
        vprint(f'drop {n_drop} sites with NaN InSAR or GNSS')
    sites, ys, xs, insar, gnss = sites[flag], ys[flag], xs[flag], insar[flag], gnss[flag]
    vprint(f'fit {ramp_type} ramp with {sites.size} GNSS sites')

    rmse0 = _rmse(insar, gnss)
    residual = insar - gnss
    coeff = fit_ramp_coeff(ys, xs, residual, ramp_type=ramp_type)
    vel_cor = apply_ramp(vel, coeff, ramp_type=ramp_type)
    insar_cor = sample_insar_at_sites(vel_cor, msk, ys, xs, radius_pix=radius_pix)
    rmse1 = _rmse(insar_cor, gnss)

    vprint(f'ramp coefficients ({ramp_type}): {np.array2string(coeff, precision=6)}')
    vprint(f'RMSE InSAR-GNSS before ramp: {rmse0*100:.2f} cm/yr')
    vprint(f'RMSE InSAR-GNSS after  ramp: {rmse1*100:.2f} cm/yr')

    if outfile is None:
        fbase, fext = os.path.splitext(vel_file)
        outfile = f'{fbase}_gnssRamp{fext}'

    atr = dict(atr)
    atr['mintpy.gnssRamp'] = ramp_type
    atr['mintpy.gnssRamp.csvFile'] = os.path.abspath(csv_file)
    atr['mintpy.gnssRamp.radius'] = str(radius)
    atr['mintpy.gnssRamp.coeff'] = ','.join(f'{c:.8g}' for c in coeff)

    ds_dict = {dset: np.array(vel_cor, dtype=np.float32)}
    vprint(f'write GNSS-ramp-corrected velocity to {outfile}')
    writefile.write(ds_dict, out_file=outfile, metadata=atr, ref_file=vel_file, print_msg=print_msg)
    return outfile
