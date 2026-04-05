#!/usr/bin/env python3
"""
Convert UNR GNSS time series from IGS20 (ITRF2020) to IGS14 (ITRF2014) and output
LOS velocity/displacement CSV in the same format as view.py --gnss-comp.

Usage:
  cd /path/to/mintpy/geo
  gnss_igs20_to_igs14.py --gnss-dir /path/to/mintpy/geo/GNSS-UNR --geo-dir /path/to/mintpy/geo

  # with date range (default: from velocity.h5 or geometry in geo dir)
  gnss_igs20_to_igs14.py --gnss-dir ./GNSS-UNR --geo-dir . --start-date 20150101 --end-date 20231201

Output: gnss_enu2los_UNR_IGS14.csv (Site, Lon, Lat, Displacement, Velocity) in geo dir.
"""

import argparse
import glob
import os

import numpy as np

from mintpy.objects import gnss
from mintpy.utils import readfile, utils as ut
from mintpy.utils import ptime, time_func
from mintpy.utils.utils0 import get_unit_vector4component_of_interest
from mintpy.utils.itrf_tools import (
    transform_enu_itrf2020_to_itrf2014,
    decimal_year,
)


def get_geom_obj(geo_dir):
    """Resolve geometry (inc/az) from geo dir: geometry file or velocity.h5 metadata."""
    geom_file = ut.get_geometry_file(
        ['incidenceAngle', 'azimuthAngle'],
        work_dir=geo_dir,
        coord='geo',
        print_msg=False,
    )
    if geom_file:
        return geom_file
    # fallback: velocity.h5 metadata (mean inc/az)
    vel_files = glob.glob(os.path.join(geo_dir, 'geo_velocity.h5')) + \
                glob.glob(os.path.join(geo_dir, 'velocity.h5'))
    if vel_files:
        return readfile.read_attribute(vel_files[0])
    raise FileNotFoundError(
        f'No geometry file or velocity.h5 found in {geo_dir}. '
        'Run with --geom-file /path/to/geometry.h5 if needed.'
    )


def run(gnss_dir, geo_dir, output_csv=None, start_date=None, end_date=None, model=None, unit_scale=1.0, verbose=False):
    """
    Convert all UNR *.tenv3 in gnss_dir from IGS20 to IGS14, project to LOS, write CSV.

    Parameters
    ----------
    gnss_dir : str
        Directory containing UNR .tenv3 files (e.g. GNSS-UNR).
    geo_dir : str
        Geo directory with geometry or velocity.h5 for inc/az.
    output_csv : str, optional
        Output CSV path. Default: geo_dir/gnss_enu2los_UNR_IGS14.csv
    start_date : str, optional
        YYYYMMDD.
    end_date : str, optional
        YYYYMMDD.
    model : dict, optional
        Time function for velocity fit. Default: {'polynomial': 1}.
    unit_scale : float, optional
        Scale factor for E/N/U from file (default 1 = meters). Use 0.001 if file is in mm.
    verbose : bool, optional
        Print debug info (first site: raw E/N/U range, inc/az, n_pts).

    Returns
    -------
    str
        Path to written CSV.
    """
    if output_csv is None:
        output_csv = os.path.join(geo_dir, 'gnss_enu2los_UNR_IGS14.csv')
    if model is None:
        model = {'polynomial': 1}

    geom_obj = get_geom_obj(geo_dir)

    # List sites: *.tenv3, exclude IGS08 (SITE.IGS08.tenv3)
    pattern = os.path.join(gnss_dir, '*.tenv3')
    files = glob.glob(pattern)
    site_names = []
    for f in files:
        name = os.path.basename(f)
        if name.endswith('.tenv3') and '.IGS08' not in name:
            site_names.append(name.replace('.tenv3', ''))

    if not site_names:
        raise FileNotFoundError(f'No *.tenv3 (non-IGS08) files found in {gnss_dir}')

    print(f'Found {len(site_names)} sites in {gnss_dir}')
    print(f'Geometry: {geom_obj if isinstance(geom_obj, str) else "metadata"}')
    print(f'Output: {output_csv}')

    # Get date range from geometry/velocity if not given
    if start_date is None or end_date is None:
        if isinstance(geom_obj, str):
            atr = readfile.read_attribute(geom_obj)
        else:
            atr = geom_obj
        start_date = start_date or atr.get('START_DATE') or atr.get('FIRST_FRAME_DATE')
        end_date = end_date or atr.get('END_DATE') or atr.get('LAST_FRAME_DATE')
    if not start_date or not end_date:
        raise ValueError('Could not infer start_date/end_date. Pass --start-date and --end-date.')

    rows = []
    for site in site_names:
        try:
            obj = gnss.GNSS_UNR(site, data_dir=gnss_dir, version='IGS20')
            if not os.path.isfile(obj.file):
                print(f'  Skip {site}: file not found {obj.file}')
                continue
            obj.get_site_lat_lon(print_msg=False)
            obj.read_displacement(start_date=start_date, end_date=end_date, print_msg=False)
        except Exception as e:
            print(f'  Skip {site}: {e}')
            continue

        lat, lon = obj.site_lat, obj.site_lon
        dates = obj.dates
        dis_e = np.asarray(obj.dis_e, dtype=np.float64) * unit_scale
        dis_n = np.asarray(obj.dis_n, dtype=np.float64) * unit_scale
        dis_u = np.asarray(obj.dis_u, dtype=np.float64) * unit_scale

        if dates is None or len(dates) < 2:
            print(f'  Skip {site}: insufficient dates')
            continue

        # Debug: first site only
        if verbose and len(rows) == 0:
            print(f'  [debug] {site}: n_pts={len(dates)}, '
                  f'E=[{dis_e.min():.4f}, {dis_e.max():.4f}] m, '
                  f'N=[{dis_n.min():.4f}, {dis_n.max():.4f}] m, '
                  f'U=[{dis_u.min():.4f}, {dis_u.max():.4f}] m')

        # Convert ENU displacement at each epoch to IGS14
        e14 = np.zeros_like(dis_e, dtype=np.float64)
        n14 = np.zeros_like(dis_n, dtype=np.float64)
        u14 = np.zeros_like(dis_u, dtype=np.float64)
        for i in range(len(dates)):
            epoch = decimal_year(dates[i])
            e14[i], n14[i], u14[i] = transform_enu_itrf2020_to_itrf2014(
                lat, lon, dis_e[i], dis_n[i], dis_u[i], epoch=epoch
            )

        # LOS geometry at site
        inc_angle, az_angle = obj.get_los_geometry(geom_obj, print_msg=False)
        unit_vec = get_unit_vector4component_of_interest(
            np.array(inc_angle), np.array(az_angle), comp='enu2los'
        )
        uv = [float(u) if np.isscalar(u) or (hasattr(u, 'size') and u.size == 1) else u for u in unit_vec]

        # Project to LOS
        dis_los = (e14 * uv[0] + n14 * uv[1] + u14 * uv[2]).astype(np.float32)

        if verbose and len(rows) == 0:
            print(f'  [debug] {site}: inc={inc_angle:.2f} deg, az={az_angle:.2f} deg, '
                  f'uv=[{uv[0]:.4f}, {uv[1]:.4f}, {uv[2]:.4f}], '
                  f'dis_los=[{dis_los.min():.4f}, {dis_los.max():.4f}] m')

        # Velocity: design matrix (polynomial 1) and fit
        date_list = [d.strftime('%Y%m%d') for d in dates]
        A = time_func.get_design_matrix4time_func(date_list, model=model)
        try:
            vel_los = np.dot(np.linalg.pinv(A), dis_los)[1]  # m/yr
        except Exception:
            vel_los = np.nan
        disp_los = float(dis_los[-1] - dis_los[0]) if len(dis_los) > 0 else np.nan

        rows.append([site, float(lon), float(lat), disp_los, vel_los])
        print(f'  {site}: vel_los = {vel_los*100:.3f} cm/yr, disp = {disp_los*100:.2f} cm')

    if not rows:
        raise RuntimeError('No site succeeded. Check GNSS dir and geometry.')

    # Write CSV (same columns as MintPy gnss_enu2los_*.csv)
    col_names = ['Site', 'Lon', 'Lat', 'Displacement', 'Velocity']
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or '.', exist_ok=True)
    with open(output_csv, 'w') as f:
        f.write(','.join(col_names) + '\n')
        for row in rows:
            f.write(','.join(str(x) for x in row) + '\n')
    print(f'Wrote {len(rows)} sites to {output_csv}')
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description='Convert UNR GNSS from IGS20 to IGS14 and output gnss_enu2los_*_IGS14.csv',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--gnss-dir', required=True,
                        help='Directory containing UNR .tenv3 files (e.g. geo/GNSS-UNR)')
    parser.add_argument('--geo-dir', default=None,
                        help='Geo directory for geometry/velocity (default: dir of --output or gnss-dir parent)')
    parser.add_argument('--output', '-o', dest='output_csv', default=None,
                        help='Output CSV path (default: geo_dir/gnss_enu2los_UNR_IGS14.csv)')
    parser.add_argument('--start-date', type=str, metavar='YYYYMMDD', default=None,
                        help='Start date for GNSS time series')
    parser.add_argument('--end-date', type=str, metavar='YYYYMMDD', default=None,
                        help='End date for GNSS time series')
    parser.add_argument('--unit-scale', type=float, default=1.0,
                        help='Scale E/N/U from file: 1=meters (default), 0.001=millimeters')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print debug info for first site (E/N/U range, inc/az, dis_los)')
    args = parser.parse_args()

    gnss_dir = os.path.abspath(args.gnss_dir)
    if not os.path.isdir(gnss_dir):
        parser.error(f'--gnss-dir is not a directory: {gnss_dir}')

    geo_dir = args.geo_dir
    if geo_dir is None:
        geo_dir = os.path.dirname(gnss_dir)
    geo_dir = os.path.abspath(geo_dir)

    run(
        gnss_dir=gnss_dir,
        geo_dir=geo_dir,
        output_csv=args.output_csv,
        start_date=args.start_date,
        end_date=args.end_date,
        unit_scale=args.unit_scale,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
