#!/usr/bin/env python3
"""Correct InSAR LOS velocity with a GNSS residual ramp."""

import os
import sys

from mintpy.utils.arg_utils import create_argument_parser

# duplicated in mintpy.objects.ramp to avoid importing mintpy.objects.__init__
RAMP_LIST = [
    'linear',
    'linear_range',
    'linear_azimuth',
    'quadratic',
    'quadratic_range',
    'quadratic_azimuth',
    'quadratic_azimuth_linear_range',
    'cubic',
    'cubic_azimuth_linear_range',
]


EXAMPLE = """example:
  # Franklin and Huang (2022): fit linear ramp to InSAR-GNSS LOS residuals
  correct_gnss_ramp.py geo_velocity.h5 gnss_enu2los_TGM.csv -m geo_maskTempCoh.h5
  correct_gnss_ramp.py geo_velocity.h5 gnss_enu2los_TGM.csv -m geo_maskTempCoh.h5 --ex-gnss GS91 CHIN S016
  correct_gnss_ramp.py geo_velocity.h5 gnss_enu2los_TGM.csv -m geo_maskTempCoh.h5 --radius 0   # nearest pixel

  # then compare:
  #   plot_insar_vs_gnss_scatter(vel_file='geo_velocity_gnssRamp.h5', csv_file='gnss_enu2los_TGM.csv', ...)
"""

REFERENCE = """reference:
  Franklin, K. R., and Huang, M.-H. (2022), Revealing crustal deformation and strain rate
    in Taiwan using InSAR and GNSS, Geophys. Res. Lett., 49, e2022GL101306.
    Supporting Information S1: ramp fit to InSAR-GNSS velocity differences.
"""


def create_parser(subparsers=None):
    synopsis = 'Correct InSAR velocity by removing a ramp fit to GNSS LOS residuals.'
    epilog = REFERENCE + '\n' + EXAMPLE
    name = __name__.split('.')[-1]
    parser = create_argument_parser(
        name, synopsis=synopsis, description=synopsis, epilog=epilog, subparsers=subparsers)

    parser.add_argument('vel_file', help='InSAR velocity file, e.g. geo_velocity.h5')
    parser.add_argument('csv_file', help='GNSS LOS CSV from view.py --gnss-comp (m/yr)')
    parser.add_argument('-m', '--mask', dest='msk_file',
                        help='Mask file for valid InSAR pixels, e.g. geo_maskTempCoh.h5')
    parser.add_argument('-o', '--outfile', dest='outfile',
                        help='Output velocity file. Default: add _gnssRamp to the input name.')
    parser.add_argument('-s', dest='ramp_type', default='linear', choices=RAMP_LIST,
                        help='Ramp type (default: %(default)s). linear = constant + x + y.')
    parser.add_argument('--radius', dest='radius', type=float, default=500.0, metavar='M',
                        help='Averaging radius in meters around each GNSS site (default: %(default)s).\n'
                             '0 uses the nearest pixel only.')
    parser.add_argument('--ex-gnss', dest='ex_gnss_sites', nargs='*', metavar='SITE',
                        help='GNSS sites excluded from the ramp fit.')
    parser.add_argument('-d', '--dset', dest='dset', default='velocity',
                        help='Dataset name in the velocity file (default: %(default)s).')
    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    if not os.path.isfile(inps.vel_file):
        raise FileNotFoundError(f'velocity file not found: {inps.vel_file}')
    if not os.path.isfile(inps.csv_file):
        raise FileNotFoundError(f'GNSS CSV file not found: {inps.csv_file}')
    if inps.msk_file and not os.path.isfile(inps.msk_file):
        raise FileNotFoundError(f'mask file not found: {inps.msk_file}')
    return inps


def main(iargs=None):
    inps = cmd_line_parse(iargs)
    from mintpy.correct_gnss_ramp import correct_velocity_with_gnss_ramp
    correct_velocity_with_gnss_ramp(
        vel_file=inps.vel_file,
        csv_file=inps.csv_file,
        msk_file=inps.msk_file,
        outfile=inps.outfile,
        ramp_type=inps.ramp_type,
        radius=inps.radius,
        ex_gnss_sites=inps.ex_gnss_sites,
        dset=inps.dset,
    )


if __name__ == '__main__':
    main(sys.argv[1:])
