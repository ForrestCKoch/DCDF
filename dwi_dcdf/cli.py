import argparse
import os
from typing import Any, Optional

from data import get_reference_cdf, save_reference, load_reference
import numpy as np

def main():
    parser = _build_parser()
    _validate_args(parser)
    args = parser.parse_args()
    
    if args.build is not None:
        reference = get_reference_cdf(
            reference_list=args.build,
            numbins=args.bins,
            indv_mask_list=args.reference_masks,
            group_mask_filename=args.group_mask,
            filter=_get_bounds_filter(args)
        )
        if args.output is not None:
            save_reference(reference,args.output)
    else:
        reference = load_reference(args.load)
     

    pass

def _build_parser() -> argparse.ArgumentParser:
    """
    Returns returns an argument parser for the CLI
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-b",
        "--build",
        nargs='+',
        type=str,
        action='store',
        default=None,
        help='Builds a reference density from the provided files'
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        action='store',
        default=None,
        help='Output file for the reference'
    )

    parser.add_argument(
        "-e",
        "--evaluate",
        nargs='+',
        type=str,
        action='store',
        default=None,
        help='List of subjects to evaluate against reference'
    )

    parser.add_argument(
        "-l",
        "--load",
        type=str,
        action='store',
        default=None,
        help='Reference file to be loaded'
    )

    ########################################
    # Reference Building Parameters
    ########################################
    parser.add_argument(
        "-L",
        "--lower-limit",
        type=float,
        action='store',
        default=None,
        help='Lower bound for values when building reference'
    )
    parser.add_argument(
        "-U",
        "--upper-limit",
        type=float,
        action='store',
        default=None,
        help='Upper bound for values when building reference'
    )
    parser.add_argument(
        "-B",
        "--bins",
        type=int,
        action='store',
        default=5000,
        help='Number of bins to use when building reference'
    )
    parser.add_argument(
        "-R",
        "--reference-masks",
        nargs='+',
        type=str,
        default=None,
        help='List of masks to be applied to reference data (Optional)'
    )
    
    ########################################
    # Evaluation Parameters
    ########################################
    parser.add_argument(
        "-E",
        "--evaluation-masks",
        nargs='+',
        type=str,
        default=None,
        help='List of masks to be applied to evaluation data (Optional)'
    )

    ########################################
    # Other Parameters
    ########################################
    parser.add_argument(
        "-G",
        "--group-mask",
        type=str,
        default=None,
        help="Mask to be applied to all data.  Do not set if either --reference-masks, or --evaluation-masks are set"
    )

    return parser

def _validate_args(parser) -> bool:
    """
    Sanity checking on the inputs.  Returns False if any checks fail.
    """
    args = parser.parse_args()

    if (args.build is None) and (args.evaluate is None):
        parser.error('Please specify at least one of {--build, --evaluate}')
        return False
    
    if args.evaluate is not None:
        if not _check_nifti(args.evaluate):
            parser.error('Invalid file list passed to --evaluate')
            return False
        if (args.build is None) == (args.load is None):
            parser.error('Please specify exactly one of {--build, --load}')
            return False

    if args.build is not None:
        if not _check_nifti(args.build):
            parser.error('Invalid file list passed to --build')
            return False
        if (args.evaluate is None) and (args.output is None):
            parser.error('Please specify at least one of {--evaluate, --output}')
            return False

    if args.group_mask is not None:
        if not _check_nifti([args.group_mask]):
            parser.error('Invalid file passed to --group-mask')
        if args.reference_masks is not None:
            parser.error('Do not specify both --group-mask and --reference-masks')
        if args.evaluation_masks is not None:
            parser.error('Do not specify both --group-mask and --evaluation-masks')

    if args.reference_masks is not None:
        if not _check_nifti(args.reference_masks):
            parser.error('Invalid file list passed to --reference-masks')
        if args.build is None:
            parser.error('Reference masks where specified without specifying reference scans!')
        if len(args.build) != len(args.reference_masks):
            parser.error('Number of reference masks does not match number of reference scans!')

    if args.evaluation_masks is not None:
        if not _check_nifti(args.evaluation_masks):
            parser.error('Invalid file list passed to --evaluation-masks')
        if args.evaluate is None:
            parser.error('Evaluation masks where specified without specifying evaluation scans!')
        if len(args.evaluate) != len(args.evaluation_masks):
            parser.error('Number of reference masks does not match number of reference scans!')
    return True

def _check_nifti(file_list) -> bool:
    for f in file_list:
        if not os.path.exists(f):
            return False
    return True

def _get_bounds_filter(args):
    """
    If lower/upper bounds have been specified by the arguments, then provide a filter
    to be applied to the data.
    """
    if (args.upper_limit is None) and (args.lower_limit is None):
        return None

    if args.lower_limit is None:
        lb = -np.inf
    else:
        lb = args.lower_limit
    if args.upper_limit is None:
        ub = np.inf
    else:
        ub = args.upper_limit

    return lambda x: x[(x>lb)&(x<ub)]

if __name__ == '__main__':
    main()
