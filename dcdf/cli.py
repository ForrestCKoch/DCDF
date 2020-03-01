import argparse
import os
from typing import Any, Optional, List

import numpy as np

from dcdf.data import get_reference_cdf, save_reference, load_reference, get_null_reference_cdf
from dcdf.measure import get_func_dict, measure_subjects, print_measurements
from dcdf.parallel import parallel_measure_subjects

def main():
    """
    This function is called on startup, and consists of the following steps:
    1. Construct our parser, validate commandline arguments, and retrieve arguments.
    2. Construct a filter based on any specified bounds.
    3. Build reference as specified.
    4. Write reference (if requested).
    5. Evaluate samples (if requested) -- run in parallel if requested.
    6. Print results.
    """
    parser = _build_parser() # Get our parser
    _validate_args(parser) # Validate our arguments
    args = parser.parse_args() # Parse arguments
    
    filter = _get_bounds_filter(args) # Construct a filter based on any specified bounds

    # Handle the various use cases for this program
    if args.build is not None: # User wants to construct a reference
        reference = get_reference_cdf(
            reference_list=_get_list(args.build,args.from_file),
            numbins=args.bins,
            indv_mask_list=_get_list(args.reference_masks,args.from_file),
            group_mask_filename=args.group_mask,
            filter=filter,
            lowerlimit=args.lower_limit,
            upperlimit=args.upper_limit
        )
        if args.output is not None: # User wants to save the reference
            save_reference(reference,args.output)
    elif args.load is not None:
        reference = load_reference(args.load)
    else:
        reference = get_null_reference_cdf(
            lowerlimit=args.lower_limit,
            upperlimit=args.upper_limit,
            numbins=args.bins
        )
                                    

    if args.evaluate is not None: # User wants to evaluate subjects against a reference
        if args.parallel:
            results = parallel_measure_subjects(
                    subjects_list=_get_list(args.evaluate,args.from_file),
                    reference=reference,
                    func_dict=get_func_dict(args.equations),
                    indv_mask_list=_get_list(args.evaluation_masks,args.from_file),
                    group_mask_filename=args.group_mask,
                    filter=filter,
                    n_procs=args.cores
            )
        else:
            results = measure_subjects(
                    subjects_list=_get_list(args.evaluate,args.from_file),
                    reference=reference,
                    func_dict=get_func_dict(args.equations),
                    indv_mask_list=_get_list(args.evaluation_masks,args.from_file),
                    group_mask_filename=args.group_mask,
                    filter=filter
            )
        print_measurements(results)

def _get_list(arg: List[str],from_file: bool) -> List[str]:
    """
    Helper function to handle the from_file = True/False usecases.

    :param arg: either a list of nifti filenames, or a list with a single 
        entry to a textfile (of filenames)
    :param from_file: if False, arg is a list of filenames, if true, it is a 
        list with a single entry to a textfile

    :returns: List of filenames
    """
    if from_file and arg is not None:
        with open(arg[0]) as fh:
           return [x.rstrip('\n') for x in fh] 
    else:
        return arg

def _build_parser() -> argparse.ArgumentParser:
    """
    Build the parser for commandline arguments.

    :returns: returns an argument parser for the CLI
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
        "-f",
        "--from-file",
        action='store_true',
        help="If set, '--build', '--evaluate', '--evaluation_masks', and '--reference_masks' will expect a path to a textfile indicating paths to subjects (one per line)."
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
        help='enforce a lower bound on considered values [Note: This should be the same for both reference and evaluation calls]'
    )
    parser.add_argument(
        "-U",
        "--upper-limit",
        type=float,
        action='store',
        default=None,
        help='enforce an upper bound on considered values [Note: This should be the same for both reference and evaluation calls]'
    )
    parser.add_argument(
        "-Q",
        "--quantile-filter",
        action='store_true',
        help="Apply a quantile filter instead of a bounds filter."
    )
    parser.add_argument(
        "-M",
        "--lower-quantile-limit",
        type=float,
        action='store',
        default=0.0,
        help='Lower quantile bound.'
    )
    parser.add_argument(
        "-S",
        "--upper-quantile-limit",
        type=float,
        action='store',
        default=1.0,
        help='Upper quantile bound.'
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

    parser.add_argument(
        "-p",
        "--parallel",
        action='store_true',
        help="Whether to perform evaluation using multiple processes"
    )

    parser.add_argument(
        "-c",
        "--cores",
        type=int,
        default=None,
        help="Number of cores to use.  Only used if '-p' is set. If None, defaults to number of available cores"
    )

    parser.add_argument(
        "-q",
        "--equations",
        type=str,
        default=None,
        help="Path to file declaring expressions which should be evaluated."
    )

    """ Just have this be the implicit default
    parser.add_argument(
        "-N",
        "--no-reference",
        action='store_true',
        help='Specify this option to calculate without reference'
    )
    """

    return parser

def _lc(filename: str) -> int:
    """
    Helper function to count the number of lines in a file.

    :param filename: name of the file to be read -- assumed to be plain text.

    :returns: count of the number of lines in file 
    """
    c = None
    with open(filename,'r') as fh:
        c = 0
        for line in fh:
            c += 1
    return c

def _validate_args(parser: argparse.ArgumentParser) -> bool:
    """
    Sanity checking on the inputs.  Returns False if any checks fail.

    :param parser: the parser returned from `_build_parser`

    :returns: False if any of the arguments fail the sanity checks, else True.
    """
    args = parser.parse_args()

    if (args.equations is None):
        parser.error('Please specify equations file ...')
        return False
    elif not os.path.exists(args.equations):
        parser.error('Path does not exist: {}'.format(args.equations))


    if (args.build is None) and (args.evaluate is None):
        parser.error('Please specify at least one of {--build, --evaluate}')
        return False
    
    if args.evaluate is not None:
        if not _check_nifti(args.evaluate,args.from_file):
            parser.error('Invalid file list passed to --evaluate')
            return False
        """ Removed so that program just defaults to using no reference
        if (args.build is None) == (args.load is None) and (not args.no_reference):
            parser.error('Please specify exactly one of {--build, --load, --no-reference}')
            return False
        """

    if args.build is not None:
        if not _check_nifti(args.build,args.from_file):
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
        if not _check_nifti(args.reference_masks,args.from_file):
            parser.error('Invalid file list passed to --reference-masks')
        if args.build is None:
            parser.error('Reference masks where specified without specifying reference scans!')
        if args.from_file and _lc(args.build[0]) != _lc(args.reference_masks[0]):
            parser.error('Number of reference masks does not match number of reference scans!')
        elif len(args.build) != len(args.reference_masks):
            parser.error('Number of reference masks does not match number of reference scans!')

    if args.evaluation_masks is not None:
        if not _check_nifti(args.evaluation_masks,args.from_file):
            parser.error('Invalid file list passed to --evaluation-masks')
        if args.evaluate is None:
            parser.error('Evaluation masks where specified without specifying evaluation scans!')
        if args.from_file and _lc(args.evaluation_masks[0]) != _lc(args.evaluate[0]):
            parser.error('Number of reference masks does not match number of reference scans!')
        elif len(args.evaluate) != len(args.evaluation_masks):
            parser.error('Number of reference masks does not match number of reference scans!')

    if args.quantile_filter:
        if args.lower_quantile_limit and args.lower_quantile_limit < 0:
            parser.error('Invalid limit range.')
        elif args.lower_quantile_limit and args.lower_quantile_limit > 1:
            parser.error('Invalid limit range.')
        elif args.upper_quantile_limit and args.upper_quantile_limit < 0:
            parser.error('Invalid limit range.')
        elif args.upper_quantile_limit and args.upper_quantile_limit > 1:
            parser.error('Invalid limit range.')
        elif args.lower_quantile_limit and args.upper_quantile_limit and args.upper_quantile_limit < args.lower_quantile_limit:
            parser.error('Invalid limit range.')

    return True

def _check_nifti(file_list: List[str], from_file: Optional[bool]=False) -> bool:
    """
    Check whether each of the nifti files can be found. 

    :param file_list: Should be one of: args.build, args.evaluate, 
        arg.reference_masks, args.evaluation_masks, args.group_mask.
        Where args are the parsed arguments.
    :param from_file: Whether the arguments have been supplied as text files.

    :returns: True if each of the nifti images can be found on disk.
    """
    # If we have been provided a file instead of explicit list use that
    if from_file:
        with open(file_list[0],'r') as fh:
            for line in fh:
                if not os.path.exists(line.rstrip('\n')):
                    return False
    # Case where list has been provided
    else:
        for f in file_list:
            if not os.path.exists(f):
                return False
    return True

def _get_bounds_filter(args):
    """
    If lower/upper bounds have been specified by the arguments, then provide a filter
    to be applied to the data.

    :param args: The parsed arguments to this program.
    """
    if (args.upper_limit is None) and (args.lower_limit is None):
        return None

    if args.quantile_filter:
        if args.lower_quantile_limit is None:
            lb = 0.0
        else:
            lb = args.lower_quantile_limit
        if args.upper_quantile_limit is None:
            ub = 1.0
        else:
            ub = args.upper_quantile_limit

        return lambda x: x[(x>np.quantile(x,lb))&(x<np.quantile(x,ub))]
        

    else:
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
